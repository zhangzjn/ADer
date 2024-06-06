import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
import copy
import math
from model import get_model
from model import MODEL
from data import get_loader
import torch
import timm


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)
    
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)

        self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))


    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.ConvTranspose2d(in_channels=channels,
                                           out_channels=self.out_channels,
                                           kernel_size=4,
                                           stride=2, padding=1)
    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv:
            x = self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(
                 self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(nn.Module):

    def __init__(
        self,
        channels,
        out_channels=None,
        use_conv=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, True)
            self.x_upd = Upsample(channels, True)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                 channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d( channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        return self.skip_connection(x) + h



class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels,
        num_res_blocks,
        channel_mult,
        attention_mult,
        num_heads = 4,
        num_heads_upsample=-1,
        num_head_channels = 64,
    ):

        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.channel_mult = channel_mult

        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, ch, 3, padding=1)]
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        out_channels=int(mult * model_channels),
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_mult:
                    layers.append(
                                AttentionBlock(
                                    ch,
                                    num_heads=num_heads,
                                    num_head_channels=num_head_channels,
                            )
                    )

                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                        ResBlock(
                            ch,
                            out_channels=out_ch,
                            down=True,
                        )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch


        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        out_channels=int(model_channels * mult),
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_mult:
                    layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads_upsample,##
                                num_head_channels=num_head_channels,
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            out_channels=out_ch,
                            up=True,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )


    def forward(self, x):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h)
        return self.out(h)

class Residual(nn.Module):
    def __init__(self, in_channels):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=1, stride=1,bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_residual_layers):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Backbone(torch.nn.Module):
    def __init__(self,
                 backbone,
                 outlayers,
                 ckpt_path
                 ):

        super(Backbone, self).__init__()
        self.backbone=backbone
        self.outlayers=outlayers

        assert self.backbone in ['resnet18','resnet34','resnet50','efficientnet_b4','wide_resnet50_2']

        if self.backbone =='resnet50' or self.backbone=='wide_resnet50_2':
            layers_idx ={'layer1':1, 'layer2':2, 'layer3':3, 'layer4':4}
            layers_strides = {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}
            layers_planes= {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}

        elif self.backbone=='resnet34' or self.backbone=='resnet18':
            layers_idx = {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4}
            layers_strides = {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}
            layers_planes = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}

        elif self.backbone == 'efficientnet_b4':
            # if you use efficientnet_b4 as backbone, make sure timm==0.5.x, we use 0.5.4
            layers_idx = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3, 'layer5': 4}
            layers_strides = {'layer1': 2, 'layer2': 4, 'layer3': 8, 'layer4': 16, 'layer5': 32}
            layers_planes = {'layer1': 24, 'layer2': 32, 'layer3': 56, 'layer4': 160, 'layer5': 448}
        else:
            raise NotImplementedError("backbone must in [resnet18, resnet34,resnet50, wide_resnet50_2, efficientnet_b4]")

        self.feature_extractor = timm.create_model(self.backbone, features_only=True,pretrained=False,
                                                  out_indices=[layers_idx[outlayer] for outlayer in self.outlayers])
        # ckpt_path = 'model/pretrain/wide_resnet50_racm-8234f177.pth'
        self.feature_extractor.load_state_dict(torch.load(ckpt_path), strict=False)
        self.feature_extractor.eval()
        self.layers_strides=layers_strides
        self.layers_planes=layers_planes

    @torch.no_grad()
    def forward(self, inputs,train=False):
        feats_dict={}
        image = inputs["img"]
        feats=self.feature_extractor(image)

        feats= {self.outlayers[idx]:{
                "feat": feats[idx],
                "stride": self.layers_strides[self.outlayers[idx]],
                "planes": self.layers_planes[self.outlayers[idx]],
                } for idx in range(len(self.outlayers))}

        feats_dict.update({"feats":feats})
        if train:
            gt_image = inputs["gt_image"]
            gt_feats = self.feature_extractor(gt_image)
            gt_feats = {self.outlayers[idx]:{
                        "feat":gt_feats[idx],
                        "stride": self.layers_strides[self.outlayers[idx]],
                        "planes": self.layers_planes[self.outlayers[idx]],
                } for idx in range(len(self.outlayers))}

            feats_dict.update({"gt_feats": gt_feats})

        return feats_dict


    def get_outplanes(self):
        return { outlayer: self.layers_planes[outlayer] for outlayer in self.outlayers}

    def get_outstrides(self):
        return { outlayer: self.layers_strides[outlayer] for outlayer in self.outlayers}
    

class AFS(nn.Module):
    def __init__(self,
                 inplanes,
                 instrides,
                 structure,
                 init_bsn,
                 ):

        super(AFS, self).__init__()

        self.inplanes=inplanes
        self.instrides=instrides
        self.structure=structure
        self.init_bsn=init_bsn

        self.indexes=nn.ParameterDict()

        for block in self.structure:
            for layer in block['layers']:
                self.indexes["{}_{}".format(block['name'],layer['idx'])]=nn.Parameter(torch.zeros(layer['planes']).long(),requires_grad=False)
                self.add_module("{}_{}_upsample".format(block['name'],layer['idx']),
                                nn.UpsamplingBilinear2d(scale_factor=self.instrides[layer['idx']]/block['stride']))


    @torch.no_grad()
    def forward(self, inputs,train=False):
        block_feats = {}
        feats = inputs["feats"]
        for block in self.structure:
            block_feats[block['name']]=[]

            for layer in block['layers']:
                feat_c=torch.index_select(feats[layer['idx']]['feat'], 1, self.indexes["{}_{}".format(block['name'],layer['idx'])].data)
                feat_c=getattr(self,"{}_{}_upsample".format(block['name'],layer['idx']))(feat_c)
                block_feats[block['name']].append(feat_c)
            block_feats[block['name']]=torch.cat(block_feats[block['name']],dim=1)

        if train:
            gt_block_feats = {}
            gt_feats = inputs["gt_feats"]
            for block in self.structure:
                gt_block_feats[block['name']] = []
                for layer in block['layers']:
                    feat_c = torch.index_select(gt_feats[layer['idx']]['feat'], 1, self.indexes["{}_{}".format(block['name'], layer['idx'])].data)
                    feat_c = getattr(self, "{}_{}_upsample".format(block['name'], layer['idx']))(feat_c)
                    gt_block_feats[block['name']].append(feat_c)
                gt_block_feats[block['name']] = torch.cat(gt_block_feats[block['name']], dim=1)
            return {'block_feats':block_feats,"gt_block_feats":gt_block_feats}

        return {'block_feats':block_feats}



    def get_outplanes(self):
        return { block['name']:sum([layer['planes'] for layer in block['layers']])  for block in self.structure}

    def get_outstrides(self):
        return { block['name']:block['stride']  for block in self.structure}


    @torch.no_grad()
    def init_idxs(self, model, train_loader, distributed=True):
        anomaly_types = copy.deepcopy(train_loader.dataset.anomaly_types)

        if 'normal' in train_loader.dataset.anomaly_types:
            del train_loader.dataset.anomaly_types['normal']

        for key in train_loader.dataset.anomaly_types:
            train_loader.dataset.anomaly_types[key] = 1.0/len(list(train_loader.dataset.anomaly_types.keys()))

        model.eval()
        criterion = nn.MSELoss(reduce=False).cuda()
        for block in self.structure:
            self.init_block_idxs(block, model, train_loader, criterion,distributed=distributed)
        train_loader.dataset.anomaly_types = anomaly_types
        model.train()


    def init_block_idxs(self,block,model,train_loader,criterion,distributed=True):

        if distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            if rank == 0:
                tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name']))
            else:
                tq = range(self.init_bsn)
        else:
            tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name']))

        cri_sum_vec=[torch.zeros(self.inplanes[layer['idx']]).cuda() for layer in block['layers']]
        iterator = iter(train_loader)

        for bs_i in tq:
            try:
                input = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                input = next(iterator)

            for key, input_value in input.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input[key] = input_value.cuda()
                    
            bb_feats = model.net_backbone(input,train=True)

            ano_feats=bb_feats['feats']
            ori_feats=bb_feats['gt_feats']
            gt_mask = input['img_mask'].cuda()

            B= gt_mask.size(0)

            ori_layer_feats=[ori_feats[layer['idx']]['feat'] for layer in block['layers']]
            ano_layer_feats=[ano_feats[layer['idx']]['feat'] for layer in block['layers']]

            for i,(ano_layer_feat,ori_layer_feat) in enumerate(zip(ano_layer_feats,ori_layer_feats)):
                layer_name=block['layers'][i]['idx']

                C = ano_layer_feat.size(1)

                ano_layer_feat = getattr(self, "{}_{}_upsample".format(block['name'], layer_name))(ano_layer_feat)
                ori_layer_feat = getattr(self, "{}_{}_upsample".format(block['name'], layer_name))(ori_layer_feat)

                layer_pred = (ano_layer_feat - ori_layer_feat) ** 2

                _, _, H, W = layer_pred.size()

                layer_pred = layer_pred.permute(1, 0, 2, 3).contiguous().view(C, B * H * W)
                (min_v, _), (max_v, _) = torch.min(layer_pred, dim=1), torch.max(layer_pred, dim=1)
                layer_pred = (layer_pred - min_v.unsqueeze(1)) / (max_v.unsqueeze(1) - min_v.unsqueeze(1)+ 1e-4)

                label = F.interpolate(gt_mask, (H, W), mode='nearest')
                label = label.permute(1, 0, 2, 3).contiguous().view(1, B * H * W).repeat(C, 1)

                mse_loss = torch.mean(criterion(layer_pred, label), dim=1)

                if distributed:
                    mse_loss_list = [mse_loss for _ in range(world_size)]
                    dist.all_gather(mse_loss_list, mse_loss)
                    mse_loss = torch.mean(torch.stack(mse_loss_list,dim=0),dim=0,keepdim=False)

                cri_sum_vec[i] += mse_loss

        for i in range(len(cri_sum_vec)):
            cri_sum_vec[i][torch.isnan(cri_sum_vec[i])] = torch.max(cri_sum_vec[i][~torch.isnan(cri_sum_vec[i])])
            values, indices = torch.topk(cri_sum_vec[i], k=block['layers'][i]['planes'], dim=-1, largest=False)
            values, _ = torch.sort(indices)

            if distributed:
                tensor_list = [values for _ in range(world_size)]
                dist.all_gather(tensor_list, values)
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(tensor_list[0].long())
            else:
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(values.long())
                
class ReconstructionLayer(nn.Module):
    def __init__(self,
                 inplanes,
                 instrides,
                 num_res_blocks,
                 hide_channels_ratio,
                 channel_mult,
                 attention_mult
                 ):

        super(ReconstructionLayer, self).__init__()

        self.inplanes=inplanes
        self.instrides=instrides
        self.num_res_blocks=num_res_blocks
        self.attention_mult=attention_mult

        for block_name in self.inplanes:
            module= UNetModel(
                in_channels=self.inplanes[block_name],
                out_channels=self.inplanes[block_name],
                model_channels=int(hide_channels_ratio*self.inplanes[block_name]),
                channel_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                attention_mult=attention_mult
            )
            self.add_module('{}_recon'.format(block_name),module)


    def forward(self, inputs,train=False):
        block_feats = inputs['block_feats']
        recon_feats = { block_name:getattr(self,'{}_recon'.format(block_name))(block_feats[block_name]) for block_name in block_feats}
        residual={ block_name: (block_feats[block_name] - recon_feats[block_name] )**2 for block_name in block_feats}
        return {'feats_recon':recon_feats,'residual':residual}


    def get_outplanes(self):
        return self.inplanes

    def get_outstrides(self):
        return self.instrides

class RRS(torch.nn.Module):
    def __init__(self,
                 inplanes,
                 instrides,
                 modes,
                 mode_numbers,
                 num_residual_layers,
                 stop_grad,
                 ):

        super(RRS, self).__init__()

        self.inplanes=inplanes
        self.instrides=instrides
        self.mode_numbers = mode_numbers
        self.modes = modes
        self.num_residual_layers=num_residual_layers
        self.stop_grad=stop_grad

        self.total_select_number=sum(self.mode_numbers)

        align_stride = min([self.instrides[block] for block in self.instrides])

        for block in self.instrides:
            self.add_module("{}_upsample".format(block),nn.UpsamplingBilinear2d(scale_factor=self.instrides[block]/align_stride))

        align_inplane = sum([self.inplanes[block] for block in self.inplanes])

        self.bn_idx = nn.BatchNorm2d(align_inplane,momentum=0.9,affine=False)

        self.decoder1 = nn.Sequential(
            ResidualStack(self.total_select_number, self.num_residual_layers),
            nn.Conv2d(self.total_select_number, 128, (3, 3), padding=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(nn.Conv2d(128, 32, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 8, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder3 = nn.Sequential(nn.Conv2d(8, 4, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(4, 2, (3, 3), padding=(1, 1), bias=True))


    @torch.no_grad()
    def select_ano_index(self,residual,mode, k):
        B,C,W,H = residual.size()
        residual = residual.view((B,C,W*H))
        if mode=='max':
            residual,_ = torch.max(residual,dim=-1)
        elif mode=='mean':
            residual  = torch.mean(residual, dim=-1)
        else:
            raise ValueError("mode must in [max,mean]")
        _,idxs=torch.topk(residual,dim=1,largest=True,k=k,sorted=True)
        return idxs


    def forward(self, inputs, train=False):

        residual = inputs['residual']

        if self.stop_grad:
            residual= { block :residual[block].detach() for block in residual}

        residual = torch.cat([ getattr(self,"{}_upsample".format(block))(residual[block]) for block in residual],dim=1)

        residual_idx = self.bn_idx(residual)

        B, C, H, W = residual.size()

        residual_choose = []
        for mode, mode_n in zip(self.modes, self.mode_numbers):
            idxs = self.select_ano_index(residual_idx, mode, mode_n)
            residual_choose.append(
                torch.gather(residual, dim=1, index=idxs.view((B,mode_n,1,1)).repeat(1,1,H,W)))

        residual = torch.cat(residual_choose, dim=1)

        decoded_residual = self.decoder1(residual)
        decoded_residual = self.decoder2(decoded_residual)

        upsample_size = (decoded_residual.size(-1) * 2,) * 2
        decoded_residual = F.interpolate(decoded_residual, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.decoder3(decoded_residual)

        # _, _, ht, wt = inputs['img'].size()
        # logit_mask = F.interpolate(logit_mask, (ht, wt), mode='bilinear', align_corners=True)
        # pred = torch.softmax(logit_mask, dim=1)
        # pred = pred[:, 1, :, :].unsqueeze(1)
        # import pdb;pdb.set_trace()
        return logit_mask
        return {'logit': logit_mask, "anomaly_score": pred}
    

class RealNet(nn.Module):
    def __init__(self, model_backbone, model_afs, model_recon, model_rrs, data_cfg):
        super(RealNet, self).__init__()

        layers=[]
        for block in model_afs.structure:
            # import pdb;pdb.set_trace()
            layers.extend([layer['idx'] for layer in block['layers']])
        layers=list(set(layers))
        layers.sort()
        model_backbone.outlayers=layers	
        self.outlayers = layers
        
        self.net_backbone = Backbone('wide_resnet50_2', self.outlayers, model_backbone.kwargs['checkpoint_path'])
        self.net_afs = AFS(inplanes=self.net_backbone.get_outplanes(), instrides=self.net_backbone.get_outstrides(), structure=model_afs.structure, init_bsn=model_afs.init_bsn)
        self.net_recon = ReconstructionLayer(inplanes=self.net_afs.get_outplanes(), instrides=self.net_afs.get_outstrides(), **model_recon.kwargs)
        self.net_rrs = RRS(inplanes=self.net_recon.get_outplanes(), instrides=self.net_recon.get_outstrides(), **model_rrs.kwargs)

        self.frozen_layers = ['net_backbone', 'net_afs']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs, gt_image=None):
        model_input = {'img':imgs, 'gt_image':gt_image}
        
        if self.training and gt_image is not None:
            feats_dict = self.net_backbone(model_input, train=True)
            afs_out = self.net_afs(feats_dict, train=True)
        else:
            feats_dict = self.net_backbone(model_input)
            afs_out = self.net_afs(feats_dict)

        recon_out = self.net_recon(afs_out)
        logit_mask = self.net_rrs(recon_out)

        _, _, ht, wt = imgs.size()
        logit_mask = F.interpolate(logit_mask, (ht, wt), mode='bilinear', align_corners=True)
        pred = torch.softmax(logit_mask, dim=1)
        pred = pred[:, 1, :, :].unsqueeze(1)

        _feats_recon = [value for key, value in recon_out['feats_recon'].items()]
        if 'gt_block_feats' in afs_out.keys():

            _gt_block_feats = [afs_out['gt_block_feats'][key] for key in recon_out['feats_recon'].keys()]
        else:
            _gt_block_feats = _feats_recon
            if self.training:
                print('wrong gt blocks')
            else:
                print('testing')

        return logit_mask, pred, _feats_recon, _gt_block_feats

@MODEL.register_module
def realnet(pretrained=False, **kwargs):
    model = RealNet(**kwargs)
    return model