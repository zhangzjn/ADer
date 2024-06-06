import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
from einops import rearrange
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch.nn.functional as F
from model import MODEL
from model import get_model
from data import get_loader

class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                            kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out
    
class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        batch_size_shape, _, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        if torch.cuda.is_available and self.use_cuda:
            input_tensor = input_tensor.cuda()
            xx_channel = xx_channel.cuda()
            yy_channel = yy_channel.cuda()
        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)

        return out
    
class Descriptor(nn.Module):
    def __init__(self, gamma_d, cnn):
        super(Descriptor, self).__init__()
        self.cnn = cnn
        if cnn == 'wrn50_2':
            dim = 1792 
            self.layer = CoordConv2d(dim, dim//gamma_d, 1)
        elif cnn == 'res18':
            dim = 448
            self.layer = CoordConv2d(dim, dim//gamma_d, 1)
        elif cnn == 'effnet-b5':
            dim = 568
            self.layer = CoordConv2d(dim, 2*dim//gamma_d, 1)
        elif cnn == 'vgg19':
            dim = 1280 
            self.layer = CoordConv2d(dim, dim//gamma_d, 1)
        

    def forward(self, p):
        sample = None
        for o in p:
            o = F.avg_pool2d(o, 3, 1, 1) / o.size(1) if self.cnn == 'effnet-b5' else F.avg_pool2d(o, 3, 1, 1)
            sample = o if sample is None else torch.cat((sample, F.interpolate(o, sample.size(2), mode='bilinear')), dim=1)
        
        phi_p = self.layer(sample)
        return phi_p
    

class DSVDD(nn.Module):
    def __init__(self, model, data_loader, cnn, gamma_c, gamma_d, device):
        super(DSVDD, self).__init__()
        self.device = device
        
        self.C   = 0
        self.nu = 1e-3
        self.scale = None

        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.K = 3
        self.J = 3

        self.r   = nn.Parameter(1e-5*torch.ones(1), requires_grad=True)
        self.Descriptor = Descriptor(self.gamma_d, cnn).to(device)
        self._init_centroid(model, data_loader)
        self.C = rearrange(self.C, 'b c h w -> (b h w) c').detach()
        
        if self.gamma_c > 1:
            self.C = self.C.cpu().detach().numpy()
            self.C = KMeans(n_clusters=(self.scale**2)//self.gamma_c, max_iter=3000).fit(self.C).cluster_centers_
            self.C = torch.Tensor(self.C).to(device)

        self.C = self.C.transpose(-1, -2).detach()
        self.C = nn.Parameter(self.C, requires_grad=False)
        # import pdb;pdb.set_trace()

    def forward(self, p):
        phi_p = self.Descriptor(p)       
        phi_p = rearrange(phi_p, 'b c h w -> b (h w) c')
        
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)    
        centers  = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C))
        dist     = features + centers - f_c
        dist     = torch.sqrt(dist)

        n_neighbors = self.K
        dist     = dist.topk(n_neighbors, largest=False).values

        dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dist = dist.unsqueeze(-1)

        score = rearrange(dist, 'b (h w) c -> b c h w', h=self.scale)
        
        loss = 0
        if self.training:
            loss = self._soft_boundary(phi_p)
            return loss
        else:
            return score

    def _soft_boundary(self, phi_p):
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C))
        dist     = features + centers - f_c
        n_neighbors = self.K + self.J
        dist     = dist.topk(n_neighbors, largest=False).values

        score = (dist[:, : , :self.K] - self.r**2) 
        L_att = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))
        
        score = (self.r**2 - dist[:, : , self.J:]) 
        L_rep  = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))
        
        loss = L_att + L_rep

        return loss 

    def _init_centroid(self, model, data_loader):
        model = model.to(self.device)
        model.eval()
        for i, inputs in enumerate(tqdm(data_loader)):
            x = inputs['img'].to(self.device)
            p = model(x)
            self.scale = p[0].size(2)
            phi_p = self.Descriptor(p)
            self.C = ((self.C * i) + torch.mean(phi_p, dim=0, keepdim=True).detach()) / (i+1)


class CFA(nn.Module):
    def __init__(self, model_backbone, model_dsvdd, data_cfg) -> None:
        super().__init__()

        # import pdb;pdb.set_trace()
        # self.model_backbone = model_backbone
        self.net_backbone = get_model(model_backbone)

        data_loader, _ = get_loader(data_cfg)
        self.net_dsvdd = DSVDD(self.net_backbone, data_loader=data_loader, **model_dsvdd.kwargs)

        self.frozen_layers = ['net_backbone']

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


    def forward(self, imgs):
        # self.net_backbone.eval()
        feature = self.net_backbone(imgs)
        for i in range(len(feature)):
            feature[i] = F.leaky_relu(feature[i])  

        out = self.net_dsvdd(feature)

        return out

@MODEL.register_module
def cfa(pretrained, **kwargs):
    model = CFA(**kwargs)
    return model
