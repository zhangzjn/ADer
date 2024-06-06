import math
import torch
from torch import nn
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import numpy as np

from model import MODEL
from model import get_model
import sys
sys.path.append('./model')
# from .cflow_resnet import resnet18, resnet34, resnet50, wide_resnet50_2 

theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()

###############Utils###############
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))


def freia_flow_head(c, n_feat):
    coder = Ff.SequenceINN(n_feat)
    print('NF coder:', n_feat)
    for k in range(c.coupling_blocks):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def freia_cflow_head(c, n_feat):
    n_cond = c.condition_vec
    coder = Ff.SequenceINN(n_feat)
    print('CNF coder:', n_feat)
    for k in range(c.coupling_blocks):
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def load_decoder_arch(c, dim_in):
    if   c.dec_arch == 'freia-flow':
        decoder = freia_flow_head(c, dim_in)
    elif c.dec_arch == 'freia-cflow':
        decoder = freia_cflow_head(c, dim_in)
    else:
        raise NotImplementedError('{} is not supported NF!'.format(c.dec_arch))
    #print(decoder)
    return decoder


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def CFLOW_encoder(model_backbone, L):
    # def __init__(self, model_backbone, L, **kwargs):
        # super().__init__()
    pool_cnt = 0
    pool_dims = list()
    pool_layers = ['layer'+str(i) for i in range(L)]

    model_name =  model_backbone.name

    if 'resnet' in model_name:
        # encoder = wide_resnet50_2(pretrained=True)
        encoder = get_model(model_backbone)
        # import pdb;pdb.set_trace()

        if L >= 3:
            encoder.layer2.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in model_name:
                pool_dims.append(encoder.layer2[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer2[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.layer3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in model_name:
                pool_dims.append(encoder.layer3[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer3[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.layer4.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in model_name:
                pool_dims.append(encoder.layer4[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer4[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
    
    elif 'vit' in model_name:
        encoder = get_model(model_backbone)
        if L >= 3:
            encoder.blocks[10].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.blocks[2].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.blocks[6].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1

    elif 'efficient' in model_name:
        if 'b5' in model_name:
            encoder = get_model(model_backbone)
            blocks = [-2, -3, -5]
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(model_name))
        #
        if L >= 3:
            encoder.blocks[blocks[2]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[2]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.blocks[blocks[1]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[1]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.blocks[blocks[0]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[0]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1

    elif 'mobile' in model_name: # TODO, To check
        if 'mobilenet_v3_small' in model_name:
            encoder = get_model(model_backbone).features
            blocks = [-2, -5, -10]
        elif 'mobilenet_v3_large' in model_name: # TODO, To check
            encoder = get_model(model_backbone).features
            blocks = [-2, -5, -11]
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(model_name))
        #
        if L >= 3:
            encoder[blocks[2]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[2]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder[blocks[1]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[1]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder[blocks[0]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[0]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
    else:
        raise NotImplementedError('{} is not supported architecture!'.format(model_name))
    #
    return encoder, pool_layers, pool_dims


class CFLOW(nn.Module):
    def __init__(self, model_backbone, L, N): # model_backbone = NameSpace()
        super(CFLOW, self).__init__()

        self.model_backbone = model_backbone
        self.P = model_backbone.condition_vec
        self.N = N
        
        self.encoder, self.pool_layers, pool_dims = CFLOW_encoder(model_backbone=model_backbone, L=L)

        decoders = [load_decoder_arch(model_backbone, pool_dim) for pool_dim in pool_dims]
        # self.decoders = [decoder.to(model_backbone.device) for decoder in decoders] # .to(model_backbone.device)

        self.decoders = nn.ModuleList([])
        for decoder in decoders:
            self.decoders.append(decoder)

        self.frozen_layers = ['encoder']
        
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
    

    def forward(self, images):
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            _ = self.encoder(images)
        # train decoder
        e_list = list()
        c_list = list()
        # for l, layer in enumerate(self.pool_layers):
            
            # for f in range(FIB):
                # pass
                # sum_loss += t2np(loss.sum())
                # loss_count += len(loss)

    def Decoder_forward(self, dec_idx, dec_layer):
        if 'vit' in self.model_backbone.name:
            e = activation[dec_layer].transpose(1, 2)[...,1:]
            e_hw = int(np.sqrt(e.size(2)))
            e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
        else:
            e = activation[dec_layer].detach()  # BxCxHxW
        #
        B, C, H, W = e.size()
        S = H*W
        E = B*S    
        #
        p = positionalencoding2d(self.P, H, W).to(self.model_backbone.device).unsqueeze(0).repeat(B, 1, 1, 1)
        c_r = p.reshape(B, self.P, S).transpose(1, 2).reshape(E, self.P)  # BHWxP
        e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
        perm = torch.randperm(E).to(self.model_backbone.device)  # BHW
        # decoder = self.decoders[dec_idx]
        
        FIB = E//self.N + int(E%self.N > 0)  # number of fiber batches
        return FIB, c_r, e_r, dec_idx, perm, E, C ,H ,W


    def FIB_forward(self, f, FIB, c_r, e_r, dec_idx, N, E, C, dec_arch, perm=None):
        if perm is not None:
            idx = torch.arange(f*N, (f+1)*N)
        
        if f < (FIB-1) and perm is None:
            idx = torch.arange(f*N, (f+1)*N)
        else:
            idx = torch.arange(f*N, E)
        
        if perm is not None:
            c_p = c_r[perm[idx]]  # NxP
            e_p = e_r[perm[idx]]  # NxC
        else:
            c_p = c_r[idx]  # NxP
            e_p = e_r[idx]  # NxC 

        if 'cflow' in dec_arch:
            z, log_jac_det = self.decoders[dec_idx](e_p, [c_p,])
        else:
            z, log_jac_det = self.decoders[dec_idx](e_p)

        decoder_log_prob = get_logp(C, z, log_jac_det)
        log_prob = decoder_log_prob / C  # likelihood per dim
        loss = -log_theta(log_prob)
        
        return log_prob, loss
    

@MODEL.register_module
def cflow(pretrained=False, **kwargs):
	model = CFLOW(**kwargs)
	return model


# if __name__ == '__main__':
# 	from argparse import Namespace as _Namespace

# 	bs = 2
# 	reso = 224
# 	x = torch.randn(bs, 3, reso, reso).cuda()
# 	model_backbone = _Namespace()
