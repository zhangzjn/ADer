import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
from scipy import linalg as la

from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Tuple
from abc import abstractmethod
from model import MODEL
from model import get_model
from einops import rearrange

def kornia_filter2d(input, kernel):
    """
        conv2d function from kornia.
    """
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    return output

class InvertibleModule(nn.Module):
    """
        Base class for constructing normalizing flow.
        You should be implemente `forward` and `inverse` function mannuly, which define a basic invertible module.
        Each function needs to implement the corresponding output tensor (the value returned by the function) for a given input tensor,
            and also needs to implement the Jacobian determinant of the output tensor relative to the input tensor.

        Note: function `_forward` and `_inverse` is hidden function for user, should not be modified or called.
    """

    def __init__(self):
        super(InvertibleModule, self).__init__()

    """ Abstract functions (`forward` and `inverse`) need to be implemented explicitly. """

    @abstractmethod
    def forward(self, inputs: Tuple[torch.Tensor], logdets: Tuple[torch.Tensor]):
        raise NotImplementedError

    @abstractmethod
    def inverse(self, outputs: Tuple[torch.Tensor], logdets: Tuple[torch.Tensor]):
        raise NotImplementedError

    """ Hidden functions (`_forward` and `_inverse`) for implementing SequentialNF. """

    def _forward(self, *inputs_logdets):
        assert len(inputs_logdets) % 2 == 0
        inputs = inputs_logdets[:len(inputs_logdets) // 2]
        logdets = inputs_logdets[len(inputs_logdets) // 2:]
        outputs, logdets = self.forward(inputs, logdets)
        return outputs + logdets  # Kept in a repeatable form.

    def _inverse(self, *outputs_logdets):
        assert len(outputs_logdets) % 2 == 0
        outputs = outputs_logdets[:len(outputs_logdets) // 2]
        logdets = outputs_logdets[len(outputs_logdets) // 2:]
        inputs, logdets = self.inverse(outputs, logdets)
        return inputs + logdets


class AutoNFSequential(torch.autograd.Function):
    """
        Automatic implementation for sequential normalizing flows.
        This class is hidden class for user, should not be modified or called.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, _forward_lst, _inverse_lst, inplogsRange, paramsRanges,
                *inplogs_and_params):  # parameter passing only by *inplogs_and_params
        assert inplogsRange[1] % 2 == 0
        inplogs = inplogs_and_params[inplogsRange[0]: inplogsRange[1]]  # Save for the gradient later

        with torch.no_grad():
            outlogs = tuple([inplog.detach() for inplog in inplogs])
            for _forward in _forward_lst:
                outlogs = _forward(*outlogs)  # Not save forward tensor
                for outlog in outlogs:
                    assert not outlog.isnan().any()

        ctx._forward_lst = _forward_lst
        ctx._inverse_lst = _inverse_lst
        ctx.outlogsRange = inplogsRange
        ctx.paramsRanges = paramsRanges

        outlogs = tuple([outlog.detach() for outlog in outlogs])
        params = inplogs_and_params[inplogsRange[1]:]
        ctx.save_for_backward(*outlogs, *params)  # only the last output

        return outlogs

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_outlogs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("This function is not compatible with .grad(), please use .backward() if possible")
        outlogs_params = ctx.saved_tensors
        outlogs = outlogs_params[ctx.outlogsRange[0]: ctx.outlogsRange[1]]
        params = [outlogs_params[range[0]: range[1]] for range in ctx.paramsRanges]
        _inverse_lst = ctx._inverse_lst
        _forward_lst = ctx._forward_lst
        grad_outlogs_loop = grad_outlogs

        # While reverse calculation, calculate gradient.There is always only one hidden tensor.
        grad_params = tuple()  # Saved parameter gradients.
        detached_outlogs_loop = tuple([outlog.detach() for outlog in outlogs])
        for _forward, _inverse, param in zip(reversed(_forward_lst), reversed(_inverse_lst), reversed(params)):
            with torch.no_grad():
                inplogs = _inverse(*detached_outlogs_loop)
            with torch.set_grad_enabled(True):
                inplogs_loop = tuple([inplog.detach().requires_grad_() for inplog in inplogs])
                outlogs_loop = _forward(*inplogs_loop)
            grad_inplogs_params = torch.autograd.grad(outputs=outlogs_loop, grad_outputs=grad_outlogs_loop,
                                                      inputs=inplogs_loop + param)

            detached_outlogs_loop = tuple([inplog.detach() for inplog in inplogs])
            grad_outlogs_loop = grad_inplogs_params[:len(inplogs_loop)]  # The gradient of the input
            grad_params = grad_inplogs_params[len(inplogs_loop):] + grad_params  # The gradient of the parameter

        grad_inplogs = grad_outlogs_loop
        return (None, None, None, None,) + grad_inplogs + grad_params


class SequentialNF(InvertibleModule):
    """ A constructor class to build memory saving normalizing flows by a tuple of `InvertibleModule` """

    def __init__(self, modules: Tuple[InvertibleModule]):
        super(SequentialNF, self).__init__()
        self.moduleslst = nn.ModuleList(modules)
        self._forward_lst = tuple([module._forward for module in modules])
        self._inverse_lst = tuple([module._inverse for module in modules])
        self.params = [[p for p in module.parameters() if p.requires_grad] for module in self.moduleslst]  # to fix

    def forward(self, inputs,
                logdets):  # Calculate the Jacobian determinant of the output return value relative to the input value,
        #   which is partial{output}/partial{input}
        assert len(inputs) == len(logdets)
        inplogsRange = [0, len(inputs) + len(logdets)]
        paramsRange, lastIdx = [], inplogsRange[-1]
        for param in self.params:
            paramsRange.append([lastIdx, lastIdx + len(param)])
            lastIdx += len(param)

        outlogs = AutoNFSequential.apply(
            self._forward_lst, self._inverse_lst,
            inplogsRange, paramsRange,
            *inputs, *logdets,
            *[p for param in self.params for p in param]
        )
        mid = len(outlogs) // 2
        return outlogs[:mid], outlogs[mid:]

    def inverse(self, outputs, logdets):  # which is partial{input}/partial{output}
        assert len(outputs) == len(logdets)
        outlogsRange = [0, len(outputs) + len(logdets)]
        paramsRange, lastIdx = [], outlogsRange[-1]
        for param in self.params:
            paramsRange.append([lastIdx, lastIdx + len(param)])
            lastIdx += len(param)

        inplogs = AutoNFSequential.apply(
            list(reversed(self._inverse_lst)), list(reversed(self._forward_lst)),
            outlogsRange, paramsRange,
            *outputs, *logdets,
            *[p for param in self.params for p in param]
        )
        mid = len(inplogs) // 2
        return inplogs[:mid], inplogs[mid:]


class SequentialNet(nn.Module):
    """ A constructor class to build pytorch-based normalizing flows by a tuple of `nn.Module` """

    def __init__(self, modules: Tuple[nn.Module]):
        super(SequentialNet, self).__init__()
        self.moduleslst = nn.ModuleList(modules)

    def forward(self, inputs, logdets):
        outputs = tuple(inputs)
        for m in self.moduleslst:
            outputs, logdets = m(outputs, logdets)
        return outputs, logdets


class SemiInvertible_1x1Conv(nn.Conv2d):
    """
        Semi-invertible 1x1Conv is used at the first stage of NF.
    """
    def __init__(self, in_channels, out_channels ) -> None:
        assert out_channels >= in_channels
        super().__init__(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.weight.data) # orth initialization
    def inverse(self, output):
        b, c, h, w = output.shape
        A = self.weight[..., 0,0] # outch, inch
        B = output.permute([1,0,2,3]).reshape(c, -1) # outch, bhw
        X = torch.linalg.lstsq(A, B)  # AX=B
        return X.solution.reshape(-1, b, h, w).permute([1, 0, 2, 3])
    @property
    def logdet(self):
        w = self.weight.squeeze() # out,in
        return 0.5*torch.logdet(w.T@w)


class LaplacianMaxPyramid(nn.Module):
    def __init__(self, num_levels) -> None:
        super().__init__()
        self.kernel = torch.tensor(
            [
                [
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [6.0, 24.0, 36.0, 24.0, 6.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                ]
            ]
        )/ 256.0
        self.num_levels = num_levels-1 # 总共有num_levels层，

    def _pyramid_down(self, input, pad_mode='constant'):
        if not len(input.shape) == 4:
            raise ValueError(f'Invalid img shape, we expect BCHW, got: {input.shape}')
        # blur 
        img_pad = F.pad(input, (2,2,2,2), mode=pad_mode)
        img_blur = kornia_filter2d(img_pad, kernel=self.kernel)
        # downsample
        out = F.max_pool2d(img_blur, kernel_size=2, stride=2)# 使用max pooling取代下采样
        return out

    def _pyramid_up(self, input, size, pad_mode='constant'):
        if not len(input.shape) == 4:
            raise ValueError(f'Invalid img shape, we expect BCHW, got: {input.shape}')
        # upsample
        img_up = F.interpolate(input, size=size, mode='nearest', )
        # blur
        img_pad = F.pad(img_up, (2,2,2,2), mode=pad_mode)
        img_blur = kornia_filter2d(img_pad, kernel=self.kernel)
        return img_blur
        
    def build_pyramid(self, input):
        gp, lp = [input], []
        for _ in range(self.num_levels):
            gp.append(self._pyramid_down(gp[-1]))
        for layer in range(self.num_levels):
            curr_gp = gp[layer]
            next_gp = self._pyramid_up(gp[layer+1], size=curr_gp.shape[2:])
            lp.append(curr_gp - next_gp)
        lp.append(gp[self.num_levels]) # 最后一层不是gp
        return lp

    def compose_pyramid(self, lp):
        rs = lp[-1]
        for i in range(len(lp)-2, -1, -1):
            rs = self._pyramid_up(rs, size=lp[i].shape[2:])
            rs = torch.add(rs, lp[i])
        return rs


class VolumeNorm(nn.Module):
    """
        Volume Normalization.
        CVN dims = (0,1);  SVN dims = (0,2,3)
    """
    def __init__(self, dims=(0,1) ):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(1,1,1,1))
        self.momentum = 0.1
        self.dims = dims
    def forward(self, x):
        if self.training:
            sample_mean = torch.mean(x, dim=self.dims, keepdim=True) 
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * sample_mean
            out = x - sample_mean
        else:
            out = x - self.running_mean
        return out

class AffineParamBlock(nn.Module):
    """
        Estimate `slog` and `t`.
    """
    def __init__(self, in_ch, out_ch=None, hidden_ch=None, ksize=7, clamp=2, vn_dims=(0,1)):
        super().__init__()
        if out_ch is None: 
            out_ch = 2*in_ch 
        if hidden_ch is None:
            hidden_ch = out_ch
        self.clamp = clamp
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=ksize, padding=ksize//2, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=ksize, padding=ksize//2, bias=False),
        )
        nn.init.zeros_(self.conv[-1].weight.data)
        self.norm = VolumeNorm(vn_dims)
    def forward(self, input, forward_mode:bool):
        output = self.conv(input)
        _dlogdet, bias = output.chunk(2, 1)
        dlogdet = self.clamp * 0.636 * torch.atan(_dlogdet / self.clamp)  # soft clip
        dlogdet = self.norm(dlogdet)
        scale = torch.exp(dlogdet)
        return (scale, bias), dlogdet # scale * x + bias

class InvConv2dLU(nn.Module):
    """
        Invertible 1x1Conv with volume normalization.
    """
    def __init__(self, in_channel, volumeNorm=True):
        super().__init__()
        self.volumeNorm = volumeNorm
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p.copy())
        w_l = torch.from_numpy(w_l.copy())
        w_s = torch.from_numpy(w_s.copy())
        w_u = torch.from_numpy(w_u.copy())

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(w_s.abs().log())
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        return out

    def inverse(self, output):
        _, _, height, width = output.shape
        weight = self.calc_weight()
        inv_weight = torch.inverse(weight.squeeze().double()).float()
        input = F.conv2d(output, inv_weight.unsqueeze(2).unsqueeze(3))
        return input

    def calc_weight(self):
        if self.volumeNorm:
            w_s = self.w_s - self.w_s.mean() # volume normalization
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)


class FlowBlock(InvertibleModule):
    """
        @Paper Figure3(c) The proposed scale-wise pyramid coupling block.
    """
    def __init__(self, channel, direct, start_level, ksize, vn_dims):
        super().__init__()
        assert direct in ['up', 'down']
        self.direct = direct
        self.start_idx = start_level
        self.affineParams = AffineParamBlock(channel, ksize=ksize, vn_dims=vn_dims)
        self.conv1x1 = InvConv2dLU(channel)

    def forward(self, inputs, logdets):
        assert self.start_idx+1 < len(inputs)
        x0, x1 = inputs[self.start_idx: self.start_idx+2]
        logdet0, logdet1 = logdets[self.start_idx: self.start_idx+2]
        if self.direct == 'up':
            y10 = F.interpolate(x1, size=x0.shape[2:], mode='nearest') # interp first
            (scale0, bias0), dlogdet0 = self.affineParams(y10, forward_mode=True)
            z0, z1 = scale0*x0+bias0, x1
            z0 = self.conv1x1(z0) 
            dlogdet1 = 0
        else:
            (scale10, bias10), dlogdet10 = self.affineParams(x0, forward_mode=True)
            scale1, bias1, dlogdet1 = F.interpolate(scale10, size=x1.shape[2:], mode='nearest'),\
                                         F.interpolate(bias10, size=x1.shape[2:], mode='nearest'),\
                                             F.interpolate(dlogdet10, size=x1.shape[2:], mode='nearest') # interp after
            z0, z1 = x0, scale1*x1+bias1
            z1 = self.conv1x1(z1) 
            dlogdet0 = 0
        outputs = inputs[:self.start_idx]+(z0, z1)+inputs[self.start_idx+2:]
        out_logdets = logdets[:self.start_idx]+(logdet0+dlogdet0, logdet1+dlogdet1)+logdets[self.start_idx+2:]
        return outputs, out_logdets

    def inverse(self, outputs, logdets):
        assert self.start_idx+1 < len(outputs)
        z0, z1 = outputs[self.start_idx: self.start_idx+2]
        logdet0, logdet1 = logdets[self.start_idx: self.start_idx+2]
        if self.direct == 'up':
            z0 = self.conv1x1.inverse(z0)
            z10 = F.interpolate(z1, size=z0.shape[2:], mode='nearest') # interp first
            (scale0, bias0), dlogdet0 = self.affineParams(z10, forward_mode=False)
            x0, x1 = (z0-bias0)/scale0, z1
            dlogdet1 = 0
        else:
            z1 = self.conv1x1.inverse(z1)
            (scale01, bias01), dlogdet01 = self.affineParams(z0, forward_mode=False)
            scale1, bias1, dlogdet1 = F.interpolate(scale01, size=z1.shape[2:], mode='nearest'),\
                                         F.interpolate(bias01, size=z1.shape[2:], mode='nearest'),\
                                             F.interpolate(dlogdet01, size=z1.shape[2:], mode='nearest') # interp after
            x0, x1 = z0, (z1-bias1)/scale1
            dlogdet0 = 0
        inputs = outputs[:self.start_idx]+(x0, x1)+outputs[self.start_idx+2:]
        in_logdets = logdets[:self.start_idx]+(logdet0-dlogdet0, logdet1-dlogdet1)+logdets[self.start_idx+2:]
        return inputs, in_logdets


class FlowBlock2(InvertibleModule):
    """
        @Paper Figure3(d) The reverse parallel and reparameterized of (c)-architecture.
    """
    def __init__(self, channel, start_level, ksize, vn_dims):
        super().__init__()
        self.start_idx = start_level
        self.affineParams = AffineParamBlock(in_ch=2*channel, out_ch=2*channel, ksize=ksize, vn_dims=vn_dims)
        self.conv1x1 = InvConv2dLU(channel)

    def forward(self, inputs, logdets):
        x0, x1, x2 = inputs[self.start_idx: self.start_idx+3]
        logdet0, logdet1, logdet2 = logdets[self.start_idx: self.start_idx+3]
        y01 = F.interpolate(x0, size=x1.shape[2:], mode='nearest')
        y21 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')
        affine_input = torch.concat([y01, y21], dim=1) # b, 2*ch, h, w
        (scale1, bias1), dlogdet1 = self.affineParams(affine_input, forward_mode=True)
        z0, z1, z2 = x0, scale1*x1+bias1, x2
        z1 = self.conv1x1(z1)
        outputs = inputs[:self.start_idx]+(z0, z1, z2)+inputs[self.start_idx+3:]
        out_logdets = logdets[:self.start_idx]+(logdet0, logdet1+dlogdet1, logdet2)+logdets[self.start_idx+3:]
        return outputs, out_logdets

    def inverse(self, outputs, logdets):
        z0, z1, z2 = outputs[self.start_idx: self.start_idx+3]
        logdet0, logdet1, logdet2 = logdets[self.start_idx: self.start_idx+3]
        z1 = self.conv1x1.inverse(z1)
        z01 = F.interpolate(z0, size=z1.shape[2:], mode='nearest')
        z21 = F.interpolate(z2, size=z1.shape[2:], mode='nearest')
        affine_input = torch.concat([z01, z21], dim=1) # b, 2*ch, h, w
        (scale1, bias1), dlogdet1 = self.affineParams(affine_input, forward_mode=False)
        x0, x1, x2 = z0, (z1-bias1)/scale1, z2
        inputs = outputs[:self.start_idx]+(x0, x1, x2)+outputs[self.start_idx+3:]
        in_logdets = logdets[:self.start_idx]+(logdet0, logdet1-dlogdet1, logdet2)+logdets[self.start_idx+3:]
        return inputs, in_logdets

class BatchDiffLoss(nn.Module):
    """
        Difference Loss within a batch.
    """
    def __init__(self, batchsize=2, p=2) -> None:
        super().__init__()
        self.idx0, self.idx1 = np.triu_indices(n=batchsize, k=1)
        self.p = p
    def forward(self, pyramid):
        diffes = []
        for input in pyramid:
            diff = (input[self.idx0] - input[self.idx1]).abs()**self.p
            diffes.append(diff)
        return diffes

class PyramidFlow(nn.Module):
    """
        PyramidFlow
        NOTE: resnetX=0 use 1x1 conv with #channel channel.
    """
    def __init__(self, resnet, channel, num_level, num_stack, ksize, vn_dims, batchsize=2, savemem=False):
        super().__init__()
        assert num_level >= 2
        self.channel = channel
        self.num_level = num_level
        
        modules = []
        for _ in range(num_stack):
            for range_start in [0, 1]:
                if range_start == 1:
                    modules.append(FlowBlock(self.channel, direct='up', start_level=0, ksize=ksize, vn_dims=vn_dims))
                for start_idx in range(range_start, num_level, 2):
                    if start_idx+2 < num_level:
                        modules.append(FlowBlock2(self.channel, start_level=start_idx, ksize=ksize, vn_dims=vn_dims))
                    elif start_idx+1 < num_level:
                        modules.append(FlowBlock(self.channel, direct='down', start_level=start_idx, ksize=ksize, vn_dims=vn_dims))
        self.nf = SequentialNF(modules) if savemem else SequentialNet(modules)

        if resnet is not None:
            self.inconv = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1
            )# 1024->256
        else:
            self.inconv = SemiInvertible_1x1Conv(3, self.channel)

        self.pyramid = LaplacianMaxPyramid(num_level)
        self.loss = BatchDiffLoss(batchsize=batchsize,p=2)

    def forward(self, imgs):
        b, c, h, w = imgs.shape
        assert h%(2**(self.num_level-1))==0 and w%(2**(self.num_level-1))==0
        with torch.no_grad():
            feat1 = self.inconv(imgs) # fix inconv/encoder
        pyramid = self.pyramid.build_pyramid(feat1)
        logdets = tuple(torch.zeros_like(pyramid_j) for pyramid_j in pyramid)
        pyramid_out, logdets_out = self.nf.forward(pyramid, logdets)
        diffes = self.loss(pyramid_out)
        diff_pixel = self.pyramid.compose_pyramid(diffes).mean(1)
        return diff_pixel

    def pred_tempelate(self, imgs):
        b, c, h, w = imgs.shape
        assert h%(2**(self.num_level-1))==0 and w%(2**(self.num_level-1))==0
        with torch.no_grad():
            feat1 = self.inconv(imgs) # fix inconv/encoder
        pyramid = self.pyramid.build_pyramid(feat1)
        logdets = tuple(torch.zeros_like(pyramid_j) for pyramid_j in pyramid)
        pyramid_out, logdets_out = self.nf.forward(pyramid, logdets)
        return pyramid_out

    def predict(self, imgs, template):
        # imgs = imgs['img'].cuda()
        b, c, h, w = imgs.shape
        assert h%(2**(self.num_level-1))==0 and w%(2**(self.num_level-1))==0
        with torch.no_grad():
            feat1 = self.inconv(imgs) # fix inconv/encoder
        pyramid = self.pyramid.build_pyramid(feat1)
        logdets = tuple(torch.zeros_like(pyramid_j) for pyramid_j in pyramid)
        pyramid_out, logdets_out = self.nf.forward(pyramid, logdets)
        if template[0].size(0) == 1:
            pyramid_diff = [(feat2 - template).abs() for feat2, template in zip(pyramid_out, template)]
        else:
            pyramid_diff = []
            for feat2, template in zip(pyramid_out, template):
                feat = rearrange(feat2,'(b v) c h w -> b v c h w', v=5)
                pyramid_diffs = []
                for i in range(feat.size(0)):
                    pyramid_diff_s = (feat[i] - template).abs()
                    pyramid_diffs.append(pyramid_diff_s.unsqueeze(dim=0))
                pyramid_diffs = torch.cat(pyramid_diffs, dim=0)
                pyramid_diffs = rearrange(pyramid_diffs, 'b v c h w -> (b v) c h w')
                pyramid_diff.append(pyramid_diffs)
        diff = self.pyramid.compose_pyramid(pyramid_diff).mean(1, keepdim=True)
        return diff

    def inverse(self, pyramid_out):
        logdets_out = tuple(torch.zeros_like(pyramid_j) for pyramid_j in pyramid_out)
        pyramid_in, logdets_in = self.nf.inverse(pyramid_out, logdets_out)
        feat1 = self.pyramid.compose_pyramid(pyramid_in)
        if self.channel != 64:
            input = self.inconv.inverse(feat1)
            return input
        return feat1


class PYRAMIDFLOW(torch.nn.Module):
    def __init__(self, model_backbone, batchsize):
        super(PYRAMIDFLOW, self).__init__()
        # self.net_backbone = NetworkFeatureAggregator(model_backbone,layers_to_extract_from=('layer2','layer3'))
        # self.prep = Preprocessing(input_dims=[512,1024],output_dim=1536)
        # self.net_proj = Projection(1536)
        # self.net_disc = Discriminator(in_planes=1536)
        self.model_backbone = get_model(model_backbone)
        self.net_pyramidflow = PyramidFlow(resnet=self.model_backbone, channel=64, num_level=4, num_stack=4, ksize=7, vn_dims=(0,1), batchsize=batchsize,savemem=False)
        # self.frozen_layers = ['net_backbone']
        self.frozen_layers = ['inconv']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.net_pyramidflow.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):
        diff_pixels = self.net_pyramidflow(imgs)
        return diff_pixels



@MODEL.register_module
def pyramidflow(pretrained=False, **kwargs):
    model = PYRAMIDFLOW(**kwargs)
    return model