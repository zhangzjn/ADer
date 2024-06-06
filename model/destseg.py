import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, Optional
from model import MODEL
from model import get_model



def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def make_layer(block, inplanes, planes, blocks, stride=1, norm_layer=None):
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, norm_layer=norm_layer))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, norm_layer=norm_layer))

    return nn.Sequential(*layers)


def l2_normalize(input, dim=1, eps=1e-12):
    denom = torch.sqrt(torch.sum(input**2, dim=dim, keepdim=True))
    return input / (denom + eps)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def get_norm_layer(norm: str):
    norm = {
        "BN": nn.BatchNorm2d,
        "LN": nn.LayerNorm,
    }[norm.upper()]
    return norm


def get_act_layer(act: str):
    act = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
    }[act.lower()]
    return act

class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        conv_kwargs=None,
        norm_layer=None,
        norm_kwargs=None,
        act_layer=None,
        act_kwargs=None,
    ):
        super(ConvNormAct2d, self).__init__()

        conv_kwargs = {}
        if norm_layer:
            conv_kwargs["bias"] = False
        if padding == "same" and stride > 1:
            # if kernel_size is even, -1 is must
            padding = (kernel_size - 1) // 2

        self.conv = self._build_conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            conv_kwargs,
        )
        self.norm = None
        if norm_layer:
            norm_kwargs = {}
            self.norm = get_norm_layer(norm_layer)(
                num_features=out_channels, **norm_kwargs
            )
        self.act = None
        if act_layer:
            act_kwargs = {}
            self.act = get_act_layer(act_layer)(**act_kwargs)

    def _build_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        conv_kwargs,
    ):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            **conv_kwargs,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class ASPP(nn.Module):
    def __init__(self, input_channels, output_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvNormAct2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    norm_layer="BN",
                    act_layer="RELU",
                ),
            )
        )
        for atrous_rate in atrous_rates:
            conv_norm_act = ConvNormAct2d
            modules.append(
                conv_norm_act(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=1 if atrous_rate == 1 else 3,
                    padding=0 if atrous_rate == 1 else atrous_rate,
                    dilation=atrous_rate,
                    norm_layer="BN",
                    act_layer="RELU",
                )
            )

        self.aspp_feature_extractors = nn.ModuleList(modules)
        self.aspp_fusion_layer = ConvNormAct2d(
            (1 + len(atrous_rates)) * output_channels,
            output_channels,
            kernel_size=3,
            norm_layer="BN",
            act_layer="RELU",
        )

    def forward(self, x):
        res = []
        for aspp_feature_extractor in self.aspp_feature_extractors:
            res.append(aspp_feature_extractor(x))
        res[0] = F.interpolate(
            input=res[0], size=x.shape[2:], mode="bilinear", align_corners=False
        )  # resize back after global-avg-pooling layer
        res = torch.cat(res, dim=1)
        res = self.aspp_fusion_layer(res)
        return res


class TeacherNet(nn.Module):
    def __init__(self,teacher=None):
        super().__init__()
        self.encoder = teacher

    def forward(self, x):
        self.eval()
        x1, x2, x3 = self.encoder(x)
        return (x1, x2, x3)


class StudentNet(nn.Module):
    def __init__(self, student=None, ed=True):
        super().__init__()
        self.ed = ed
        if self.ed:
            self.decoder_layer4 = make_layer(BasicBlock, 512, 512, 2)
            self.decoder_layer3 = make_layer(BasicBlock, 512, 256, 2)
            self.decoder_layer2 = make_layer(BasicBlock, 256, 128, 2)
            self.decoder_layer1 = make_layer(BasicBlock, 128, 64, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.encoder = student

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        if not self.ed:
            return (x1, x2, x3)
        x = x4
        b4 = self.decoder_layer4(x)
        b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
        b3 = self.decoder_layer3(b3)
        b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
        b2 = self.decoder_layer2(b2)
        b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
        b1 = self.decoder_layer1(b1)
        return (b1, b2, b3)


class SegmentationNet(nn.Module):
    def __init__(self, inplanes=448):
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x


class DeSTSeg(nn.Module):
    def __init__(self, teacher=None, student=None, dest=True, ed=True):
        super().__init__()
        self.teacher_net = TeacherNet(teacher)
        self.student_net = StudentNet(student, ed)
        self.dest = dest
        self.segmentation_net = SegmentationNet(inplanes=448)

    def forward(self, img_aug, img_origin=None, mask=None):
        self.teacher_net.eval()

        if img_origin is None:  # for inference
            img_origin = img_aug.clone()

        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_aug)
        ]
        outputs_student_aug = [
            l2_normalize(output_s) for output_s in self.student_net(img_aug)
        ]
        output = torch.cat(
            [
                F.interpolate(
                    -output_t * output_s,
                    size=outputs_student_aug[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
            ],
            dim=1,
        )

        output_segmentation = self.segmentation_net(output)

        if self.dest:
            outputs_student = outputs_student_aug
        else:
            outputs_student = [
                l2_normalize(output_s) for output_s in self.student_net(img_origin)
            ]
        outputs_teacher = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_origin)
        ]

        output_de_st_list = []
        for output_t, output_s in zip(outputs_teacher, outputs_student):
            a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
            output_de_st_list.append(a_map)
        output_de_st = torch.cat(
            [
                F.interpolate(
                    output_de_st_instance,
                    size=outputs_student[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_de_st_instance in output_de_st_list
            ],
            dim=1,
        )  # [N, 3, H, W]
        output_de_st = torch.prod(output_de_st, dim=1, keepdim=True)

        if mask is not None:
            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )
        return output_segmentation, output_de_st, output_de_st_list, mask

    # def evaluate(self, img_aug, img_origin=None, mask=None):
    #     self.teacher_net.eval()
    #
    #     if img_origin is None:  # for inference
    #         img_origin = img_aug.clone()
    #
    #     outputs_teacher_aug = [
    #         l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_aug)
    #     ]
    #     outputs_student_aug = [
    #         l2_normalize(output_s) for output_s in self.student_net(img_aug)
    #     ]
    #     output = torch.cat(
    #         [
    #             F.interpolate(
    #                 -output_t * output_s,
    #                 size=outputs_student_aug[0].size()[2:],
    #                 mode="bilinear",
    #                 align_corners=False,
    #             )
    #             for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
    #         ],
    #         dim=1,
    #     )
    #
    #     output_segmentation = self.segmentation_net(output)
    #
    #     if self.dest:
    #         outputs_student = outputs_student_aug
    #     else:
    #         outputs_student = [
    #             l2_normalize(output_s) for output_s in self.student_net(img_origin)
    #         ]
    #     outputs_teacher = [
    #         l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_origin)
    #     ]
    #
    #     output_de_st_list = []
    #     for output_t, output_s in zip(outputs_teacher, outputs_student):
    #         a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
    #         output_de_st_list.append(a_map)
    #     output_de_st = torch.cat(
    #         [
    #             F.interpolate(
    #                 output_de_st_instance,
    #                 size=outputs_student[0].size()[2:],
    #                 mode="bilinear",
    #                 align_corners=False,
    #             )
    #             for output_de_st_instance in output_de_st_list
    #         ],
    #         dim=1,
    #     )  # [N, 3, H, W]
    #     output_de_st = torch.prod(output_de_st, dim=1, keepdim=True)
    #
    #     mask = F.interpolate(
    #         mask,
    #         size=output_segmentation.size()[2:],
    #         mode="bilinear",
    #         align_corners=False,
    #     )
    #     mask = torch.where(
    #         mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
    #     )
    #     return output_segmentation, output_de_st, output_de_st_list, mask

class DESTSEG(nn.Module):
    def __init__(self, model_t, model_s):
        super(DESTSEG, self).__init__()
        self.net_t = get_model(model_t)
        self.net_s = get_model(model_s)
        self.destseg = DeSTSeg(teacher=self.net_t, student=self.net_s)
        self.frozen_layers = ['net_t']

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

    def forward(self, aug_img, ori_img=None, img_mask=None):
        output_segmentation, output_de_st, output_de_st_list, new_mask = self.destseg(aug_img, ori_img, img_mask)

        return output_segmentation, output_de_st, output_de_st_list, new_mask


@MODEL.register_module
def destseg(pretrained=False, **kwargs):
    model = DESTSEG(**kwargs)
    return model
