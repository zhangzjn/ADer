"""__init__.py - all efficientnet models.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).


__version__ = "0.7.1"
from .model import EfficientNet

__all__ = [
    "EfficientNet",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_b8",
    "efficientnet_l2",
]


def efficientnet_b0(pretrained, outblocks, outstrides, pretrained_model=""):
    return build_efficient(
        "efficientnet-b0", pretrained, outblocks, outstrides, pretrained_model
    )


def efficientnet_b1(pretrained, outblocks, outstrides, pretrained_model=""):
    return build_efficient(
        "efficientnet-b1", pretrained, outblocks, outstrides, pretrained_model
    )


def efficientnet_b2(pretrained, outblocks, outstrides, pretrained_model=""):
    return build_efficient(
        "efficientnet-b2", pretrained, outblocks, outstrides, pretrained_model
    )


def efficientnet_b3(pretrained, outblocks, outstrides, pretrained_model=""):
    return build_efficient(
        "efficientnet-b3", pretrained, outblocks, outstrides, pretrained_model
    )


def efficientnet_b4(pretrained, outblocks, outstrides, pretrained_model=""):
    return build_efficient(
        "efficientnet-b4", pretrained, outblocks, outstrides, pretrained_model
    )


def efficientnet_b5(pretrained, outblocks, outstrides, pretrained_model=""):
    return build_efficient(
        "efficientnet-b5", pretrained, outblocks, outstrides, pretrained_model
    )


def efficientnet_b6(pretrained, outblocks, outstrides, pretrained_model=""):
    return build_efficient(
        "efficientnet-b6", pretrained, outblocks, outstrides, pretrained_model
    )


def efficientnet_b7(pretrained, outblocks, outstrides, pretrained_model=""):
    return build_efficient(
        "efficientnet-b7", pretrained, outblocks, outstrides, pretrained_model
    )


def efficientnet_b8(pretrained, outblocks, outstrides, pretrained_model=""):
    return build_efficient(
        "efficientnet-b8", pretrained, outblocks, outstrides, pretrained_model
    )


def efficientnet_l2(pretrained, outblocks, outstrides, pretrained_model=""):
    return build_efficient(
        "efficientnet-l2", pretrained, outblocks, outstrides, pretrained_model
    )


def build_efficient(model_name, pretrained, outblocks, outstrides, pretrained_model=""):
    if pretrained:
        model = EfficientNet.from_pretrained(
            model_name,
            outblocks=outblocks,
            outstrides=outstrides,
            pretrained_model=pretrained_model,
        )
    else:
        model = EfficientNet.from_name(
            model_name, outblocks=outblocks, outstrides=outstrides
        )
    return model
