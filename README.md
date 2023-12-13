# ADer

ADer is an open source visual anomaly detection toolbox based on PyTorch, which supports multiple popular AD datasets and approaches. <br>
**Full codes will be available soon**

---
## Property
- [x] Support single-class setting
- [x] Multi-/Single-class Training and Testing
- [x] DDP Training
- [x] Reproduced popular counterparts: 
  - [x] [RD, CVPR'22](https://github.com/hq-deng/RD4AD): download [wide_resnet50_2](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth) in `TIMM` to `model/pretrain`
  - [x] [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD): download [efficientnet_b4](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth) in `TIMM` to `model/pretrain`
  - [x] [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM)
  - [x] [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet)
  - [x] [PatchCore, CVPR'22](https://github.com/amazon-science/patchcore-inspection)
  - [x] [PyramidFlow, CVPR'23](https://github.com/gasharper/PyramidFlow)
  - [ ] [DeSTSeg, CVPR'23](https://github.com/apple/ml-destseg)
- [x] [InvAD](): download [wide_resnet50_2](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth) in `TIMM` to `model/pretrain`
- [x] [ViTAD](https://github.com/zhangzjn/ADer)
