# ADer

ADer is an open source visual anomaly detection toolbox based on PyTorch, which supports multiple popular AD datasets and approaches. <br>
**Full codes will be available soon**

---
## Property
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

---
# Getting Started


## Installation
- Clone this repo:

  ```shell
  git clone https://github.com/zhangzjn/ADer.git && cd ADer
  ```
- Prepare general experimental environment
  ```shell
  pip3 install timm==0.8.15dev0 mmselfsup pandas transformers openpyxl imgaug numba numpy tensorboard fvcore accimage Ninja
  pip3 install mmdet==2.25.3
  pip3 install --upgrade protobuf==3.20.1 scikit-image faiss-cpu faiss-gpu
  pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
  pip3 install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
  ```
  
## Dataset Preparation 
Please refer to [Datasets Description](data/README.md) for preparing visual AD datasets.
- [x] [MVTec AD](data/README.md/###MVTec)
- [x] [VisA](data/README.md/###VisA)
- [x] [MVTec 3D-AD](data/README.md/###MVTec3D)
- [x] [Cifar10 & Cifar100](data/README.md/###Cifar) for one-class-train, one-class-test, and unified settings
- [x] [Tiny-ImageNet](data/README.md/###Tiny-ImageNet-200)

## Results on Popular Datasets
<span style="color:red">**Red metrics**</span> are recommended for comprehensive evaluations.<br>
Subscripts `I`, `R`, and `P` represent `image-level`, `region-level`, and `pixel-level`, respectively.

### MUAD on MVTec AD
|                       Method                       | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> |<span style="color:red">mAU-PRO<sub>R</sub></span> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | m*F*1<sub>P/.2/.8</sub> | mAcc<sub>P/.2/.8</sub> |mIoU<sub>P/.2/.8</sub> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD<sub>I</sub></span> | <span style="color:red">mAD<sub>P</sub></span> | <span style="color:red">mAD<sub>.2/.8</sub></span>| <span style="color:red">mAD</span> |                                                                            <span style="color:blue">Download</span>                                                                            |
|:--------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:----------------:|:-----------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     [DRAEM](https://github.com/VitjanZ/DRAEM)      |        88.8         | 94.7 | 92.0 | 71.1 | 88.6 | 52.6 | 48.6 |          21.8           | 15.3 | 14.2 | 35.1 | 91.8 | 63.2 | 17.1 | 76.6| [log](https://drive.google.com/file/d/1B1lBa5qzq8nCpY7blXW2EcA3zcuyp2t3/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1BzOkg93kWkHna_cHzC0Dgh-zM9Psaegg/view?usp=drive_link) |
|       [RD](https://github.com/hq-deng/RD4AD)       |        94.6         | 96.5 | 95.2 | 91.2 | 96.1 | 48.6 | 53.8 |          25.8           | 39.8 | 16.4 | 37.4 | 95.4 | 66.2 | 27.4 | 82.3| [log](https://drive.google.com/file/d/1-GkbG3PR3-n2kuGLjwsLez4wxqg8n3my/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1RPiGMKjaApJ5F7pyE5Bz6G8NLo6mfWo2/view?usp=drive_link) |
|    [UniAD](https://github.com/zhiyuanyou/UniAD)    |        97.5         | 99.1 | 97.3 | 90.7 | 97.0 | 45.1 | 50.4 |          22.4           | 37.5 | 13.9 | 34.2 | 98.0 | 64.1 | 24.6 | 82.4| [log](https://drive.google.com/file/d/1fxS7cf_aqdiBF8VVK2u6uAf-utxyyH9O/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1RzaeuSU9dtf-q_HX0neT8ZtDD1jutks_/view?usp=drive_link) |
|   [DeSTSeg](https://github.com/apple/ml-destseg)   |        89.2         | 95.5 | 91.6 | 64.8 | 93.1 | 54.3 | 50.9 |          29.7           | 22.7 | 18.8 | 35.3 | 92.1 | 66.1 | 23.7 | 77.1|                                                                                               -                                                                                                |
| [SimpleNet](https://github.com/DonaldRR/SimpleNet) |        95.3         | 98.4 | 95.8 | 86.5 | 96.9 | 45.9 | 49.7 |          25.3           | 47.7 | 16.0 | 34.4 | 96.5 | 64.2 | 29.7 | 81.2|                                                                                               -                                                                                                |
|                     [ViTAD]()                      |        98.3         | 99.4 | 97.3 | 91.4 | 97.7 | 55.3 | 58.7 |          30.9           | 40.8 | 20.4 | 42.6 | 98.3 | 70.6 | 30.7 | 85.4| [log](https://drive.google.com/file/d/1rZv53vHbtz7NjL0quHt26bfSVu-DzK9c/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1kkJCrFBI-JdtCIi2ZOz79Bjgcf5tgfp9/view?usp=drive_link) |

### MUAD on VisA
|                       Method                       | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> |<span style="color:red">mAU-PRO<sub>R</sub></span> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | m*F*1<sub>P/.2/.8</sub> | mAcc<sub>P/.2/.8</sub> |mIoU<sub>P/.2/.8</sub> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD<sub>I</sub></span> | <span style="color:red">mAD<sub>P</sub></span> | <span style="color:red">mAD<sub>.2/.8</sub></span>| <span style="color:red">mAD</span> |                                                                            <span style="color:blue">Download</span>                                                                             |
|:--------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:----------------:|:-----------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     [DRAEM](https://github.com/VitjanZ/DRAEM)      |        79.5         | 82.8 | 79.4 | 59.1 | 91.4 | 24.8 | 30.4 |          12.6           | 8.7 | 7.4 | 18.8 | 80.5 | 48.8 | 9.6 | 63.9| [log](https://drive.google.com/file/d/1i95L83WdXCvcFCFwr6k_lA8bNd6FStC3/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1C44PxLzlZfhaaCD9D_P9aDHaIIfBic1U/view?usp=drive_link)  |
|       [RD](https://github.com/hq-deng/RD4AD)       |        92.4         | 92.4 | 89.6 | 91.8 | 98.1 | 38.0 | 42.6 |          21.2           | 46.5 | 13.1 | 28.5 | 91.5 | 59.6 | 26.9 | 77.8| [log](https://drive.google.com/file/d/1F5A7H84eiKRaENicLKnfXCuulAr7Cg_c/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1bDCQDf0UIgbCVW_9TwVjn4FhS7smVlD2/view?usp=drive_link)  |
|    [UniAD](https://github.com/zhiyuanyou/UniAD)    |        88.8         | 90.8 | 85.8 | 85.5 | 98.3 | 33.7 | 39.0 |          17.9           | 47.1 | 10.9 | 25.7 | 88.4 | 57.0 | 25.3 | 74.5| [log](https://drive.google.com/file/d/1D9njahAg4GIItlO388Woc0KtYkBdPQh1/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1eSrKlNY9XAhrpcF289StiJ7VL4oFjNeF/view?usp=drive_link)  |
|   [DeSTSeg](https://github.com/apple/ml-destseg)   |        88.9         | 89.0 | 85.2 | 67.4 | 96.1 | 39.6 | 43.4 |          27.4           | 41.0 | 17.3 | 26.9 | 87.7 | 59.7 | 28.6 | 72.8|                                                                                                -                                                                                                |
| [SimpleNet](https://github.com/DonaldRR/SimpleNet) |        87.2         | 87.0 | 81.7 | 81.4 | 96.8 | 34.7 | 37.8 |          17.5           | 50.6 | 11.0 | 25.9 | 85.3 | 56.4 | 26.3 | 72.4|                                                                                                -                                                                                                |
|                     [ViTAD]()                      |        90.5         | 91.7 | 86.3 | 85.1 | 98.2 | 36.6 | 41.1 |          21.6           | 38.2 | 13.5 | 27.6 | 89.5 | 58.7 | 24.4 | 75.6| [log](https://drive.google.com/file/d/1ZTlom8ciynuxd5CuYoA3Ws70NWj5Wkfp/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1Xu-mssogQN-j6TTPIasbxwTkOZwy-u1P/view?usp=drive_link)  |

Note: Each method trains 100 epochs.


### MUAD on MVTec 3D-AD RGB
|                       Method                       | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> |<span style="color:red">mAU-PRO<sub>R</sub></span> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | m*F*1<sub>P/.2/.8</sub> | mAcc<sub>P/.2/.8</sub> |mIoU<sub>P/.2/.8</sub> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD<sub>I</sub></span> | <span style="color:red">mAD<sub>P</sub></span> | <span style="color:red">mAD<sub>.2/.8</sub></span>| <span style="color:red">mAD</span> |                                                                            <span style="color:blue">Download</span>                                                                            |
|:--------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:----------------:|:-----------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     [DRAEM](https://github.com/VitjanZ/DRAEM)      |        63.2 | 86.1 | 89.2 | 55.0 | 93.2 | 16.8 | 20.2 | 4.3 | 2.5 | 2.4 | 11.9 | 79.5 | 43.4 | 3.0 | 60.5 | [log](https://drive.google.com/file/d/1TW9qAiySFtuNlRxwPSUOsIa38bz3sLAN/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1GmpEuQsa9fghsPtxaGUWf3oCy7L9PikO/view?usp=drive_link) |
|       [RD](https://github.com/hq-deng/RD4AD)       |        77.9 | 92.4 | 91.4 | 93.5 | 98.4 | 29.8 | 36.4 | 16.0 | 53.1 | 9.4 | 22.8 | 87.2 | 54.9 | 26.2 | 74.3| [log](https://drive.google.com/file/d/1YPLDl__PSGj45v-FFvsSFDSkL8Z6qU0W/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1vBZmDQ9iKF_24-iVemyPouL9RUukbpjK/view?usp=drive_link) |
|    [UniAD](https://github.com/zhiyuanyou/UniAD)    |        78.9 | 93.4 | 91.4 | 88.1 | 96.5 | 21.2 | 28.0 | 12.2 | 43.6 | 7.0 | 16.8 | 87.9 | 48.6 | 20.9 | 71.1| [log](https://drive.google.com/file/d/1nO5DyG5EiBJuZb9_5BSQx5tJOfzATX9M/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1ihxOr9AJoUP3lryL_FNnIbXq75xx4QcB/view?usp=drive_link) |
|                     [ViTAD]()                      |        79.0 | 93.1 | 91.8 | 91.6 | 98.2 | 27.3 | 33.3 | 17.2 | 45.3 | 10.0 | 20.5 | 88.0 | 52.9 | 24.1 | 73.5| [log](https://drive.google.com/file/d/1Ff2Z1AYzJqSvxa4qMdotgkPhoLDAtS2B/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1hn2qD7LuyASfnfvQxFDobWnLToCmPJ1D/view?usp=drive_link) |

## Train
- Check `data` and `model` settings for the config file `configs/METHOD/METHOD_CFG.py`
- Train with single GPU example: `CUDA_VISIBLE_DEVICES=0 python run.py -c configs/METHOD/METHOD_cfg.py -m train`
- Train with multiple GPUs (DDP) in one node: 
  - `export nproc_per_node=8`
  - `export nnodes=1`
  - `export node_rank=0`
  - `export master_addr=YOUR_MACHINE_ADDRESS`
  - `export master_port=12315`
  - `python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port --use_env run.py -c configs/METHOD/METHOD_CFG.py -m train`.
- Modify `trainer.resume_dir` to resume training.

## Test
- Modify `trainer.resume_dir` or `model.kwargs['checkpoint_path']`
- Test with single GPU example: `CUDA_VISIBLE_DEVICES=0 python run.py -c configs/METHOD/METHOD_cfg.py -m test`
- Test with multiple GPUs (DDP) in one node:  `python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port --use_env run.py -c configs/METHOD/METHOD_CFG.py -m test`.


## Citation
If you use this toolbox or benchmark in your research, please cite our related works.
```angular2html
@article{vitad,
  title={Exploring Plain ViT Reconstruction for Multi-class Unsupervised Anomaly Detection},
  author={Jiangning Zhang and Xuhai Chen and Yabiao Wang and Chengjie Wang and Yong Liu and Xiangtai Li and Ming-Hsuan Yang and Dacheng Tao},
  journal={arXiv preprint arXiv:2312.07495},
  year={2023}
}
```