<div align="center">
  <img src="assets/ADer_git_1024.png" width="30%" />
</div>

> - <span style="color:red">**ADer**</span> is an open source visual **A**nomaly **D**etection toolbox based on PyTorch, which supports multiple popular AD datasets and approaches. </br> 
> - We reproduce popular AD methods under the **M**ulti-class **U**nsupervised **A**nomaly **D**etection (<span style="color:red">**MUAD**</span>) by default. </br>
> - We hope it can bring convenience to your research and application. </br>

[//]: # (<p align="center">)

[//]: # (    <a href="https://zhangzjn.github.io/"><strong>Jiangning Zhang<sup>*</strong></a>)

[//]: # (    ·)

[//]: # (</p>)

## 🐉 News
- 🔥 We have released several reproduced models, configuration files, and training logs in our <span style="color:red">**benchmark**</span> paper 🐲 [**Paper**](https://arxiv.org/abs/2406.03262) | [**Results & CFGs**](configs/benchmark/README.md)
- 🔥 <span style="color:red">**COCO-AD**</span> and powerful <span style="color:red">**InvAD**</span> is released 🐲 [**Paper**](https://arxiv.org/abs/2404.10760) | [**Project**](https://zhangzjn.github.io/projects/InvAD) | [**Code**](https://github.com/zhangzjn/ader/configs/invad)
- 🔥 Plain ViT based <span style="color:red">**ViTAD**</span> is released 🐲 [**Paper**](https://arxiv.org/abs/2312.07495) | [**Project**](https://zhangzjn.github.io/projects/ViTAD) | [**Code**](https://github.com/zhangzjn/ader/configs/vitad)
- 🔥 <span style="color:red">**Real-IAD**</span> is released: a new large-scale challenging industrial AD dataset 🐲 [**Paper**](https://arxiv.org/abs/2403.12580) | [**Project**](https://realiad4ad.github.io/Real-IAD) | [**Code**](https://github.com/TencentYoutuResearch/AnomalyDetection_Real-IAD) 

## 💡 Property

- [x] 🚀 Multi-/Single-class Training and Testing
- [x] 🚀 Convenient and flexible way to implement a new approach, refer to [here](#How-to-Build-a-Custom-Approach).
- [x] Reproduced popular methods in [ADer Benchmark](https://arxiv.org/abs/2406.03262): 
  - 🚀Augmentation-based
    - [x] [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM)
    - [x] [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet)
    - [x] [RealNet, CVPR'24](https://github.com/cnulab/RealNet)
  - 🚀Embedding-based
    - [x] [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization)
    - [x] [PatchCore, CVPR'22](https://github.com/amazon-science/patchcore-inspection)
    - [x] [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad)
    - [x] [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow)
  - 🚀Reconstruction-based
    - [x] [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD)
    - [x] [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD)
    - [x] [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD)
    - [ ] [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD): See tripartite implementation in this [website](https://github.com/lewandofskee/DiAD)
    - [x] [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD)
    - [x] [RD, CVPR'22](https://github.com/hq-deng/RD4AD)
  - 🚀Hybrid
    - [x] [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD): download  in `TIMM` to `model/pretrain`
    - [x] [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation)
    - [x] [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg)
- [x] 🚀 (Opt.) DDP Training
- [x] By default, the weights used by different methods are automatically downloaded (`model.kwargs.pretrained=True`, `model.kwargs.checkpoint_path=''`). If you prefer to specify offline weights, you can download the model weights to `model/pretrain` and modify the settings to `model.kwargs.pretrained=False`, `model.kwargs.checkpoint_path='model/pretrain/xxx.pth'`. 

[//]: # (- Some pre-trained weights that may be used: )
[//]: # (  - [wide_resnet50_2]&#40;https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth&#41; in `TIMM`)
[//]: # (  - [efficientnet_b4]&#40;https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth&#41; in `TIMM`)



---
## 🛠️ Getting Started

### Installation
- Clone this repo:

  ```shell
  git clone https://github.com/zhangzjn/ader.git && cd ader
  ```
- Prepare general experimental environment
  ```shell
  pip3 install timm==0.8.15dev0 mmselfsup pandas transformers openpyxl imgaug numba numpy tensorboard fvcore accimage Ninja
  pip3 install mmdet==2.25.3
  pip3 install --upgrade protobuf==3.20.1 scikit-image faiss-gpu
  pip3 install adeval
  pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
  pip3 install fastprogress geomloss FrEIA mamba_ssm adeval fvcore==0.1.5.post20221221
  (or) conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
  
### Dataset Preparation 
Please refer to [Datasets Description](data/README.md) for preparing visual AD datasets as needed.
- [x] [Real-IAD](data/README.md/###Real-IAD): A new large-scale challenging industrial AD dataset, containing 30 classes with totally 151,050 images; 2,000 ∼ 5,000 resolution; 0.01% ~ 6.75% defect proportions; 1:1 ~ 1:10 defect ratio.
- [x] [COCO-AD](data/README.md/###COCO-AD): Large-scale and general-purpose challenging AD-adapted dataset.
- [x] [MVTec AD](data/README.md/###MVTec AD): Most popular AD dataset.
- [x] [VisA](data/README.md/###VisA): Popular AD dataset.
- [x] [Uni-Medical](data/README.md/###Uni-Medical): Unified medical AD dataset.
- [x] (Opt.) [MVTec 3D-AD](data/README.md/###MVTec 3D-AD): Improved 3D version of MVTec AD.
- [x] (Opt.) [Cifar10 & Cifar100](data/README.md/###Cifar): For one-class-train, one-class-test, and unified settings.
- [x] (Opt.) [Tiny-ImageNet](data/README.md/###Tiny-ImageNet-200): A larger one-class dataset.

### Train (Multi-class Unsupervised AD setting by default, MUAD)
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
- Single-class Unsupervised AD (SUAD, not recommend currently)
  1. one `GPU-0` for training one `grid` class in `mvtec` dataset: `CUDA_VISIBLE_DEVICES=0 python3 run.py -c configs/vitad/single_cls/vitad_mvtec_bs16.py -m train data.cls_names=grid trainer.checkpoint=runs/vitad/single_class/vitad_mvtec_bs16/grid`
  2. one `GPU-0` for training all classes serially in `mvtec` dataset: `python3 runs_single_class.py -d mvtec -c configs/vitad/single_cls/vitad_mvtec_bs16.py -n 1 -m -1 -g 0`
  3. `$GPU_NUM` GPUs for training all classes parallelly in `mvtec` dataset:: `python3 runs_single_class.py -d mvtec -c configs/vitad/single_cls/vitad_mvtec_bs16.py -n $GPU_NUM -m 1`
  4. results will be saved in default dir: `runs/vitad/single_cls/vitad_mvtec_bs16`


### Test
- Modify `trainer.resume_dir` or `model.kwargs['checkpoint_path']`
- Test with single GPU example: `CUDA_VISIBLE_DEVICES=0 python run.py -c configs/METHOD/METHOD_cfg.py -m test`
- Test with multiple GPUs (DDP) in one node:  `python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port --use_env run.py -c configs/METHOD/METHOD_CFG.py -m test`.

### How to Build a Custom Approach
1. Add a model config `cfg_model_MODEL_NAME` to `configs/__base__`
2. Add configs to `configs/MODEL_NAME/CFG.py` for training and testing.
3. Add a model implementation file `model/MODEL_NAME.py`
4. Add a trainer implementation file `trainer/MODEL_NAME_trainer.py`
5. (Optional) Add specific files to `data`, `loss`, `optim`, *etc*.

---

## 📜 MUAD Results on Popular AD Datasets
- <span style="color:red">**Red metrics**</span> are recommended for comprehensive evaluations.<br>
Subscripts `I`, `R`, and `P` represent `image-level`, `region-level`, and `pixel-level`, respectively.
- The following results are derived from the original paper. Since the [benchmark](configs/benchmark/README.md) re-runs all experiments on the same `L40S` platform, slight differences in the results are reasonable.

### MVTec AD
|                         Method                          | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> |<span style="color:red">mAU-PRO<sub>R</sub></span> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | m*F*1<sub>P/.2/.8</sub> | mAcc<sub>P/.2/.8</sub> |mIoU<sub>P/.2/.8</sub> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD<sub>I</sub></span> | <span style="color:red">mAD<sub>P</sub></span> | <span style="color:red">mAD<sub>.2/.8</sub></span> | <span style="color:red">mAD</span> |                                                                            <span style="color:blue">Download</span>                                                                            |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:----------------:|:-----------------------:|:--------:|:--------:|:--------:|:----------------------------------------------:|:----------------------------------------------:|:--------------------------------------------------:|:----------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      [UniAD](https://github.com/zhiyuanyou/UniAD)       |        97.5         | 99.1 | 97.3 | 90.7 | 97.0 | 45.1 | 50.4 |          22.4           | 37.5 | 13.9 | 34.2 |                      98.0                      |                      64.1                      |                        24.6                        |                82.4                | [log](https://drive.google.com/file/d/1fxS7cf_aqdiBF8VVK2u6uAf-utxyyH9O/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1RzaeuSU9dtf-q_HX0neT8ZtDD1jutks_/view?usp=drive_link) |
|    [DiAD](https://github.com/lewandofskee/DiAD)    |        97.2 | 99.0 | 96.5 | 90.7 | 96.8 | 52.6 | 55.5 | 19.5 | 40.7 | 12.0 | 21.3 | 97.6 | 68.3 | 24.1 |               84.0                 | [log](https://github.com/lewandofskee/DiAD) & [weight](https://github.com/lewandofskee/DiAD) |
|   [ViTAD](https://zhangzjn.github.io/projects/ViTAD)    |        98.3         | 99.4 | 97.3 | 91.4 | 97.7 | 55.3 | 58.7 |          30.9           | 40.8 | 20.4 | 42.6 |                      98.3                      |                      70.6                      |                        30.7                        |                85.4                | [log](https://drive.google.com/file/d/1rZv53vHbtz7NjL0quHt26bfSVu-DzK9c/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1kkJCrFBI-JdtCIi2ZOz79Bjgcf5tgfp9/view?usp=drive_link) |
|   [InvAD](https://zhangzjn.github.io/projects/InvAD)    |        98.9         | 99.6 | 98.1 | 94.1 | 98.2 | 57.6 | 60.1 |          34.6           | 46.9 | 23.0 | 43.7 |                      98.9                      |                      72.0                      |                        34.8                        |                86.7                | [log](https://drive.google.com/file/d/1rSBaQfn2DCWq2Ds75_kcsjiBiF1sT4U4/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1Dg67j-dTkQK_z6mr9ScwW4t4ONEyqWpQ/view?usp=drive_link) |
| [InvAD-lite](https://zhangzjn.github.io/projects/InvAD) |        98.2 |	99.2 |	97.2 |	97.3 |	55.0 |	58.1 |	92.7 |	32.6 |	47.1 |	21.3 |	41.7  |                      98.2                      |                      68.6                      |                        33.7                        |                85.4                |                                           [log](https://drive.google.com/file/d/1fxqUXJXqlAe2vJQboz-zf_HNTtsFYHNZ/view?usp=sharing) & [weight](https://drive.google.com/file/d/1aUxUIP3RSNArfij-rSQeKfEVaKhTMzyP/view?usp=sharing)                                           |

### VisA
|                         Method                          | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> |<span style="color:red">mAU-PRO<sub>R</sub></span> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | m*F*1<sub>P/.2/.8</sub> | mAcc<sub>P/.2/.8</sub> |mIoU<sub>P/.2/.8</sub> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD<sub>I</sub></span> | <span style="color:red">mAD<sub>P</sub></span> | <span style="color:red">mAD<sub>.2/.8</sub></span> | <span style="color:red">mAD</span> |                                                                            <span style="color:blue">Download</span>                                                                             |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:----------------:|:-----------------------:|:--------:|:--------:|:--------:|:----------------------------------------------:|:----------------------------------------------:|:--------------------------------------------------:|:----------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      [UniAD](https://github.com/zhiyuanyou/UniAD)       |        88.8         | 90.8 | 85.8 | 85.5 | 98.3 | 33.7 | 39.0 |          17.9           | 47.1 | 10.9 | 25.7 |                      88.4                      |                      57.0                      |                        25.3                        |                74.5                | [log](https://drive.google.com/file/d/1D9njahAg4GIItlO388Woc0KtYkBdPQh1/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1eSrKlNY9XAhrpcF289StiJ7VL4oFjNeF/view?usp=drive_link)  |
|       [DiAD](https://github.com/lewandofskee/DiAD)       |        86.8 | 88.3 | 85.1 | 75.2 | 96.0 | 26.1 | 33.0 | 13.2 | 46.2 | 8.0 | 16.2 | 86.7 | 51.7 | 22.5 |               70.1                 | [log](https://github.com/lewandofskee/DiAD) & [weight](https://github.com/lewandofskee/DiAD)  |
|   [ViTAD](https://zhangzjn.github.io/projects/ViTAD)    |        90.5         | 91.7 | 86.3 | 85.1 | 98.2 | 36.6 | 41.1 |          21.6           | 38.2 | 13.5 | 27.6 |                      89.5                      |                      58.7                      |                        24.4                        |                75.6                | [log](https://drive.google.com/file/d/1ZTlom8ciynuxd5CuYoA3Ws70NWj5Wkfp/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1Xu-mssogQN-j6TTPIasbxwTkOZwy-u1P/view?usp=drive_link)  |
|   [InvAD](https://zhangzjn.github.io/projects/InvAD)    |        95.5         | 95.8 | 92.1 | 92.5 | 98.9 | 43.1 | 47.0 |          28.0           | 45.6 | 17.9 | 32.7 |                      94.5                      |                      63.0                      |                        30.5                        |                80.7                | [log](https://drive.google.com/file/d/1f6_S3wcD-KJX-tiluPgTRkQXfRKuGeKD/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1i2INJXWUohF_PM2ySz-Uztf3JiOu57aq/view?usp=drive_link)  |
| [InvAD-lite](https://zhangzjn.github.io/projects/InvAD) |        94.9 |	95.2 |	90.8 |	98.6 |	40.3 |	44.3 |	92.5 |	25.8 |	44.2 |	16.1 |	30.0  |                      93.6                      |                      59.0                      |                        28.7                        |                79.5                |                                           [log](https://drive.google.com/file/d/1m5RAQgQJHKG-0751DLPQOYC7yUjDk4Lg/view?usp=sharing) & [weight](https://drive.google.com/file/d/1fiX0lIWO0Kd_PX3PGN36VbHnH35CzPfG/view?usp=sharing)                                           |

### COCO-AD
|                         Method                          |                           mAU-ROC<sub>I</sub>                           | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> |<span style="color:red">mAU-PRO<sub>R</sub></span> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | m*F*1<sub>P/.2/.8</sub> | mAcc<sub>P/.2/.8</sub> |mIoU<sub>P/.2/.8</sub> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD<sub>I</sub></span> | <span style="color:red">mAD<sub>P</sub></span> | <span style="color:red">mAD<sub>.2/.8</sub></span> | <span style="color:red">mAD</span> |                                                                  <span style="color:blue">Download</span>                                                                   |
|:-------------------------------------------------------:|:-----------------------------------------------------------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:----------------:|:-----------------------:|:--------:|:--------:|:--------:|:----------------------------------------------:|:----------------------------------------------:|:--------------------------------------------------:|:----------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      [UniAD](https://github.com/zhiyuanyou/UniAD)       |                                  56.2                                   |      49.0       | 61.8 | 31.7 | 65.4 | 12.9 | 19.4 |           6.6           | 26.3 | 3.7 | 11.1 |                      55.7                      |                      32.6                      |                        12.2                        |                42.3                |                                         [log](https://drive.google.com/drive/folders/1M5iCcWqqtpX-E9eRD18LDYBfKUlLqCWL?usp=sharing) & [weight]()                                         |
|       [DiAD](https://github.com/lewandofskee/DiAD)       |59.0 | 53.0 | 63.2 | 30.8 | 68.1 | 20.5 | 14.2 | 9.6 | 31.1 | 6.1 | 11.6 | 58.4 | 34.2 | 15.6 |               44.1                 |                                         [log](https://github.com/lewandofskee/DiAD) & [weight]()                                         |
|   [ViTAD](https://zhangzjn.github.io/projects/ViTAD)    |                                  69.3                                   |      60.4       | 64.9 | 41.0 | 78.3 | 27.9 | 31.9 |          12.4           | 37.4 | 7.2 | 19.8 |                      64.9                      |                      46.0                      |                        19.0                        |                53.4                |                                         [log](https://drive.google.com/drive/folders/1D9MFsUOJBHPvEKY2ZR6eAw4VzvbQ0XkF?usp=sharing) & [weight]()                                         |
|   [InvAD](https://zhangzjn.github.io/projects/InvAD)    |                                  65.9                                   |      57.8       | 64.1 | 44.9 | 73.2 | 19.7 | 25.4 |          12.4           | 37.5 | 7.1 | 15.2 |                      62.6                      |                      39.4                      |                        19.0                        |                50.1                |                                         [log](https://drive.google.com/drive/folders/1D9MFsUOJBHPvEKY2ZR6eAw4VzvbQ0XkF?usp=sharing) & [weight]()                                         |
| [InvAD-lite](https://zhangzjn.github.io/projects/InvAD) |     64.7 |	56.7 |	63.5 |	70.5 |	18.3 |	23.4 |	38.2 	|     11.2 	      |34.2 |	6.4 |	13.8     |                      61.6                      |                      26.6                      |                        17.3                        |                47.9                |                                         [log](https://drive.google.com/drive/folders/1dsKrGOXP7Npt95GDfhuxv0cADUEnn0cX?usp=sharing) & [weight]()                                         |

### MVTec 3D-AD (RGB)
|                         Method                          | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> |<span style="color:red">mAU-PRO<sub>R</sub></span> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | m*F*1<sub>P/.2/.8</sub> | mAcc<sub>P/.2/.8</sub> |mIoU<sub>P/.2/.8</sub> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD<sub>I</sub></span> | <span style="color:red">mAD<sub>P</sub></span> | <span style="color:red">mAD<sub>.2/.8</sub></span> | <span style="color:red">mAD</span> |                                                                            <span style="color:blue">Download</span>                                                                            |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:----------------:|:-----------------------:|:--------:|:--------:|:--------:|:----------------------------------------------:|:----------------------------------------------:|:--------------------------------------------------:|:----------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      [UniAD](https://github.com/zhiyuanyou/UniAD)       |        78.9 | 93.4 | 91.4 | 88.1 | 96.5 | 21.2 | 28.0 | 12.2 | 43.6 | 7.0 | 16.8 |                      87.9                      |                      48.6                      |                        20.9                        |                71.1                | [log](https://drive.google.com/file/d/1nO5DyG5EiBJuZb9_5BSQx5tJOfzATX9M/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1ihxOr9AJoUP3lryL_FNnIbXq75xx4QcB/view?usp=drive_link) |
|       [DiAD](https://github.com/lewandofskee/DiAD)       |84.6 | 94.8 | 95.6 | 87.8 | 96.4 | 25.3 | 32.3 | 5.0 | 71.4 | 2.6 | 5.4 | 91.7 | 51.3 | 26.3 |               73.8                 | [log](https://github.com/lewandofskee/DiAD) & [weight](https://github.com/lewandofskee/DiAD) |
|   [ViTAD](https://zhangzjn.github.io/projects/ViTAD)    |        79.0 | 93.1 | 91.8 | 91.6 | 98.2 | 27.3 | 33.3 | 17.2 | 45.3 | 10.0 | 20.5 |                      88.0                      |                      52.9                      |                        24.1                        |                73.5                | [log](https://drive.google.com/file/d/1Ff2Z1AYzJqSvxa4qMdotgkPhoLDAtS2B/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1hn2qD7LuyASfnfvQxFDobWnLToCmPJ1D/view?usp=drive_link) |
|   [InvAD](https://zhangzjn.github.io/projects/InvAD)    |        86.1 | 95.8 | 93.2 | 94.7 | 98.8 | 37.8 | 42.5 | 22.0 | 50.5 | 13.2 | 27.5 |                      91.7                      |                      59.7                      |                        28.6                        |                78.4                | [log](https://drive.google.com/file/d/1je5NFQ1C5vLIEv_p7XgZM7d2kRCT5yyO/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1Icf0PLkKEWEridPb1Pn5D-qF76YCPE7O/view?usp=drive_link) |
| [InvAD-lite](https://zhangzjn.github.io/projects/InvAD) |        85.3 |	95.2 |	93.0 |	98.6 |	37.2 |	41.4 |	94.1 |	21.6 |	55.3 |	12.9 |	26.5  |                      91.2                      |                      57.6                      |                        30.0                        |                77.8                |                                           [log](https://drive.google.com/file/d/1Dr_M0uzuMuW4JbrZCMobcjV8o3JIevhg/view?usp=sharing) & [weight](https://drive.google.com/file/d/1nYuPTgectAUIpXNB3PgGEWZz1sda2F51/view?usp=sharing)                                           |

### Uni-Medical
|                         Method                          | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> |<span style="color:red">mAU-PRO<sub>R</sub></span> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | m*F*1<sub>P/.2/.8</sub> | mAcc<sub>P/.2/.8</sub> |mIoU<sub>P/.2/.8</sub> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD<sub>I</sub></span> | <span style="color:red">mAD<sub>P</sub></span> | <span style="color:red">mAD<sub>.2/.8</sub></span> | <span style="color:red">mAD</span> |                                                                           <span style="color:blue">Download</span>                                                                            |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:----------------:|:-----------------------:|:--------:|:--------:|:--------:|:----------------------------------------------:|:----------------------------------------------:|:--------------------------------------------------:|:----------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      [UniAD](https://github.com/zhiyuanyou/UniAD)       |        78.5 |      	75.2      |	76.6 |	96.4 	|         37.6          |	40.2 |	85.0 |	13.3 |	37.8 |	8.0 |	26.8  |                      76.8                      |                      54.3                      |                        19.7                        |                69.9                |                                                                                     [log](https://drive.google.com/file/d/1fFrg-SFTcZYdFMZPfCDDEdI3WFeY8r14/view?usp=sharing) & [weight](https://drive.google.com/file/d/1XWaE3K_644Ym09Av2IUE0zOkOaVqEjLS/view?usp=sharing)                                                                                      |
|       [DiAD](https://github.com/lewandofskee/DiAD)       |85.1 | 84.5 | 81.2 | 85.4 | 95.9 | 38.0 | 35.6 | 19.0 | 57.6 | 11.4 | 25.0 | 83.6 | 56.5 | 29.3 |               72.2                 |  [log](https://github.com/lewandofskee/DiAD) & [weight](https://github.com/lewandofskee/DiAD)                                                                                      |
|   [ViTAD](https://zhangzjn.github.io/projects/ViTAD)    |        82.2 |      	81.0      |	80.1 |	97.2 |	49.9 |	49.6 |	86.1 |	18.6 |	36.5 |	11.7 |	35.1  |                      81.1                      |                      61.9                      |                        22.3                        |                75.2                |                                                                                     [log](https://drive.google.com/file/d/1kvPpih1m0KbSFBZqZTs2INEalAN6PPgz/view?usp=sharing) & [weight](https://drive.google.com/file/d/1xFuMzO0enpBGxVu78HeQbkwV8QBzi9Ex/view?usp=sharing)                                                                                      |
|   [InvAD](https://zhangzjn.github.io/projects/InvAD)    |        82.2 |      	79.6      |	80.6 |	97.4 |	47.5 |	47.1 |	89.6 |	21.8 |	45.2 |	13.8 |	33.3 |                      80.8                      |                      61.4                      |                        26.9                        |                74.9                |                                           [log](https://drive.google.com/file/d/1lKlKfsx3xQQM0u7nkxp9GreEGaSv8m34/view?usp=sharing) & [weight](https://drive.google.com/file/d/1jwd3ly9OXFDwZR7DMH067fRjOZVXauTY/view?usp=sharing)                                           |
| [InvAD-lite](https://zhangzjn.github.io/projects/InvAD) |        79.5 |	78.3 |	79.1 |	96.4 |	40.1 |	40.4 |	85.5 	|      18.3       |	40.5 |	11.2 |	27.6 |                      79.0                      |                      55.3                      |                        23.3                        |                71.3                |                                           [log](https://drive.google.com/file/d/1MU9vDqt02W7c6pq46jKYiajXLEe2HRqJ/view?usp=sharing) & [weight](https://drive.google.com/file/d/1BSnOEyS6Dr9PTYtBQYziwnsuOw6fdNIH/view?usp=sharing)                                           |

### Real-IAD
|                         Method                          | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | m*F*1<sub>P/.2/.8</sub> | mAcc<sub>P/.2/.8</sub> | mIoU<sub>P/.2/.8</sub> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD<sub>I</sub></span> | <span style="color:red">mAD<sub>P</sub></span> | <span style="color:red">mAD<sub>.2/.8</sub></span>| <span style="color:red">mAD</span> |                                                                         <span style="color:blue">Download</span>                                                                         |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:--------------------------------------------------:|:-------------------:|:---------------------:|:---------------------:|:-----------------------:|:----------------------:|:----------------------:|:---------------------------------------------------:|:--------:|:----------------------------------------------:|:--------:|:----------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      [UniAD](https://github.com/zhiyuanyou/UniAD)       |        82.9         |      80.8       |        	74.4          |                      	97.4                         |        	22.9        |	30.3 	|         86.4          |	10.5 |         	35.0          |          6.0           |                        	18.3                        |	79.4 |                     	46.5                      |	17.2 	|                67.9                | [log](https://drive.google.com/file/d/1FKPNy5PJpTd-j586qphB33tl5fSArsI4/view?usp=sharing) & [weight](https://drive.google.com/file/d/1a7QB30THNhoNgBih8m0o3f_G6g0GEX7j/view?usp=sharing) |
|       [DiAD](https://github.com/lewandofskee/DiAD)       |75.6 | 66.4 | 69.9 | 58.1 | 88.0 | 2.9 | 7.1 | 2.9 | 41.9 | 1.5 | 3.7 | 70.6 | 32.7 | 15.4 |               52.6                 |                                               [log](https://github.com/lewandofskee/DiAD) & [weight](https://github.com/lewandofskee/DiAD)                                               |
|   [ViTAD](https://zhangzjn.github.io/projects/ViTAD)    |        82.7 |	80.1 |	73.7 |	97.3 	|        24.2         |	32.3 |	83.9 |	13.4 |	27.3 |	7.7 |	19.6 	|               78.8 |	46.8 |	16.1 |               	67.8                | [log](https://drive.google.com/file/d/1gRBBL8EXkQeeiWspQhKTQEPSihBIGZP6/view?usp=sharing) & [weight](https://drive.google.com/file/d/1M7s6bxUZUmMFPcbCYpK6Akgpplpclx1n/view?usp=sharing) |
|   [InvAD](https://zhangzjn.github.io/projects/InvAD)    |        89.0 |	86.4 |	79.6 |	98.4 |	30.7 |	37.6 |	91.9 |	17.7 |	36.1 |	10.4 |	23.5 |	85.0 |	53.4 |	21.4 |               	73.4                | [log](https://drive.google.com/file/d/1lquuvY4MbGm5ZJrzdF0iZGzOkG89GEpS/view?usp=sharing) & [weight](https://drive.google.com/file/d/1B62_FZOvcdpjb0o3AOlcXb6cGrSWk0vt/view?usp=sharing) |
| [InvAD-lite](https://zhangzjn.github.io/projects/InvAD) |        87.2 |	85.1 |	77.8 |	98.1 |	31.6 |	37.9 |	91.6 |	17.6 |	36.8 |	10.3 |	23.7 |	83.3 |	53.7 |	21.6 |               	72.7                | [log](https://drive.google.com/file/d/1ftzREsAY7iZwDsqQVfDbL_4Yq7DEa4Bn/view?usp=sharing) & [weight](https://drive.google.com/file/d/1G0Wsnai0gbQZPX6lk-dSJ8WXu8nNCCuH/view?usp=sharing) |



## Citation
If you use this toolbox or benchmark in your research, please cite our related works.
```angular2html
@article{ader,
  title={ADer: A Comprehensive Benchmark for Multi-class Visual Anomaly Detection},
  author={Jiangning Zhang and Haoyang He and Zhenye Gan and Qingdong He and Yuxuan Cai and Zhucun Xue and Yabiao Wang and Chengjie Wang and Lei Xie and Yong Liu},
  journal={arXiv preprint arXiv:2406.03262},
  year={2024}
}

@inproceedings{realiad,
  title={Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection},
  author={Wang, Chengjie and Zhu, Wenbing and Gao, Bin-Bin and Gan, Zhenye and Zhang, Jianning and Gu, Zhihao and Qian, Shuguang and Chen, Mingang and Ma, Lizhuang},
  booktitle={CVPR},
  year={2024}
}

@article{vitad,
  title={Exploring Plain ViT Reconstruction for Multi-class Unsupervised Anomaly Detection},
  author={Zhang, Jiangning and Chen, Xuhai and Wang, Yabiao and Wang, Chengjie and Liu, Yong and Li, Xiangtai and Yang, Ming-Hsuan and Tao, Dacheng},
  journal={arXiv preprint arXiv:2312.07495},
  year={2023}
}

@article{invad,
  title={Learning Feature Inversion for Multi-class Anomaly Detection under General-purpose COCO-AD Benchmark},
  author={Jiangning Zhang and Chengjie Wang and Xiangtai Li and Guanzhong Tian and Zhucun Xue and Yong Liu and Guansong Pang and Dacheng Tao},
  journal={arXiv preprint arXiv:2404.10760},
  year={2024}
}

@article{mambaad,
  title={MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection},
  author={He, Haoyang and Bai, Yuhu and Zhang, Jiangning and He, Qingdong and Chen, Hongxu and Gan, Zhenye and Wang, Chengjie and Li, Xiangtai and Tian, Guanzhong and Xie, Lei},
  year={2024}
}

```