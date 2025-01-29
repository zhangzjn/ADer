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
- 🔥 Plain ViT based <span style="color:red">**ViTAD**</span> is accepted by CVIU'25 🐲 [**Paper**](https://arxiv.org/abs/2312.07495) | [**Project**](https://zhangzjn.github.io/projects/ViTAD) | [**Code**](https://github.com/zhangzjn/ader/configs/vitad)
- 🔥 We have released several reproduced models, configuration files, and training logs in our <span style="color:red">**benchmark**</span> paper 🐲 [**Paper**](https://arxiv.org/abs/2406.03262) | [**Results & CFGs**](configs/benchmark/README.md)
- 🔥 <span style="color:red">**COCO-AD**</span> and powerful <span style="color:red">**InvAD**</span> is released 🐲 [**Paper**](https://arxiv.org/abs/2404.10760) | [**Project**](https://zhangzjn.github.io/projects/InvAD) | [**Code**](https://github.com/zhangzjn/ader/configs/invad)
- 🔥 <span style="color:red">**Real-IAD**</span> is released: a new large-scale challenging industrial AD dataset 🐲 [**Paper**](https://arxiv.org/abs/2403.12580) | [**Project**](https://realiad4ad.github.io/Real-IAD) | [**Code**](https://github.com/TencentYoutuResearch/AnomalyDetection_Real-IAD) 

## 💡 Property

- [x] 🚀Support Visualization
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

### Visualization
- Modify `trainer.resume_dir` or `model.kwargs['checkpoint_path']`
- Visualize with single GPU example: `CUDA_VISIBLE_DEVICES=0 python run.py -c configs/METHOD/METHOD_cfg.py -m test vis=True vis_dir=VISUALIZATION_DIR`

### How to Build a Custom Approach
1. Add a model config `cfg_model_MODEL_NAME` to `configs/__base__`
2. Add configs to `configs/MODEL_NAME/CFG.py` for training and testing.
3. Add a model implementation file `model/MODEL_NAME.py`
4. Add a trainer implementation file `trainer/MODEL_NAME_trainer.py`
5. (Optional) Add specific files to `data`, `loss`, `optim`, *etc*.

---

## 📜 MUAD Results on Popular AD Datasets
> Detailed results are available on the [benchmark page](configs/benchmark/README.md)


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
  journal={CVIU},
  year={2025}
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