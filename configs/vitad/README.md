# ViTAD

> [Exploring Plain Vision Transformer for Unsupervised Multi-class Anomaly Detection](https://arxiv.org/abs/2312.07495) </br>
> [**Paper**](https://arxiv.org/abs/2312.07495) | [**Project**](https://zhangzjn.github.io/projects/ViTAD) | [**Code**](https://github.com/zhangzjn/ader/configs/vitad)

## MUAD Results, Models, and Logs

|                         Dataset                          | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> |<span style="color:red">mAU-PRO<sub>R</sub></span> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | m*F*1<sub>P/.2/.8</sub> | mAcc<sub>P/.2/.8</sub> |mIoU<sub>P/.2/.8</sub> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD<sub>I</sub></span> | <span style="color:red">mAD<sub>P</sub></span> | <span style="color:red">mAD<sub>.2/.8</sub></span>| <span style="color:red">mAD</span> |                                                                            <span style="color:blue">Download</span>                                                                            |
|:--------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:----------------:|:-----------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|           [MVTec AD](data/README.md/###MVTec)            |        98.3         | 99.4 | 97.3 | 91.4 | 97.7 | 55.3 | 58.7 |          30.9           | 40.8 | 20.4 | 42.6 | 98.3 | 70.6 | 30.7 | 85.4| [log](https://drive.google.com/file/d/1rZv53vHbtz7NjL0quHt26bfSVu-DzK9c/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1kkJCrFBI-JdtCIi2ZOz79Bjgcf5tgfp9/view?usp=drive_link) |
|              [VisA](data/README.md/###VisA)              |        90.5         | 91.7 | 86.3 | 85.1 | 98.2 | 36.6 | 41.1 |          21.6           | 38.2 | 13.5 | 27.6 | 89.5 | 58.7 | 24.4 | 75.6| [log](https://drive.google.com/file/d/1ZTlom8ciynuxd5CuYoA3Ws70NWj5Wkfp/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1Xu-mssogQN-j6TTPIasbxwTkOZwy-u1P/view?usp=drive_link) |
|           [COCO-AD](data/README.md/###COCO-AD)           |        69.3         | 60.4 | 64.9 | 41.0 | 78.3 | 27.9 | 31.9 |          12.4           | 37.4 | 7.2 | 19.8 | 64.9 | 46.0 | 19.0 | 53.4|                                            [log](https://drive.google.com/drive/folders/1D9MFsUOJBHPvEKY2ZR6eAw4VzvbQ0XkF?usp=sharing) & [weight]()                                            |
|      [MVTec 3D-AD (RGB)](data/README.md/###MVTec3D)      |        79.0 | 93.1 | 91.8 | 91.6 | 98.2 | 27.3 | 33.3 | 17.2 | 45.3 | 10.0 | 20.5 | 88.0 | 52.9 | 24.1 | 73.5| [log](https://drive.google.com/file/d/1Ff2Z1AYzJqSvxa4qMdotgkPhoLDAtS2B/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1hn2qD7LuyASfnfvQxFDobWnLToCmPJ1D/view?usp=drive_link) |
|          [Real-IAD](data/README.md/###Real-IAD)          |        82.7 |	80.1 |	73.7 |	97.3 	|        24.2         |	32.3 |	83.9 |	13.4 |	27.3 |	7.7 |	19.6 	|               78.8 |	46.8 |	16.1 |	67.8                |                                     [log](https://drive.google.com/file/d/1gRBBL8EXkQeeiWspQhKTQEPSihBIGZP6/view?usp=sharing) & [weight](https://drive.google.com/file/d/1M7s6bxUZUmMFPcbCYpK6Akgpplpclx1n/view?usp=sharing)                                      |
| [Uni-Medical](data/README.md/###Uni-Medical) |        82.2 |      	81.0      |	80.1 |	97.2 |	49.9 |	49.6 |	86.1 |	18.6 |	36.5 |	11.7 |	35.1  |                      81.1                      |                      61.9                      |                        22.3                        |                75.2                |                                                                                     [log](https://drive.google.com/file/d/1kvPpih1m0KbSFBZqZTs2INEalAN6PPgz/view?usp=sharing) & [weight](https://drive.google.com/file/d/1xFuMzO0enpBGxVu78HeQbkwV8QBzi9Ex/view?usp=sharing)                                                                                      |

## Citation
```angular2html
@article{vitad,
  title={Exploring Plain ViT Reconstruction for Multi-class Unsupervised Anomaly Detection},
  author={Zhang, Jiangning and Chen, Xuhai and Wang, Yabiao and Wang, Chengjie and Liu, Yong and Li, Xiangtai and Yang, Ming-Hsuan and Tao, Dacheng},
  journal={arXiv preprint arXiv:2312.07495},
  year={2023}
}
```
