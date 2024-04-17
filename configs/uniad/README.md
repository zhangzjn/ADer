# UniAD

> [A Unified Model for Multi-class Anomaly Detection, NeurIPS'22](https://papers.nips.cc/paper_files/paper/2022/hash/1d774c112926348c3e25ea47d87c835b-Abstract-Conference.html) </br>
> [**Paper**](https://papers.nips.cc/paper_files/paper/2022/hash/1d774c112926348c3e25ea47d87c835b-Abstract-Conference.html) | [**Code**](https://github.com/zhiyuanyou/UniAD)

## MUAD Results, Models, and Logs

|                      Dataset                       | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> |<span style="color:red">mAU-PRO<sub>R</sub></span> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | m*F*1<sub>P/.2/.8</sub> | mAcc<sub>P/.2/.8</sub> |mIoU<sub>P/.2/.8</sub> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD<sub>I</sub></span> | <span style="color:red">mAD<sub>P</sub></span> | <span style="color:red">mAD<sub>.2/.8</sub></span>| <span style="color:red">mAD</span> |                                                                            <span style="color:blue">Download</span>                                                                            |
|:--------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:----------------:|:-----------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [MVTec AD](data/README.md/###MVTec) |        97.5         | 99.1 | 97.3 | 90.7 | 97.0 | 45.1 | 50.4 |          22.4           | 37.5 | 13.9 | 34.2 | 98.0 | 64.1 | 24.6 | 82.4| [log](https://drive.google.com/file/d/1fxS7cf_aqdiBF8VVK2u6uAf-utxyyH9O/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1RzaeuSU9dtf-q_HX0neT8ZtDD1jutks_/view?usp=drive_link) |
|                     [VisA](data/README.md/###VisA)                      |        88.8         | 90.8 | 85.8 | 85.5 | 98.3 | 33.7 | 39.0 |          17.9           | 47.1 | 10.9 | 25.7 | 88.4 | 57.0 | 25.3 | 74.5| [log](https://drive.google.com/file/d/1D9njahAg4GIItlO388Woc0KtYkBdPQh1/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1eSrKlNY9XAhrpcF289StiJ7VL4oFjNeF/view?usp=drive_link)  |
|                     [COCO-AD](data/README.md/###COCO-AD)                      |        56.2         | 49.0 | 61.8 | 31.7 | 65.4 | 12.9 | 19.4 |           6.6           | 26.3 | 3.7 | 11.1 | 55.7 | 32.6 | 12.2 | 42.3|           [log](https://drive.google.com/drive/folders/1M5iCcWqqtpX-E9eRD18LDYBfKUlLqCWL?usp=sharing) & [weight]()           |
|                     [MVTec 3D-AD (RGB)](data/README.md/###MVTec3D)                      |        78.9 | 93.4 | 91.4 | 88.1 | 96.5 | 21.2 | 28.0 | 12.2 | 43.6 | 7.0 | 16.8 | 87.9 | 48.6 | 20.9 | 71.1| [log](https://drive.google.com/file/d/1nO5DyG5EiBJuZb9_5BSQx5tJOfzATX9M/view?usp=drive_link) & [weight](https://drive.google.com/file/d/1ihxOr9AJoUP3lryL_FNnIbXq75xx4QcB/view?usp=drive_link) |
|                     [Real-IAD](data/README.md/###Real-IAD)                      |        82.9         |      80.8       |        	74.4          |                      	97.4                         |        	22.9        |	30.3 	|         86.4          |	10.5 |         	35.0          |          6.0           |                        	18.3                        |	79.4 |                     	46.5                      |	17.2 	|67.9 |                                                                          [log](https://drive.google.com/file/d/1FKPNy5PJpTd-j586qphB33tl5fSArsI4/view?usp=sharing) & [weight](https://drive.google.com/file/d/1a7QB30THNhoNgBih8m0o3f_G6g0GEX7j/view?usp=sharing) |
| [Uni-Medical](data/README.md/###Uni-Medical) |        78.5 |      	75.2      |	76.6 |	96.4 	|         37.6          |	40.2 |	85.0 |	13.3 |	37.8 |	8.0 |	26.8  |                      76.8                      |                      54.3                      |                        19.7                        |                69.9                |                                                                                     [log](https://drive.google.com/file/d/1fFrg-SFTcZYdFMZPfCDDEdI3WFeY8r14/view?usp=sharing) & [weight](https://drive.google.com/file/d/1XWaE3K_644Ym09Av2IUE0zOkOaVqEjLS/view?usp=sharing)                                                                                      |

## Citation
```angular2html
@inproceedings{uniad,
  title={A unified model for multi-class anomaly detection},
  author={You, Zhiyuan and Cui, Lei and Shen, Yujun and Yang, Kai and Lu, Xin and Zheng, Yu and Le, Xinyi},
  booktitle={NeurIPS},
  year={2022}
}
```
