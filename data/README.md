# Datasets Descriptions for Anomaly Detection

---

## Both Semantic and Sensory AD
### COCO-AD
- Download and extract [COCO 2017](https://cocodataset.org/) into `data/coco`.
- run`python data/gen_benchmark/coco.py` to obtain `data/coco/meta_20_x.json` and `val2017_mask_ad_20_x`, where `x=0,1,2,3` represent splits.
- Use `DefaultAD` in `data/ad_dataset.py` as the dataloader.
```
data
├── coco
    ├── annotations
        ├── instances_train2017.json
        ├── instances_val2017.json
    ├── train2017
    ├── val2017
    ├── meta_20_0.json
    ├── val2017_mask_ad_20_0
        ├── 000000000139.png
        ├── 000000000724.png
```


## Sensory AD
### MVTec AD
- Download and extract [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) into `data/mvtec`.
- run`python data/gen_benchmark/mvtec.py` to obtain `data/mvtec/meta.json` that matches standard `DefaultAD` in `data/ad_dataset.py`.
```
data
├── mvtec
    ├── meta.json
    ├── bottle
        ├── train
            └── good
                ├── 000.png
        ├── test
            ├── good
                ├── 000.png
            ├── anomaly1
                ├── 000.png
        └── ground_truth
            ├── anomaly1
                ├── 000.png
```

### MVTec 3D-AD
- Download and extract [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) into `data/mvtec3d`.
- run`python data/gen_benchmark/mvtec.py` to obtain `data/mvtec3d/meta.json` that matches standard `DefaultAD` in `data/ad_dataset.py`.

### VisA
- Download and extract [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) into `data/visa`.
- Refer to [project page](https://github.com/amazon-science/spot-diff#data-preparation) for data preparation, and run.
- run`python data/gen_benchmark/visa.py` to obtain `data/visa/meta.json` that matches standard `DefaultAD` in `data/ad_dataset.py`.

### Real-IAD
- Download and extract [Real-IAD](https://realiad4ad.github.io/Real-IAD/) into `data/realiad`.

### Uni-Medical
- Download and extract [Uni-Medical](https://drive.google.com/file/d/1Q33X6UMS_2rfdOlHq-Levf7Df7z3tUKp/view?usp=sharing) into `data/medical`.
- Refer to [ViTAD](https://zhangzjn.github.io/projects/ViTAD) and [BMAD](https://arxiv.org/abs/2306.11876) for data declaration.

## Semantic AD
### Cifar
- Download and extract [Cifar10 / Cifar100]() into `data/cifar`
- Use `CifarAD` in `data/ad_dataset.py` as the dataloader, which covers three general settings for `Cifar10` and one unified setting for `Cifar100`.

| Dataset  |                    Setting                     |                                Description                                | 
|:--------:|:----------------------------------------------:|:-------------------------------------------------------------------------:|
| Cifar10  |  [Unified](https://arxiv.org/abs/2206.03687)   | 5 normals & 5 abnormals <br> 5x5,000 for train & 5x1,000+5x1,000 for test |
| Cifar10  |                One-Class-Train                 |  1 normal & 9 abnormals <br> 1x5,000 for train & 1x1,000+1,000 for test   |
| Cifar10  |                 One-Class-Test                 |  9 normals & 1 abnormal <br> 9x5,000 for train & 9x1,000+6,000 for test   |
| Cifar100 |                    Unified                     | 50 normals & 50 abnormals <br> 50x500 for train & 50x100+50x100 for test  |

```
data
├── cifar
    ├── cifar-100-python
    ├── cifar-10-batches-py
```

### Tiny-ImageNet-200
- Download and extract [Tiny ImageNet](https://paperswithcode.com/dataset/tiny-imagenet) into `data/tiny-imagenet-200`.
- run`python data/gen_benchmark/coco.py` to obtain `data/coco/meta_20_x.json` and `val2017_mask_ad_20_x`, where `x=0,1,2,3` represent splits.
- Use `TinyINAD` in `data/ad_dataset.py` as the dataloader.
```
data
├── cifar
    ├── cifar-100-python
    ├── cifar-10-batches-py
```

| Dataset  | Setting |                                 Description                                 | 
|:--------:|:-------:|:---------------------------------------------------------------------------:|
| Cifar100 | Unified | 100 normals & 100 abnormals <br> 100x500 for train & 100x50+100x50 for test |


