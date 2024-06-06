## 📜 ADer Benchmark under the Multi-class Setting on Popular AD Datasets
- <span style="color:red">**Red metrics**</span> are recommended for comprehensive evaluations.<br>
Subscripts `I`, `R`, and `P` represent `image-level`, `region-level`, and `pixel-level`, respectively.
- The following results are derived from the original paper. Since the [benchmark](configs/benchmark/README.md) re-runs all experiments on the same `L40S` platform, slight differences in the results are reasonable.
- All experiments are conducted on `256x256` resolution and `100/300` epochs by default.

## Content
- 100 Epochs Setting
  - [MVTec AD-100epochs](#MVTec-AD-100epochs)
  - [MVTec 3D-100epochs](#MVTec-3D-100epochs)
  - [MVTec LOCO-100epochs](#MVTec-LOCO-100epochs)
  - [VisA-100epochs](#VisA-100epochs)
  - [BTAD-100epochs](#BTAD-100epochs)
  - [MPDD-100epochs](#MPDD-100epochs)
  - [MAD_Real-100epochs](#MAD_Real-100epochs)
  - [MAD_Sim-100epochs](#MAD_Sim-100epochs)
  - [Uni-Medical-100epochs](#Uni-Medical-100epochs)
  - [Real-IAD-100epochs](#Real-IAD-100epochs)
  - [COCO-AD-100epochs](#COCO-AD-100epochs)
- 300 Epochs Setting
  - [MVTec AD-300epochs](#MVTec-AD-300epochs)
  - [MVTec 3D-300epochs](#MVTec-3D-300epochs)
  - [MVTec LOCO-300epochs](#MVTec-LOCO-300epochs)
  - [VisA-300epochs](#VisA-300epochs)
  - [BTAD-300epochs](#BTAD-300epochs)
  - [MPDD-300epochs](#MPDD-300epochs)
  - [MAD_Real-300epochs](#MAD_Real-300epochs)
  - [MAD_Sim-300epochs](#MAD_Sim-300epochs)
  - [Uni-Medical-300epochs](#Uni-Medical-300epochs)


## 100epoch

### MVTec AD-100epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
 | [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 54.5 | 76.3 | 83.6 | 47.6 | 3.2 | 6.7 | 14.3 | 3.5 | 44.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mvtec_100e.log) |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 95.4 | 98.3 | 95.7 | 96.8 | 48.8 | 51.9 | 86.9 | 36.4 | 83.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mvtec_100e.log) |
| [RealNet, CVPR'24](https://github.com/cnulab/RealNet) | 84.8 | 94.1 | 90.9 | 72.6 | 48.2 | 41.4 | 56.8 | 28.8 | 72.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/realnet/mvtec_100e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 57.6 | 78.3 | 84.7 | 54.8 | 11.9 | 14.7 | 25.3 | 8.9 | 50.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mvtec_100e.log) |
| [PatchCore, CVPR'22](https://github.com/amazon-science/patchcore-inspection) | **98.8** | **99.5** | **98.4** | **98.3** | <u>59.9</u> | <u>61.0</u> | <u>94.2</u> | <u>44.9</u> | **88.6** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/patchcore/mvtec_100e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 91.6 | 96.7 | 93.4 | 95.7 | 45.9 | 48.6 | 88.3 | 33.2 | 81.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/mvtec_100e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 70.2 | 85.5 | 85.5 | 80.0 | 22.3 | 22.0 | 47.5 | 12.8 | 61.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mvtec_100e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 93.6 | 97.2 | 95.6 | 95.8 | 48.2 | 53.6 | 91.2 | 37.0 | 83.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mvtec_100e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 88.9 | 95.8 | 93.5 | 89.3 | 27.0 | 32.5 | 63.9 | 21.1 | 70.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/diad/mvtec_100e.log)  |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | <u>98.3</u>| <u>99.3</u> | 97.3 | 97.6 | 55.2 | 58.4 | 92.0 | 42.3 | 87.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mvtec_100e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 98.1 | 99.0 | <u>97.6</u> | <u>98.0</u> | 56.3 | 59.2 | **94.4** | 42.8 | 87.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mvtec_100e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 97.9 | 99.2 | 96.8 | 97.3 | 54.4 | 57.8 | 93.3 | 41.4 | 86.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mvtec_100e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 97.8 | <u>99.3</u> | 97.3 | 97.4 | 55.1 | 57.6 | 93.4 | 41.2 | 87.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mvtec_100e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 92.5 | 97.3 | 95.4 | 95.8 | 42.7 | 48.0 | 89.3 | 32.5 | 82.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mvtec_100e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 97.9 | 98.8 | 96.4 | 97.3 | 54.7 | 58.0 | 93.2 | 41.5 | 86.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mvtec_100e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 96.4 | 98.6 | 96.2 | 92.0 | **71.1** | **68.2** | 83.4 | **52.8** | <u>87.9</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mvtec_100e.log)  |




### MVTec 3D-100epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 55.7 | 81.8 | 88.8 | 46.6 | 0.8 | 1.9 | 21.9 | 1.0 | 46.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mvtec3d_100e.log) |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 75.8 | 92.3 | 90.4 | 94.7 | 17.3 | 23.4 | 81.0 | 13.9 | 70.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mvtec3d_100e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 67.1 | 89.3 | 89.6 | 71.6 | 9.7 | 16.1 | 43.2 | 9.3 | 58.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mvtec3d_100e.log) |
| [PatchCore, CVPR'22](https://github.com/amazon-science/patchcore-inspection) | 84.1 | 95.1 | 92.5 | <u>98.6</u> | 33.7 | 38.5 | <u>94.4</u> | 24.5 | 78.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/patchcore/mvtec3d_100e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 73.1 | 91.0 | 90.2 | 96.8 | 21.6 | 26.6 | 89.0 | 15.8 | 71.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/mvtec3d_100e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 58.7 | 85.6 | 88.6 | 90.8 | 7.2 | 12.0 | 74.9 | 6.4 | 61.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mvtec3d_100e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 75.8 | 91.2 | 91.5 | 98.4 | 29.6 | 35.5 | 93.9 | 22.2 | 75.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mvtec3d_100e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 75.4 | 91.0 | 90.9 | 91.7 | 5.3 | 10.2 | 74.9 | 5.4 | 62.8 |  [log](https://github.com/zhangzjn/data/tree/main/ader_log/diad/mvtec_100e.log) |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 79.6 | 93.4 | 91.6 | 98.0 | 27.2 | 33.3 | 91.6 | 20.4 | 75.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mvtec3d_100e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **86.9** | **96.2** | **93.3** | **98.7** | 37.4 | <u>42.4</u> | **95.2** | <u>27.4</u> | **80.3** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mvtec3d_100e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | <u>84.7</u> | <u>95.6</u> | 92.6 | <u>98.6</u> | <u>38.3</u> | 41.9 | 94.2 | 26.9 | <u>79.6</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mvtec3d_100e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 84.1 | 95.1 | 92.2 | <u>98.6</u> | 36.9 | 40.8 | 94.2 | 25.9 | 79.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mvtec3d_100e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 75.3 | 92.1 | 90.3 | 96.6 | 21.2 | 27.9 | 88.9 | 16.7 | 72.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mvtec3d_100e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 81.8 | 93.9 | <u>92.8</u> | 98.5 | 34.4 | 39.6 | 94.1 | 25.2 | 78.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mvtec3d_100e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 79.1 | 93.8 | 90.7 | 81.1 | **41.3** | **42.7** | 64.0 | **28.4** | 72.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mvtec3d_100e.log)  |




### MVTec LOCO-100epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 52.5 | 65.4 | 77.6 | 46.7 | 4.5 | 10.4 | 12.4 | 5.6 | 41.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mvtec_loco_100e.log) |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | <u>81.6</u> | 88.5 | **82.9** | **76.5** | 29.0 | 32.7 | 63.8 | 21.2 | 67.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mvtec_loco_100e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 66.5 | 78.6 | 77.9 | 58.5 | 14.2 | 16.3 | 31.1 | 9.1 | 52.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mvtec_loco_100e.log)  |
| [PatchCore, CVPR'22](https://github.com/amazon-science/patchcore-inspection) | 80.5 | <u>89.0</u> | 81.5 | 75.1 | 29.9 | 31.8 | 69.9 | 20.4 | <u>67.7</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/patchcore/mvtec_loco_100e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 75.1 | 84.2 | 79.8 | 70.7 | 25.0 | 27.9 | 69.5 | 17.3 | 64.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/mvtec_loco_100e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 53.4 | 69.3 | 77.5 | 57.7 | 15.0 | 14.5 | 25.7 | 8.0 | 47.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mvtec_loco_100e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 74.6 | 85.3 | 78.9 | 70.7 | 21.0 | 26.1 | 67.7 | 15.8 | 63.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mvtec_loco_100e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 71.3 | 79.5 | 80.9 | 71.8 | 19.0 | 24.8 | 48.4 | 14.9 | 56.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/diad/mvtec_100e.log)  |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 76.4 | 85.2 | 79.6 | 73.4 | 28.7 | 31.2 | 63.1 | 19.8 | 64.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mvtec_loco_100e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **82.0** | **89.9** | <u>82.0</u> | <u>76.4</u> | <u>31.1</u> | **34.8** | **73.0** | **23.1** | **69.2** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mvtec_loco_100e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 79.4 | 87.8 | 80.9 | 75.8 | 28.5 | 32.2 | 69.4 | 20.6 | 67.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mvtec_loco_100e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 75.8 | 85.5 | 81.0 | **76.5** | 28.8 | 32.5 | <u>70.2</u> | 20.6 | 66.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mvtec_loco_100e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 73.2 | 82.1 | 79.6 | 73.1 | 30.7 | <u>33.4</u> | 64.7 | <u>21.6</u> | 64.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mvtec_loco_100e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 76.4 | 86.8 | 79.2 | 71.5 | 24.3 | 28.3 | 68.8 | 17.6 | 64.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mvtec_loco_100e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 77.9 | 86.7 | 81.3 | 64.1 | **35.2** | 30.5 | 53.4 | 20.3 | 63.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mvtec_loco_100e.log)  |




### VisA-100epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 55.1 | 62.4 | 72.9 | 37.5 | 0.6 | 1.7 | 10.0 | 0.9 | 38.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/visa_100e.log) |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 86.4 | 89.1 | 82.8 | 96.6 | 34.0 | 37.8 | 79.2 | 25.7 | 74.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/visa_100e.log) |
| [RealNet, CVPR'24](https://github.com/cnulab/RealNet) | 71.4 | 79.5 | 74.7 | 61.0 | 25.7 | 22.6 | 27.4 | 13.5 | 54.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/realnet/visa_100e.log) |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 66.3 | 74.3 | 74.2 | 81.3 | 22.1 | 26.2 | 50.8 | 17.0 | 58.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/visa_100e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 86.5 | 88.8 | 84.9 | 97.7 | 33.9 | 37.2 | 86.8 | 24.9 | 75.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/visa_100e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 58.2 | 66.3 | 74.4 | 77.0 | 7.2 | 9.6 | 42.8 | 5.6 | 50.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/visa_100e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 90.6 | 90.9 | 89.3 | 98.0 | 35.4 | 42.5 | 91.9 | 27.9 | 78.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/visa_100e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 84.8 | 88.5 | 86.9 | 82.5 | 17.9 | 23.2 | 44.5 | 14.9 | 61.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/diad/mvtec_100e.log)  |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 90.4 | 91.1 | 86.0 | 98.2 | 36.4 | 41.0 | 85.7 | 27.5 | 77.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/visa_100e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **95.4** | **95.7** | **91.6** | **98.9** | <u>43.3</u> | <u>46.8</u> | **93.1** | <u>32.5</u> | **82.4** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/visa_100e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | <u>94.9</u> | <u>95.2</u> | <u>90.7</u> | <u>98.6</u> | 40.2 | 44.0 | **93.1** | 29.8 | <u>81.3</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/visa_100e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 94.5 | 94.9 | 90.2 | 98.4 | 39.3 | 43.7 | <u>92.1</u> | 29.5 | 80.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/visa_100e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 89.0 | 91.0 | 85.8 | 98.3 | 34.5 | 39.6 | 86.5 | 26.4 | 76.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/visa_100e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 93.9 | 94.7 | 90.2 | 98.4 | 42.3 | 46.3 | 91.9 | 31.2 | <u>81.3</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/visa_100e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 89.9 | 91.4 | 86.7 | 86.7 | **46.6** | **47.2** | 61.1 | **32.7** | 74.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/visa_100e.log)  |


### BTAD-100epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 71.3 | 78.5 | 78.0 | 49.0 | 3.7 | 6.5 | 16.2 | 3.4 | 47.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/btad_100e.log) |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 93.2 | 97.3 | 93.3 | 96.3 | 41.5 | 44.3 | 69.8 | 28.6 | 78.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/btad_100e.log) |
| [RealNet, CVPR'24](https://github.com/cnulab/RealNet) | 89.7 | 95.3 | 92.8 | 84.0 | 48.1 | 52.7 | 53.4 | 36.6 | 76.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/realnet/btad_100e.log) |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 92.7 | 97.5 | 93.5 | 96.3 | 47.4 | 50.2 | 69.5 | 33.6 | 80.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/btad_100e.log)  |
| [PatchCore, CVPR'22](https://github.com/amazon-science/patchcore-inspection) | 94.4 | <u>98.2</u> | **94.6** | 97.5 | 55.0 | 54.9 | 76.0 | 38.0 | 83.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/patchcore/btad_100e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 91.2 | 94.8 | 88.3 | 96.8 | 45.6 | 50.1 | 72.7 | 33.8 | 78.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/btad_100e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 87.0 | 83.1 | 81.0 | 90.9 | 29.6 | 26.9 | 64.1 | 18.3 | 68.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/btad_100e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 94.4 | 96.6 | 94.0 | **98.1** | <u>59.6</u> | 59.2 | <u>80.7</u> | 42.1 | 84.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/btad_100e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 90.1 | 88.4 | 92.6 | 91.7 | 19.6 | 26.7 | 70.4 | 15.7 | 68.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/diad/mvtec_100e.log)  |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 93.6 | 96.8 | 93.3 | 97.6 | 59.2 | 56.7 | 73.2 | 40.1 | 83.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/btad_100e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **96.1** | 97.8 | <u>94.3</u> | **98.1** | **62.6** | **61.4** | **80.9** | **44.3** | **85.9** |  [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/btad_100e.log) |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 93.1 | 97.4 | **94.6** | 97.9 | 59.2 | 59.7 | 78.6 | 42.6 | 84.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/btad_100e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 93.1 | 96.1 | 92.5 | 97.7 | 52.5 | 56.0 | 78.2 | 39.0 | 82.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/btad_100e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | <u>94.8</u> | **98.3** | 94.2 | 97.2 | 50.3 | 53.8 | 78.8 | 36.9 | 82.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/btad_100e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 94.6 | 97.8 | 94.1 | <u>98.0</u> | <u>59.6</u> | <u>59.8</u> | 79.0 | <u>42.8</u> | <u>84.8</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/btad_100e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 92.8 | 95.9 | 92.3 | 92.2 | 34.8 | 44.3 | 70.0 | 29.0 | 77.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/btad_100e.log)  |




### MPDD-100epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
 | [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 35.6 | 53.3 | 72.5 | 42.1 | 2.2 | 4.6 | 19.0 | 2.5 | 35.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mpdd_100e.log) |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 88.4 | 92.0 | 87.9 | 96.5 | 32.0 | 34.6 | 89.0 | 24.5 | 76.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mpdd_100e.log) |
| [RealNet, CVPR'24](https://github.com/cnulab/RealNet) | 85.1 | 90.2 | 88.3 | 83.3 | 36.1 | 39.6 | 68.1 | 28.2 | 72.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/realnet/mpdd_100e.log) |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 81.6 | 87.7 | 85.7 | 84.9 | 19.6 | 22.9 | 53.5 | 16.6 | 65.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mpdd_100e.log)  |
| [PatchCore, CVPR'22](https://github.com/amazon-science/patchcore-inspection) | **94.6** | **96.9** | **93.4** | **98.9** | **46.1** | **47.6** | **95.7** | **35.0** | **83.5** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/patchcore/mpdd_100e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 75.7 | 80.1 | 81.7 | 96.8 | 26.3 | 28.0 | 89.5 | 20.1 | 69.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/mpdd_100e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 73.6 | 77.0 | 79.4 | 94.1 | 21.1 | 17.8 | 77.2 | 10.4 | 64.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mpdd_100e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 91.3 | 93.6 | 91.8 | 98.3 | 40.4 | 41.8 | <u>95.5</u> | 31.4 | 80.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mpdd_100e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 68.3 | 77.9 | 80.1 | 90.4 | 10.9 | 13.1 | 66.1 | 8.2 | 58.1 | [log]()  |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 87.8 | 90.5 | 88.0 | 97.7 | 35.2 | 37.4 | 92.8 | 27.7 | 77.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mpdd_100e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | <u>93.7</u> | <u>93.9</u> | <u>93.0</u> | 98.2 | 42.4 | <u>45.7</u> | 94.8 | <u>34.0</u> | <u>81.9</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mpdd_100e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 90.9 | 92.9 | 89.5 | 98.0 | 39.7 | 42.6 | 94.0 | 30.9 | 79.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mpdd_100e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 88.7 | 93.2 | 90.8 | 97.5 | 33.6 | 38.1 | 92.3 | 26.8 | 78.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mpdd_100e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 70.5 | 76.0 | 78.0 | 93.9 | 13.7 | 19.9 | 79.7 | 12.5 | 63.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mpdd_100e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 90.2 | 93.3 | 90.5 | <u>98.5</u> | <u>43.0</u> | 44.1 | <u>95.5</u> | 33.6 | 80.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mpdd_100e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 91.3 | 90.8 | 90.2 | 82.0 | 32.6 | 34.6 | 63.3 | 25.6 | 71.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mpdd_100e.log)  |




### MAD_Real-100epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 59.1 | **88.1** | 90.4 | 49.4 | 0.6 | 1.7 | 22.3 | 0.8 | 48.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mad_real_100e.log) |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 47.8 | 81.9 | 90.7 | 83.1 | 5.4 | 11.6 | 59.8 | 6.3 | 56.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mad_real_100e.log) |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 49.3 | 81.7 | 91.4 | 83.3 | 10.5 | 15.2 | 64.2 | 8.7 | 58.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mad_real_100e.log)  |
| [PatchCore, CVPR'22](https://github.com/amazon-science/patchcore-inspection) | 59.7 | 84.8 | <u>93.5</u> | **95.4** | <u>20.2</u> | **27.3** | **85.5** | <u>16.6</u> | **68.2** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/patchcore/mad_real_100e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 59.8 | 86.0 | 91.2 | <u>90.9</u> | 10.7 | 15.4 | <u>74.4</u> | 8.8 | 63.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/mad_real_100e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 52.1 | 83.6 | 90.3 | 88.4 | 5.0 | 9.4 | 68.2 | 5.1 | 59.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mad_real_100e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 55.5 | 85.8 | 90.6 | 87.0 | 8.2 | 12.8 | 67.8 | 7.2 | 60.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mad_real_100e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 49.9 | 85.1 | 90.8 | 87.7 | 2.9 | 6.8 | 67.4 | 3.6 | 55.8 | [log]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 47.0 | 81.3 | 90.2 | 87.3 | 4.5 | 9.3 | 66.6 | 5.0 | 57.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mad_ral_100e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **66.5** | <u>87.7</u> | **93.8** | 88.3 | **22.1** | <u>26.2</u> | 73.2 | **16.8** | <u>67.6</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mad_real_100e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | <u>60.4</u> | 84.3 | 92.2 | 88.2 | 9.0 | 14.3 | 72.9 | 8.1 | 62.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mad_real_100e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 56.2 | 84.3 | 90.3 | 87.8 | 7.5 | 12.8 | 68.5 | 7.2 | 60.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mad_real_100e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 47.6 | 81.3 | 90.4 | 88.2 | 5.8 | 10.6 | 68.6 | 5.8 | 58.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mad_real_100e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 54.7 | 83.9 | 92.3 | 88.5 | 9.4 | 14.9 | 70.4 | 8.5 | 61.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mad_real_100e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 45.4 | 80.3 | 90.7 | 72.7 | 5.0 | 8.2 | 48.3 | 4.5 | 52.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mad_real_100e.log)  |





### MAD_Sim-100epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 49.5 | 88.3 | 93.8 | 48.4 | 0.8 | 1.5 | 24.5 | 0.7 | 48.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mad_sim_100e.log) |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 58.9 | 91.7 | 93.8 | 85.5 | 4.2 | 8.0 | 63.9 | 4.2 | 60.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mad_sim_100e.log) |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 56.4 | 91.1 | 93.8 | 76.6 | 4.2 | 8.7 | 54.7 | 4.6 | 58.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mad_sim_100e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 55.8 | 90.6 | 93.8 | 86.9 | 3.4 | 5.2 | 63.9 | 2.7 | 60.0 |  [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/mad_sim_100e.log) |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 50.8 | 88.9 | 93.8 | 82.0 | 2.3 | 4.8 | 58.8 | 2.5 | 57.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mad_sim_100e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 58.6 | 91.7 | 93.8 | 88.3 | 4.7 | 8.5 | 74.2 | 4.5 | 62.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mad_sim_100e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | <u>65.7</u> | <u>93.3</u> | **94.1** | 87.2 | 3.9 | 7.9 | 60.1 | 4.2 | 58.9 | [log]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 58.5 | 91.5 | 93.8 | 89.0 | 5.0 | 9.5 | 73.5 | 5.0 | 62.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mad_sim_100e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **70.4** | **94.6** | <u>93.9</u> | **90.5** | **8.8** | **15.7** | **79.3** | **8.6** | **67.4** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mad_sim_100e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 64.7 | <u>93.3</u> | 93.8 | <u>89.4</u> | <u>6.4</u> | <u>11.2</u> | <u>76.1</u> | <u>6.0</u> | <u>64.9</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mad_sim_100e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 63.0 | 92.9 | 93.8 | 88.3 | 5.2 | 9.4 | 73.3 | 5.0 | 63.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mad_sim_100e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 54.2 | 90.0 | 93.8 | 88.1 | 3.5 | 6.8 | 71.8 | 3.5 | 60.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mad_sim_100e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 59.3 | 91.7 | 93.8 | 88.3 | 4.6 | 8.5 | 73.8 | 4.4 | 62.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mad_sim_100e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 57.9 | 91.8 | 93.8 | 54.5 | 6.2 | 7.7 | 35.8 | 4.1 | 53.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mad_sim_100e.log)  |




### Uni-Medical-100epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 51.6 | 60.6 | 70.6 | 33.8 | 2.0 | 5.7 | 10.5 | 3.0 | 37.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/medical_100e.log) |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 77.5 | 77.7 | 76.7 | 94.3 | 34.4 | 36.0 | 77.0 | 23.3 | 68.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/medical_100e.log) |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 71.0 | 73.3 | 73.1 | 78.3 | 19.5 | 24.9 | 44.7 | 14.7 | 57.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/medical_100e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 79.2 | 79.6 | 77.2 | 95.4 | 40.4 | 26.9 | 81.8 | 17.7 | 69.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/medical_100e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 53.9 | 57.3 | 70.5 | 78.8 | 8.4 | 17.1 | 34.2 | 9.4 | 47.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/medical_100e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 76.1 | 75.7 | 78.2 | 96.5 | 38.8 | 39.8 | 86.8 | 26.9 | 71.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/medical_100e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 78.8 | 77.2 | 77.7 | 95.8 | 34.2 | 35.5 | 84.3 | 23.2 | 69.1 | [log]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 81.8 | <u>80.7</u> | 80.0 | <u>97.1</u> | **48.3** | **48.2** | 86.7 | **33.7** | <u>75.5</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/medical_100e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | <u>82.4 | 80.5 | <u>80.5</u> | **97.3** | <u>46.2</u> | 46.1 | **89.3** | 32.6 | 75.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/medical_100e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 77.9 | 77.3 | 79.1 | 96.2 | 39.0 | 39.3 | 85.8 | 26.5 | 71.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/medical_100e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | **83.9** | **80.8** | **81.9** | 96.8 | 45.8 | <u>47.5</u> | <u>88.2</u> | <u>33.5</u> | **75.9** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/medical_100e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 79.0 | 76.1 | 77.1 | 96.6 | 39.3 | 41.1 | 86.0 | 27.6 | 71.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/medical_100e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 77.2 | 76.5 | 79.2 | 96.7 | 41.5 | 42.2 | 87.8 | 29.4 | 72.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/medical_100e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 78.5 | 77.0 | 78.2 | 65.7 | 41.7 | 34.0 | 35.3 | 21.2 | 61.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/medical_100e.log)  |




### Real-IAD-100epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 50.9 | 45.9 | 61.3 | 44.0 | 0.2 | 0.4 | 13.6 | 0.2 | 33.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/realiad_100e.log) |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 54.9 | 50.6 | 61.5 | 76.1 | 1.9 | 4.9 | 42.4 | 2.5 | 43.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/realiad_100e.log) |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 55.7 | 50.5 | 61.9 | 81.3 | 1.6 | 3.8 | 48.8 | 2.0 | 45.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/realiad_100e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 77.0 | 75.8 | 69.9 | 94.8 | 17.6 | 21.7 | 80.4 | 12.4 | 63.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/realiad_100e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 54.4 | 48.0 | 62.0 | 71.1 | 1.2 | 1.1 | 34.9 | 0.5 | 40.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/realiad_100e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 82.7 | 79.3 | 74.1 | 97.2 | 25.2 | 32.8 | 90.0 | 20.0 | 70.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/realiad_100e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 75.6 | 66.4 | 69.9 | 88.0 | 2.9 | 7.1 | 58.1 | 3.7 | 52.6 | [log]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 82.7 | 80.2 | 73.7 | 97.2 | 24.3 | 32.3 | 84.8 | 19.6 | 69.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/realiad_100e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **89.4** | **87.0** | **80.2** | <u>98.4</u> | <u>32.6</u> | <u>38.9</u> | **92.7** | <u>24.6</u> | **75.6** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/realiad_100e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | <u>87.2</u> | 85.2 | <u>77.8</u> | 98.0 | 31.7 | 37.9 | <u>92.0</u> | 23.8 | <u>74.2</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/realiad_100e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 87.0 | <u>85.3</u> | 77.6 | **98.6** | 32.4 | 38.1 | 91.2 | 23.9 | <u>74.2</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/realiad_100e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 83.1 | 81.2 | 74.5 | 97.4 | 23.3 | 30.9 | 87.1 | 18.6 | 69.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/realiad_100e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 83.6 | 80.6 | 74.8 | 97.7 | 25.9 | 33.6 | 90.7 | 20.5 | 70.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/realiad_100e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 79.3 | 76.7 | 70.7 | 80.3 | **36.9** | **40.3** | 56.1 | **26.2** | 64.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/realiad_100e.log)  |





### COCO-AD-100epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> | <span style="color:blue">Download</span> | <span style="color:blue">Download</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 53.5 | 48.5 | 62.5 | 49.9 | 7.6 | 14.4 | 15.3 | 8.0 | 38.3 | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/draem/coco_0_100e.log) |[log1](https://github.com/zhangzjn/data/tree/main/ader_log/draem/coco_1_100e.log) |[log2](https://github.com/zhangzjn/data/tree/main/ader_log/draem/coco_2_100e.log) |[log3](https://github.com/zhangzjn/data/tree/main/ader_log/draem/coco_3_100e.log) |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 55.6 | 49.4 | 62.3 | 60.2 | 14.5 | 19.2 | 26.1 | 11.5 | 42.9 | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/coco_3_100e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 56.7 | 51.2 | 62.6 | 56.2 | 10.3 | 16.0 | 17.9 | 8.9 | 41.0 | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/coco_3_100e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | **67.7** | <u>57.9</u> | **64.5** | <u>76.0</u> | 20.3 | <u>26.4</u> | **47.7** | <u>16.0</u> | <u>53.0</u> | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/coco_3_100e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 51.6 | 47.4 | 61.6 | 50.0 | 13.4 | 14.4 | 15.0 | 8.0 | 38.4 | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/coco_3_100e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 57.6 | 49.9 | 62.0 | 66.5 | 13.9 | 20.0 | 39.8 | 11.5 | 45.8 | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/rd/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/rd/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/rd/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/rd/coco_3_100e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 59.0 | 53.0 | 63.2 | 68.1 | <u>20.5</u> | 14.2 | 30.8 | 11.6 | 44.1 | [log0]()   | [log1]()   | [log2]()   | [log3]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | <u>66.9</u> | **59.3** | <u>63.7</u> | **76.2** | **27.6** | **32.2** | 39.1 | **20.1** | **53.5** | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/coco_3_100e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 64.2 | 56.0 | 63.4 | 71.2 | 18.5 | 24.1 | <u>45.8</u> | 14.3 | 50.5 | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/invad/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/invad/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/invad/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/invad/coco_3_100e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 63.3 | 56.0 | 62.9 | 69.9 | 18.2 | 23.1 | 39.7 | 13.7 | 49.2 | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/coco_3_100e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 62.8 | 55.2 | 62.8 | 68.9 | 16.7 | 22.0 | 41.6 | 12.9 | 48.8 | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/coco_3_100e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 55.2 | 49.3 | 61.7 | 64.6 | 12.8 | 19.0 | 34.3 | 10.9 | 44.0 | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/coco_3_100e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 57.5 | 49.3 | 62.0 | 68.2 | 15.5 | 20.1 | 42.2 | 11.8 | 46.4 | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/coco_3_100e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 54.4 | 49.4 | 61.8 | 54.5 | 10.6 | 15.4 | 24.4 | 8.5 | 40.7 | [log0](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/coco_0_100e.log)  | [log1](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/coco_1_100e.log)  | [log2](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/coco_2_100e.log)  | [log3](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/coco_3_100e.log)  |



## 300epochs


### MVTec AD-300epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 55.2 | 77.0 | 83.9 | 48.7 | 3.1 | 6.3 | 15.8 | 3.3 | 45.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mvtec_300e.log)  |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 79.2 | 90.8 | 87.6 | 82.4 | 24.0 | 29.0 | 62.0 | 17.8 | 67.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mvtec_300e.log)  |
| [RealNet, CVPR'24](https://github.com/cnulab/RealNet) | 82.9 | 93.3 | 90.9 | 69.8 | 50.0 | 40.4 | 51.2 | 28.5 | 70.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/realnet/mvtec_300e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 55.8 | 78.8 | 84.5 | 43.9 | 4.8 | 8.9 | 19.3 | 4.7 | 46.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mvtec_300e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 92.7 | 97.2 | 94.0 | 95.8 | 46.8 | 49.6 | 89.0 | 34.0 | 82.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/mvtec_300e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 66.2 | 84.3 | 85.1 | 74.2 | 17.2 | 19.6 | 40.0 | 11.4 | 58.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mvtec_300e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 90.5 | 95.0 | 95.1 | 95.9 | 47.1 | 52.1 | 91.2 | 35.8 | 82.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mvtec_300e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 92.0 | 96.8 | 94.4 | 89.3 | 27.3 | 32.7 | 64.4 | 21.3 | 71.0 |  [log]()  |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 98.4 | 99.4 | 97.5 | 97.5 | 55.2 | 58.1 | 91.7 | 42.0 | 87.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mvtec_300e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **98.9** | **99.6** | **98.2** | **98.1** | <u>57.1</u> | <u>59.6</u> | **94.4** | <u>43.1</u> | <u>88.1</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mvtec_300e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 98.1 | 99.1 | 96.8 | 97.3 | 55.0 | 58.1 | 93.1 | 41.7 | 86.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mvtec_300e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | <u>98.5</u> | <u>99.5</u> | <u>97.7</u> | <u>97.6</u> | 56.1 | 58.7 | <u>93.6</u> | 42.3 | 87.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mvtec_300e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 96.8 | 98.9 | 97.0 | 96.8 | 45.0 | 50.2 | 91.0 | 34.2 | 84.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mvtec_300e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 95.8 | 98.0 | 96.6 | 97.3 | 53.0 | 57.0 | 92.9 | 40.5 | 85.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mvtec_300e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 96.3 | 98.8 | 96.1 | 92.6 | **75.8** | **71.3** | 82.6 | **56.6** | **88.8** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mvtec_300e.log)  |




### MVTec 3D-300epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 49.2 | 79.4 | 89.0 | 50.7 | 0.7 | 2.2 | 23.4 | 1.1 | 45.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mvtec3d_300e.log)  |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 74.9 | 91.8 | 90.2 | 94.0 | 14.4 | 21.4 | 79.2 | 12.4 | 68.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mvtec3d_300e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 60.1 | 86.3 | 89.5 | 52.0 | 7.0 | 11.9 | 28.7 | 6.9 | 51.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mvtec3d_300e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 74.8 | 91.4 | 90.5 | 96.9 | 21.6 | 26.8 | 89.2 | 15.9 | 72.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/mvtec3d_300e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 52.5 | 80.8 | 88.6 | 87.2 | 5.9 | 10.4 | 68.0 | 5.6 | 58.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mvtec3d_300e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 79.3 | 93.0 | 92.3 | 98.3 | 28.6 | 35.4 | 93.5 | 22.1 | 76.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mvtec3d_300e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 78.8 | 91.3 | 91.7 | 91.8 | 5.2 | 10.5 | 75.1 | 5.6 | 63.5 | [log]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 78.7 | 93.3 | 91.5 | 98.0 | 27.0 | 33.0 | 91.3 | 20.2 | 75.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mvtec3d_300e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **86.3** | **95.9** | **93.4** | **98.7** | 36.2 | <u>41.1</u> | **94.9** | <u>26.4</u> | **79.8** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mvtec3d_300e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 85.1 | <u>95.6</u> | 92.3 | 98.4 | 34.4 | 39.0 | 93.4 | 24.6 | 78.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mvtec3d_300e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | <u>85.8</u> | 95.5 | <u>92.6</u> | <u>98.6</u> | <u>37.1</u> | 40.8 | <u>94.1</u> | 26.0 | <u>79.5</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mvtec3d_300e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 77.0 | 92.6 | 91.0 | 96.8 | 23.7 | 30.6 | 89.4 | 18.6 | 73.5 |  [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mvtec3d_300e.log) |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 80.1 | 93.5 | 92.2 | 98.4 | 31.5 | 37.2 | 93.6 | 23.4 | 76.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mvtec3d_300e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 82.5 | 95.1 | 90.9 | 82.1 | **43.8** | **45.2** | 65.1 | **30.2** | 74.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mvtec3d_300e.log)  |




### MVTec LOCO-300epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 51.4 | 66.9 | 77.5 | 48.0 | 5.2 | 10.5 | 13.7 | 5.6 | 42.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mvtec_loco_300e.log)  |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | <u>81.8</u> | <u>89.1</u> | **82.5** | 70.9 | 28.3 | 32.1 | 61.2 | 20.9 | 66.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mvtec_loco_300e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 60.6 | 73.7 | 77.6 | 64.7 | 16.5 | 18.8 | 36.4 | 10.8 | 52.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mvtec_loco_300e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 76.0 | 85.0 | 79.9 | 70.9 | 25.6 | 28.4 | 70.0 | 17.7 | 64.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/patchcore/mvtec_loco_300e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 58.1 | 70.4 | 77.4 | 55.5 | 16.8 | 14.4 | 20.4 | 8.1 | 47.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mvtec_loco_300e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 73.7 | 84.6 | 79.4 | 70.7 | 21.0 | 26.2 | 67.5 | 15.9 | 62.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mvtec_loco_300e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 77.2 | 84.6 | 81.5 | 72.1 | 18.6 | 25.3 | 54.4 | 15.2 | 59.1 | [log]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 76.2 | 85.0 | 80.1 | 73.2 | 27.9 | 30.6 | 62.0 | 19.3 | 64.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mvtec_loco_300e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **82.3** | **89.9** | <u>82.4</u> | **77.7** | 30.9 | <u>34.1</u> | **72.8** | <u>22.4</u> | **69.4** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mvtec_loco_300e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 78.0 | 86.8 | 81.6 | 77.0 | 28.0 | 32.0 | 69.4 | 20.4 | 66.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mvtec_loco_300e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 78.2 | 87.1 | 81.8 | <u>77.5</u> | 28.0 | 32.9 | 68.6 | 20.8 | 67.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mvtec_loco_300e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 78.7 | 87.1 | 80.9 | 74.6 | **32.6** | **35.8** | <u>70.7</u> | **23.4** | <u>67.8</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mvtec_loco_300e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 75.7 | 85.9 | 79.3 | 72.1 | 24.2 | 28.5 | 67.9 | 17.8 | 64.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mvtec_loco_300e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 81.2 | 88.6 | 81.8 | 63.7 | <u>31.8</u> | 31.9 | 59.2 | 21.1 | 65.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mvtec_loco_300e.log)  |




### VisA-300epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 56.2 | 64.6 | 74.9 | 45.0 | 0.7 | 1.8 | 16.0 | 0.9 | 40.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/visa_300e.log)  |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 80.7 | 83.8 | 79.3 | 94.4 | 29.2 | 33.1 | 74.2 | 22.1 | 69.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/visa_300e.log)  |
| [RealNet, CVPR'24](https://github.com/cnulab/RealNet) | 79.2 | 84.8 | 78.3 | 65.4 | 29.2 | 27.9 | 33.9 | 17.4 | 59.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/realnet/visa_300e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 67.1 | 73.8 | 75.3 | 83.0 | 13.7 | 18.7 | 48.7 | 11.3 | 56.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/visa_300e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 87.2 | 89.5 | 85.1 | 97.8 | 34.2 | 37.2 | 87.3 | 24.9 | 75.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/visa_300e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 69.0 | 72.9 | 75.8 | 79.1 | 7.9 | 8.7 | 52.6 | 4.7 | 54.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/visa_300e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 93.9 | 94.8 | 90.4 | 98.1 | 38.4 | 43.7 | 91.9 | 29.0 | 80.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/visa_300e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 90.5 | 91.4 | 90.4 | 83.4 | 19.2 | 25.0 | 44.3 | 16.2 | 63.5 | [log]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 90.3 | 91.2 | 86.4 | 98.2 | 36.4 | 40.9 | 85.8 | 27.5 | 77.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/visa_300e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **95.6** | **96.0** | **92.3** | **99.0** | **43.7** | **46.9** | <u>93.0</u> | **32.6** | **82.6** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/visa_100e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | <u>95.3</u> | <u>95.8</u> | <u>91.0</u> | <u>98.7</u> | 41.2 | <u>44.9</u> | **93.2** | <u>30.6</u> | <u>81.8</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/visa_300e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 93.6 | 93.9 | 89.8 | 98.2 | 34.0 | 39.3 | 90.5 | 25.9 | 79.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/visa_300e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 91.4 | 93.3 | 87.5 | 98.5 | 35.3 | 40.2 | 89.0 | 26.5 | 78.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/visa_300e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 93.1 | 94.1 | 90.0 | 98.4 | 40.4 | 44.8 | 91.4 | 29.9 | 80.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/visa_300e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 89.0 | 90.3 | 85.9 | 84.8 | <u>43.3</u> | 44.4 | 57.5 | 30.1 | 73.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/visa_300e.log)  |


### BTAD-300epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 69.3 | 76.8 | 79.5 | 47.7 | 3.2 | 6.1 | 16.1 | 3.2 | 46.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/btad_300e.log)  |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 94.0 | <u>97.9</u> | 93.9 | 96.2 | 41.0 | 43.7 | 69.6 | 28.1 | 78.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/btad_300e.log)  |
| [RealNet, CVPR'24](https://github.com/cnulab/RealNet) | 93.1 | 96.3 | 92.5 | 87.2 | 48.0 | 55.5 | 57.9 | 38.7 | 78.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/realnet/btad_300e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 92.7 | 97.8 | <u>94.6</u> | 96.2 | 46.7 | 49.7 | 69.0 | 33.2 | 80.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/btad_300e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 91.6 | 95.6 | 89.7 | 96.9 | 46.0 | 49.0 | 72.7 | 33.0 | 79.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/btad_300e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 85.3 | 81.7 | 79.0 | 76.1 | 27.7 | 23.4 | 43.1 | 15.0 | 62.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/btad_300e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 94.1 | 96.8 | 93.8 | **98.0** | 57.1 | 58.0 | **79.9** | 41.0 | 84.1 |  [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/btad_300e.log) |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 90.2 | 88.3 | 92.6 | 91.9 | 20.5 | 27.0 | 70.3 | 16.0 | 68.7 | [log]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 94.0 | 97.0 | 93.7 | 97.6 | <u>58.3</u> | 56.5 | 72.8 | 39.9 | 83.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/btad_300e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **95.3** | 97.3 | 93.7 | <u>97.9</u> | **58.7** | **58.8** | 78.8 | **41.7** | **84.5** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/btad_300e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 94.2 | <u>97.9</u> | 94.4 | 97.7 | 55.7 | 57.0 | 77.4 | 40.0 | 83.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/btad_300e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 92.9 | 96.2 | 93.0 | 97.6 | 51.2 | 55.1 | 77.3 | 38.2 | 82.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/btad_300e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | <u>94.5</u> | **98.4** | **94.9** | 97.4 | 52.4 | 55.5 | <u>78.9</u> | 38.4 | 83.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/btad_300e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | <u>94.5</u> | 97.6 | 94.3 | **98.0** | 57.7 | <u>58.2</u> | <u>78.9</u> | <u>41.2</u> | <u>84.3</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/btad_300e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 93.5 | 96.7 | 93.8 | 94.8 | 39.1 | 38.5 | 72.9 | 24.6 | 78.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/btad_300e.log)  |





### MPDD-300epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 60.2 | 67.5 | 75.1 | 50.4 | 5.4 | 4.7 | 21.8 | 2.6 | 44.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mpdd_300e.log)  |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 90.6 | <u>94.1</u> | 89.7 | 97.1 | 33.6 | 35.7 | 90.0 | 25.6 | 77.8 |  [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mpdd_300e.log) |
| [RealNet, CVPR'24](https://github.com/cnulab/RealNet) | 86.0 | 90.0 | 87.3 | 74.7 | 39.2 | 39.7 | 52.3 | 28.0 | 69.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/realnet/mpdd_300e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 92.0 | 93.5 | 90.2 | 95.7 | 29.0 | 33.0 | 83.2 | 23.8 | 76.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mpdd_300e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 80.4 | 85.1 | 84.7 | 97.3 | 28.4 | 30.1 | 90.9 | 21.5 | 72.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/mpdd_300e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 72.5 | 76.2 | 77.6 | 89.0 | 12.4 | 15.8 | 68.5 | 9.1 | 60.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mpdd_300e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 90.3 | 92.8 | 90.5 | 98.3 | 39.6 | 40.6 | <u>95.2</u> | 30.2 | 79.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mpdd_300e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | 85.8 | 89.2 | 86.5 | 91.4 | 15.3 | 19.2 | 66.1 | 12.0 | 64.8 |  [log]()  |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 87.4 | 90.8 | 87.0 | 97.8 | 34.6 | 37.8 | 92.9 | 28.0 | 77.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mpdd_300e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **96.0** | **97.1** | **94.5** | **98.5** | **44.1** | **46.4** | **95.3** | **34.4** | **83.5** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mpdd_300e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | <u>92.8</u> | 93.2 | 91.4 | 98.3 | 41.4 | <u>43.9</u> | 94.6 | 32.1</u> | <u>81.0</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mpdd_300e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 89.2 | 93.1 | 90.3 | 97.7 | 33.5 | 38.6 | 92.8 | 27.2 | 78.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mpdd_300e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 80.1 | 83.2 | 85.1 | 95.4 | 19.0 | 25.6 | 83.8 | 16.8 | 69.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mpdd_300e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 90.6 | 93.3 | 90.0 | <u>98.4</u> | <u>42.8</u> | 43.0 | **95.3** | <u>32.7</u> | 80.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mpdd_300e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 92.6 | 91.8 | <u>92.8</u> | 90.8 | 30.6 | 32.9 | 78.3 | 24.1 | 75.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mpdd_300e.log)  |




### MAD_Real-300epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 46.3 | 82.5 | 90.2 | 42.5 | 0.5 | 1.2 | 14.9 | 0.6 | 43.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mad_real_300e.log)  |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 36.1 | 77.0 | 89.6 | 77.2 | 2.6 | 6.3 | 48.9 | 3.3 | 50.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mad_real_300e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 48.5 | 81.2 | 91.4 | 83.3 | 10.4 | 15.0 | 64.1 | 8.6 | 58.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mad_real_300e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 57.4 | <u>85.9</u> | 91.2 | <u>90.9</u> | 10.8 | 15.3 | 74.1 | 8.8 | 63.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/mad_real_300e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 50.6 | 82.9 | 90.5 | 88.6 | 7.5 | 12.0 | 68.8 | 6.7 | 59.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mad_real_300e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 56.0 | 84.9 | 90.7 | 86.9 | 8.7 | 13.2 | 66.9 | 7.4 | 60.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mad_real_300e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | <u>58.0</u> | 85.7 | 91.1 | 87.8 | 5.1 | 9.6 | 69.4 | 5.2 | 58.1 | [log]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 44.5 | 81.3 | 90.2 | 86.9 | 4.5 | 8.9 | 65.0 | 4.8 | 56.7 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mad_ral_300e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **63.5** | **86.5** | **92.7** | **91.6** | **18.9** | **24.2** | **77.2** | **14.9** | **66.9** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mad_real_300e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 57.3 | 84.8 | 91.9 | 90.8 | 11.4 | 16.3 | <u>74.7</u> | 9.5 | <u>63.1</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mad_real_300e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 54.7 | 83.1 | <u>92.2</u> | 90.8 | <u>13.1</u> | <u>19.2</u> | <u>74.7</u> | <u>11.4 | <u>63.1</u> |[log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mad_real_300e.log)   |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 45.4 | 80.3 | 90.8 | 87.8 | 6.4 | 11.6 | 68.3 | 6.4 | 57.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mad_real_300e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 52.3 | 84.0 | 91.3 | 88.7 | 9.3 | 14.4 | 70.4 | 8.2 | 60.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mad_real_300e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 36.1 | 77.9 | 89.8 | 82.5 | 3.2 | 6.0 | 52.7 | 3.2 | 52.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mad_real_300e.log)  |



### MAD_Sim-300epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 48.8 | 88.1 | 93.8 | 47.9 | 0.7 | 1.5 | 20.0 | 0.8 | 47.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/mad_sim_300e.log)  |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 51.5 | 88.6 | 93.8 | 86.1 | 2.3 | 4.5 | 63.9 | 2.3 | 58.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/mad_sim_300e.log)  |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 55.7 | 91.1 | 93.8 | 71.8 | 3.7 | 7.7 | 45.3 | 4.1 | 56.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/mad_sim_300e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 56.9 | 90.9 | 93.8 | 86.9 | 3.4 | 5.1 | 63.7 | 2.6 | 60.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/mad_sim_300e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 49.0 | 88.1 | 93.8 | 84.4 | 2.4 | 5.0 | 62.7 | 2.6 | 57.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/mad_sim_300e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 57.9 | 91.3 | 93.8 | 88.3 | 4.4 | 8.2 | 73.7 | 4.3 | 62.3 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/mad_sim_300e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | <u>65.6</u> | <u>93.4</u> | **94.1** | 87.7 | 4.4 | 8.8 | 61.4 | 4.7 | 59.4 | [log]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 59.0 | 91.4 | 93.8 | 89.0 | 4.8 | 9.3 | 73.6 | 4.9 | 62.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/mad_sim_300e.log)  |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | **70.4** | **94.5** | <u>94.0</u> | **90.7** | **8.9** | **15.3** | **79.1** | **8.4** | **67.4** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/mad_sim_300e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 65.3 | <u>93.4</u> | 93.9 | <u>89.7</u> | <u>6.6</u> | <u>11.4</u> | <u>76.4</u> | <u>6.1</u> | <u>65.1</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/mad_sim_300e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | 61.4 | 92.4 | 93.8 | 88.3 | 4.8 | 9.0 | 73.0 | 4.8 | 63.2 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/mad_sim_300e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 59.4 | 91.5 | 93.8 | 89.0 | 4.4 | 8.4 | 74.6 | 4.4 | 62.8 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/mad_sim_300e.log)  |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 58.6 | 91.7 | 93.8 | 88.3 | 4.4 | 8.1 | 73.2 | 4.2 | 62.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/mad_sim_300e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 59.2 | 91.9 | 93.8 | 56.8 | 5.7 | 8.6 | 41.8 | 4.5 | 54.9 |  [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/mad_sim_300e.log) |




### Uni-Medical-300epochs
| Method | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | <span style="color:red">mAU-PRO<sub>R</sub></span> | <span style="color:red">mIoU-max<sub>P</sub></span> | <span style="color:red">mAD</span> | <span style="color:blue">Download</span> |
|:-------------------------------------------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:---------------------------------------------------:|:-----------------------:|:--------:|:--------:|
| [DRAEM, ICCV'21](https://github.com/VitjanZ/DRAEM) | 58.5 | 63.2 | 71.7 | 32.5 | 2.0 | 6.1 | 9.1 | 3.2 | 38.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/draem/medical_300e.log)  |
| [SimpleNet, CVPR'23](https://github.com/DonaldRR/SimpleNet) | 72.5 | 75.3 | 76.7 | 94.9 | 35.9 | 37.0 | 79.2 | 24.2 | 68.3 |  [log](https://github.com/zhangzjn/data/tree/main/ader_log/simplenet/medical_300e.log) |
| [CFA, Access'22](https://github.com/sungwool/CFA_for_anomaly_localization) | 56.3 | 62.8 | 70.4 | 38.1 | 4.8 | 6.6 | 8.9 | 3.4 | 38.9 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cfa/medical_300e.log)  |
| [CFLOW-AD, WACV'22](https://github.com/gudovskiy/cflow-ad) | 79.0 | 78.5 | 77.3 | 95.4 | 39.6 | 26.3 | 81.7 | 17.0 | 69.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/cflow/medical_300e.log)  |
| [PyramidalFlow, CVPR'23](https://github.com/gasharper/PyramidFlow) | 60.0 | 62.8 | 71.9 | 82.2 | 8.8 | 14.9 | 46.9 | 8.3 | 51.5 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/pyramidflow/medical_300e.log)  |
| [RD, CVPR'22](https://github.com/hq-deng/RD4AD) | 76.4 | 75.8 | 77.9 | 96.4 | 38.9 | 39.8 | 86.5 | 27.0 | 71.1 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd/medical_300e.log)  |
| [DiAD, AAAI'24](https://github.com/lewandofskee/DiAD) | **82.9** | <u>81.9</u> | 79.4 | 96.0 | 35.9 | 38.2 | 85.4 | 25.2 | 71.4 | [log]()   |
| [ViTAD, arXiv'23](https://zhangzjn.github.io/projects/ViTAD) | 81.5 | 80.6 | <u>80.1</u> | <u>97.0</u> | <u>46.8</u> | <u>46.9</u> | 86.5 | <u>32.7</u> | **75.0** | [log](https://github.com/zhangzjn/data/tree/main/ader_log/vitad/medical_300e.log)   |
| [InvAD, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 81.3 | 80.0 | 79.9 | **97.2** | 45.4 | 45.5 | **89.0** | 32.3 | <u>74.8</u> | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad/medical_300e.log)  |
| [InvAD-lite, arXiv'24](https://zhangzjn.github.io/projects/InvAD) | 77.6 | 77.2 | 79.3 | 96.2 | 38.7 | 39.7 | 85.7 | 27.1 | 71.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/invad-lite/medical_300e.log)  |
| [MambaAD, arXiv'24](https://github.com/lewandofskee/MambaAD) | <u>82.3</u> | 78.9 | **81.3** | 96.7 | 43.0 | 46.1 | <u>87.9</u> | 32.4 | 74.6 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/mambaad/medical_300e.log)  |
| [UniAD, NeurIPS'22](https://github.com/zhiyuanyou/UniAD) | 80.4 | 76.6 | 77.5 | 96.5 | 39.0 | 41.3 | 85.8 | 27.8 | 71.9 |  [log](https://github.com/zhangzjn/data/tree/main/ader_log/uniad/medical_300e.log) |
| [RD++, CVPR'23](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | 77.2 | 76.1 | 78.5 | 96.7 | 41.0 | 41.9 | 87.4 | 29.1 | 72.0 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/rd++/medical_300e.log)  |
| [DesTSeg, CVPR'23](https://github.com/apple/ml-destseg) | 82.0 | **85.0** | 79.7 | 83.2 | **50.6** | **50.5** | 66.1 | **34.1** | 72.4 | [log](https://github.com/zhangzjn/data/tree/main/ader_log/destseg/medical_300e.log)  |

