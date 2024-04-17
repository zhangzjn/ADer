import json
import os
import shutil

import numpy as np
import cv2
from pycocotools.coco import maskUtils
from mmdet.datasets.coco import CocoDataset

def convert_png(path: str):
    return path.split('.')[0] + '.png'


class COCOSolver(object):
    def __init__(self, root='data/coco', ano_num=20, split_idx=0):
        self.root = root
        self.meta_path = f'{root}/meta_{ano_num}_{split_idx}.json'
        self.ano_num = ano_num
        self.split_idx = split_idx
        self.coco_train = CocoDataset(f'{root}/annotations/instances_train2017.json', pipeline=[])
        self.coco_val = CocoDataset(f'{root}/annotations/instances_val2017.json', pipeline=[])
        self.cat2label = self.coco_train.cat2label
        self.label2cat = {l: c for c, l in self.cat2label.items()}

        SPLITS = []
        for group in range(80 // ano_num):
            labels = list(range(0, 80, 1))
            labels_rm = list(range(group * ano_num, (group + 1) * ano_num))
            for label_rm in labels_rm:
                labels.remove(label_rm)
            SPLITS.append(dict(train=labels, test=labels_rm))
        splits = SPLITS[split_idx]
        labels_train = splits['train']
        self.cats_train = [self.label2cat[l] for l in labels_train]

    def run(self):
        test_no, test_abno = 0, 0
        info = dict(train={}, test={})
        info['train']['coco'] = []
        info['test']['coco'] = []
        for idx, coco in enumerate([self.coco_train, self.coco_val]):
            # mode = 'train' if idx == 0 else 'val'
            # mask_dir = f'{self.root}/{mode}2017_mask_ad_{self.ano_num}_{self.split_idx}'
            # if os.path.exists(mask_dir):
            #     shutil.rmtree(mask_dir)
            # os.makedirs(mask_dir, exist_ok=True)
            if idx == 0:
                mode = 'train'
                mask_dir = f'{self.root}/{mode}2017_mask_ad_{self.ano_num}_{self.split_idx}'
            else:
                mode = 'val'
                mask_dir = f'{self.root}/{mode}2017_mask_ad_{self.ano_num}_{self.split_idx}'
                if os.path.exists(mask_dir):
                    shutil.rmtree(mask_dir)
                os.makedirs(mask_dir, exist_ok=True)

            cnt = 0
            for img_id, img_anns in coco.coco.imgToAnns.items():
                cnt += 1
                img_info = coco.coco.imgs[img_id]
                img_anns = coco.coco.imgToAnns[img_id]
                img_name = img_info['file_name']
                img_path = f'{mode}2017/{img_name}'
                mask_path = f'{mode}2017_mask_ad_{self.ano_num}_{self.split_idx}/{convert_png(img_name)}'
                img = cv2.imread(f'{self.root}/{img_path}')
                img_mask = np.zeros(img.shape, dtype=img.dtype)
                isnormal = True
                for img_ann in img_anns:
                    if img_ann['category_id'] not in self.cats_train:
                        isnormal = False
                        if type(img_ann['segmentation']) == list:
                            for seg in img_ann['segmentation']:
                                img_mask = cv2.fillPoly(img_mask, [np.array(seg).astype(np.int64).reshape((len(seg) // 2, 2))], (255, 255, 255))
                        else:
                            if type(img_ann['segmentation']['counts']) == list:
                                rle = maskUtils.frPyObjects([img_ann['segmentation']], img_info['height'], img_info['width'])
                            else:
                                rle = [img_ann['segmentation']]
                            mask = maskUtils.decode(rle)
                            img_mask[mask.repeat(3, axis=2) > 0.5] = 255
                if isnormal:
                    if idx == 0:
                        info['train']['coco'].append(dict(img_path=img_path, mask_path='', cls_name='coco', specie_name='', anomaly=0))
                    elif idx == 1:
                        info['test']['coco'].append(dict(img_path=img_path, mask_path='',  cls_name='coco', specie_name='', anomaly=0))
                        test_no += 1
                else:
                    if idx == 0:
                        pass  # only normal data in val set is used
                        # info['test']['coco'].append(dict(img_path=img_path, mask_path=mask_path, cls_name='coco', specie_name='', anomaly=1))
                        # cv2.imwrite(f'{self.root}/{mask_path}', img_mask)
                    elif idx == 1:
                        info['test']['coco'].append(dict(img_path=img_path, mask_path=mask_path, cls_name='coco', specie_name='', anomaly=1))
                        cv2.imwrite(f'{self.root}/{mask_path}', img_mask)
                        test_abno += 1
                print(f'\r{idx+1}/2 | {cnt}/{len(coco.coco.imgToAnns)}', end='')
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")


if __name__ == '__main__':
    for i in range(0, 4):
        runner = COCOSolver(root='data/coco', ano_num=20, split_idx=i)
        # runner.run()
