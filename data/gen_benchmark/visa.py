import os
import json
import pandas as pd


class VisASolver(object):
    CLSNAMES = [
        'pcb1', 'pcb2', 'pcb3', 'pcb4',
        'macaroni1', 'macaroni2', 'capsules', 'candle',
        'cashew', 'chewinggum', 'fryum', 'pipe_fryum',
    ]

    def __init__(self, root='data/visa'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test']
        self.csv_data = pd.read_csv(f'{root}/split_csv/1cls.csv', header=0)

    def run(self):
        columns = self.csv_data.columns  # [object, split, label, image, mask]
        info = {phase: {} for phase in self.phases}
        for cls_name in self.CLSNAMES:
            cls_data = self.csv_data[self.csv_data[columns[0]] == cls_name]
            for phase in self.phases:
                cls_info = []
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                cls_data_phase.index = list(range(len(cls_data_phase)))
                for idx in range(cls_data_phase.shape[0]):
                    data = cls_data_phase.loc[idx]
                    is_abnormal = True if data[2] == 'anomaly' else False
                    info_img = dict(
                        img_path=data[3],
                        mask_path=data[4] if is_abnormal else '',
                        cls_name=cls_name,
                        specie_name='',
                        anomaly=1 if is_abnormal else 0,
                    )
                    cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")


if __name__ == '__main__':
    runner = VisASolver(root='data/visa')
    runner.run()



# class VisASolver(object):
#     CLSNAMES = [
#         'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
#         'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
#         'pcb4', 'pipe_fryum',
#     ]
#
#     def __init__(self, root='data/visa')
#         self.root = root
#         self.meta_path = f'{root}/meta.json'
#
#     def run(self):
#         info = dict(Normal={}, Anomaly={})
#         for cls_name in self.CLSNAMES:
#             cls_dir = f'{self.root}/{cls_name}'
#             for phase in ['Normal', 'Anomaly']:
#                 cls_info = []
#                 is_abnormal = True if phase == 'Anomaly' else False
#                 img_names = os.listdir(f'{cls_dir}/Data/Images/{phase}')
#                 mask_names = os.listdir(f'{cls_dir}/Data/Images/{phase}') if is_abnormal else None
#                 img_names.sort()
#                 mask_names.sort() if mask_names is not None else None
#
#                 for idx, img_name in enumerate(img_names):
#                     info_img = dict(
#                         img_path=f'{cls_name}/Data/Images/{phase}/{img_name}',
#                         mask_path=f'{cls_name}/Data/Masks/{phase}/{mask_names[idx]}' if is_abnormal else '',
#                         cls_name=cls_name,
#                         anomaly=1 if is_abnormal else 0,
#                     )
#                     cls_info.append(info_img)
#                 info[phase][cls_name] = cls_info
#         with open(self.meta_path, 'w') as f:
#             f.write(json.dumps(info, indent=4) + "\n")
#
#
# if __name__ == '__main__':
#     runner = VisASolver(root='data/visa')
#     runner.run()
