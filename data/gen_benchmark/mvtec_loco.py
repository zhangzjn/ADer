import os
import json


class MVTecSolver(object):
    CLSNAMES = [
        'breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors',
    ]

    def __init__(self, root='data/mvtec', is2D=False):
        self.root = root
        self.is2D = is2D
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test', 'validation']
        self.CLSNAMES = self.CLSNAMES

    def run(self):
        info = {phase: {} for phase in self.phases}
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in self.phases:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_dir = f'{cls_dir}/{phase}/{specie}/'
                    mask_dir = f'{cls_dir}/ground_truth/{specie}'
                    img_names = os.listdir(img_dir)
                    mask_names = os.listdir(mask_dir) if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    if mask_names is not None:
                        mask_final_names = []
                        for i in range(len(mask_names)):
                            mask_new_dir = f'{mask_dir}/{mask_names[i]}'
                            mask_new_dir_name = os.listdir(mask_new_dir)
                            mask_final_names.append(mask_names[i] + '/' + mask_new_dir_name[0])

                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{img_dir.replace(cls_dir, cls_name)}/{img_name}',
                            mask_path=f'{mask_dir.replace(cls_dir, cls_name)}/{mask_final_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")


if __name__ == '__main__':
    # runner = MVTecSolver(root='data/mvtec', is2D=True)
    # runner.run()
    runner = MVTecSolver(root='/fuxi_team14/users/haoyanghe/codes/data/ft_local/mvtec_loco', is2D=False)
    runner.run()
