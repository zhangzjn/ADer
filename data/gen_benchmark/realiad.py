import os
import json
from torch.utils.data import Dataset, DataLoader

class MVTecSolver(object):
    CLSNAMES = [
        'audiojack', 'pcb', 'phone_battery', 'sim_card_set', 'switch', 'terminalblock',
		 					   'toothbrush', 'toy', 'transistor1', 'usb', 'usb_adaptor', 'zipper', 'bottle_cap',
		 					   'end_cap', 'fire_hood', 'mounts', 'plastic_nut', 'plastic_plug', 'regulator',
		 					   'rolled_strip_base', 'toy_brick', 'u_block', 'vcpill', 'wooden_beads', 'woodstick',
		 					   'tape', 'porcelain_doll', 'mint', 'eraser', 'button_battery'
    ]

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test']
        self.CLSNAMES = self.CLSNAMES

    def run(self):
        json_data = []
        num_train = 0
        # 遍历目录中的所有文件
        for filename in os.listdir(self.root):
            if filename.endswith('.json'):
                file_path = os.path.join(self.root, filename)
                # 读取 JSON 文件内容
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    for image in data['test']:
                        if image['anomaly_class'] == 'OK':
                            num_train += 1
                    # num_train = num_train + len(data['train'])
                    # json_data.append(data)
        print(num_train)
        # return json_data


if __name__ == '__main__':
    # runner = MVTecSolver(root='data/mvtec', is2D=True)
    # runner.run()
    runner = MVTecSolver(root='/fuxi_team14/users/haoyanghe/codes/ad/ader_hhytest/data/realiad')
    runner.run()
