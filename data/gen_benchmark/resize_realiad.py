import os
from PIL import Image
import shutil
import argparse
import glob

def downresolution_datasets(folder_path, target_size, out_folder_path):
    # img_paths = glob.glob(f'{folder_path}/**/*.*', recursive=True)
    os.makedirs(out_folder_path, exist_ok=True)
    idx1 = 0
    for root, dirs, files in os.walk(folder_path):
        if len(files) > 0:
            idx1 += 1
        for idx2, file in enumerate(files):
            if file.endswith(".jpg") or file.endswith(".png"):
                input_path = os.path.join(root, file)
                output_folder = root.replace(folder_path, out_folder_path)
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, file)
                img = Image.open(input_path)
                resized_img = img.resize(target_size, Image.Resampling.BICUBIC)
                resized_img.save(output_file, quality=100)
            elif file.endswith(".json"):
                input_path = os.path.join(root, file)
                output_folder = root.replace(folder_path, out_folder_path)
                os.makedirs(output_folder, exist_ok=True)
                shutil.copy(input_path, output_folder)
            else:
                pass
            print(f'\r{idx1} {idx2}/{len(files)}', end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_path', type=str, default='./explicit_full/')
    parser.add_argument('-t', '--target_root', type=str, default='./')
    parser.add_argument('-s', '--target_reso', type=int, default=1024)
    cfg_terminal = parser.parse_args()
    target_tuple = (cfg_terminal.target_reso, cfg_terminal.target_reso)
    save_path = cfg_terminal.target_root + f'realiad_{cfg_terminal.target_reso}/'
    downresolution_datasets(cfg_terminal.root_path, target_tuple, save_path)

    # python3 resize_realiad.py --root_path /fuxi_team14/public/I3Datasets/AD/raw/coco_data_clean_final --target_root /fuxi_team14/public/I3Datasets/AD/ --target_reso 1024


