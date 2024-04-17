import glob
import os
import shutil


# solve imagenet
for phase in ['val']:
    target_folder = f'data/tiny-imagenet-200/{phase}'
    phase_dict = {}
    with open(f'{target_folder}/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            phase_dict[split_line[0]] = split_line[1]
    paths = glob.glob(f'{target_folder}/images/*')
    for path in paths:
        file = path.split('/')[-1]
        file_folder = phase_dict[file]
        if not os.path.exists(f'{target_folder}/{file_folder}/images'):
            os.makedirs(f'{target_folder}/{file_folder}/images', exist_ok=True)
        path_dest = f'{target_folder}/{file_folder}/images/{file}'
        shutil.copy(path, path_dest)
    shutil.rmtree(f'{target_folder}/images')

