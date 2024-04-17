import subprocess
from multiprocessing import Process
import argparse


def runcmd(command):
    # ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    ret = subprocess.run(command, shell=True)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='mvtec')
    parser.add_argument('-c', '--config', type=str, default='configs/rd/rd_mvtec.py')
    parser.add_argument('-p', '--prefix', type=str, default='single_cls')
    parser.add_argument('-n', '--nproc_per_node', type=int, default=8)
    parser.add_argument('-m', '--multiprocessing', type=int, default=-1)
    parser.add_argument('-g', '--gpu_num', type=int, default=0)
    cfg = parser.parse_args()

    if cfg.dataset == 'mvtec':
        names = ['carpet', 'grid', 'leather', 'tile', 'wood',
                 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                 'pill', 'screw', 'toothbrush', 'transistor', 'zipper', ]
    elif cfg.dataset == 'mvtec3d':
        names = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                 'foam', 'peach', 'potato', 'rope', 'tire', ]
    elif cfg.dataset == 'visa':
        names = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
                 'pcb4', 'pipe_fryum', ]
    config = cfg.config
    suffix = config.split('/')[-1].split('.')[0]
    nproc_per_node = cfg.nproc_per_node
    use_multiprocessing = True if cfg.multiprocessing > 0 else False
    if use_multiprocessing:
        process_list = []
        for i, name in enumerate(names):
            command = f'CUDA_VISIBLE_DEVICES={i % nproc_per_node} '
            command += f'python3 run.py -c {cfg.config} -m train data.cls_names={name} trainer.checkpoint=runs/{cfg.prefix}/{suffix}/{name}'
            p = Process(target=runcmd, args=(command,))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()
    else:
        for i, name in enumerate(names):
            command = f'CUDA_VISIBLE_DEVICES={cfg.gpu_num} '
            command += f'python3 run.py -c {cfg.config} -m train data.cls_names={name} trainer.checkpoint=runs/{cfg.prefix}/{suffix}/{name}'
            ret = runcmd(command)
            print(command, ret)