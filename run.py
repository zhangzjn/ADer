import argparse
from configs import get_cfg
from util.net import init_training
from util.util import run_pre, init_checkpoint
from trainer import get_trainer
import warnings
warnings.filterwarnings("ignore")


def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument('-c', '--cfg_path', default='configs/rd_mvtec_debug.py')
	# parser.add_argument('-c', '--cfg_path', default='configs/invad_mvtec_debug.py')
	parser.add_argument('-c', '--cfg_path', default='configs/vitad_mvtec_debug.py')
	parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
	parser.add_argument('--sleep', type=int, default=-1)
	parser.add_argument('--memory', type=int, default=-1)
	parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
	parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
	parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER,)
	cfg_terminal = parser.parse_args()
	cfg = get_cfg(cfg_terminal)
	run_pre(cfg)
	init_training(cfg)
	init_checkpoint(cfg)
	trainer = get_trainer(cfg)
	trainer.run()


if __name__ == '__main__':
	main()
