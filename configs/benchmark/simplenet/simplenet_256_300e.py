from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from ..__base__ import *


class cfg(cfg_common, cfg_dataset_default, cfg_model_simplenet):

	def __init__(self):
		cfg_common.__init__(self)
		cfg_dataset_default.__init__(self)
		cfg_model_simplenet.__init__(self)

		self.seed = 0
		# self.size = 329  # original paper
		# self.image_size = 288
		self.size = 256  # for fair comparison
		self.image_size = 256
		self.input_size = (3, self.image_size, self.image_size)
		self.epoch_full = 300
		self.warmup_epochs = 0
		self.test_start_epoch = self.epoch_full
		self.test_per_epoch = self.epoch_full // 10
		self.batch_train = 32  # official 8
		self.batch_test_per = 32
		self.lr = 0.001 * self.batch_train / 32
		self.weight_decay = 0.01
		self.metrics = [
			'mAUROC_sp_max', 'mAP_sp_max', 'mF1_max_sp_max',
			'mAUPRO_px',
			'mAUROC_px', 'mAP_px', 'mF1_max_px',
			'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1',
			'mIoU_max_px',
		]
		self.use_adeval = True

		# ==> data
		self.data.type = 'DefaultAD'
		self.data.root = 'data/mvtec'
		self.data.meta = 'meta.json'
		self.data.cls_names = []

		self.data.anomaly_source_path = 'data/dtd/images/'
		self.data.resize_shape = [self.size, self.size]

		self.data.use_sample = False
		self.data.views = []  # ['C1', 'C2', 'C3', 'C4', 'C5']

		self.data.train_transforms = [
			dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(self.size, self.size)),
			dict(type='ToTensor'),
			dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		]
		self.data.test_transforms = self.data.train_transforms
		self.data.target_transforms = [
			dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(self.size, self.size)),
			dict(type='ToTensor'),
		]

		# ==> modal
		checkpoint_path = 'model/pretrain/wide_resnet50_2-95faca4d.pth'
		self.model_backbone = Namespace()
		self.model_backbone.name = 'tv_wide_resnet50_2'
		self.model_backbone.kwargs = dict(pretrained=True,
										  checkpoint_path='', strict=False)

		self.layers_to_extract_from = ('layer2', 'layer3')
		self.model = Namespace()
		self.model.name = 'simplenet'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_backbone=self.model_backbone,
								 layers_to_extract_from=self.layers_to_extract_from, input_size=self.input_size)

		# ==> evaluator
		self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=None, max_step_aupro=100, use_adeval=self.use_adeval)

		# ==> optimizer
		self.optim.proj_opt = Namespace()
		self.optim.dsc_opt = Namespace()
		self.optim.lr = self.lr

		self.optim.proj_opt.kwargs = dict(name='adamw', betas=(0.9, 0.999), eps=1e-8,
										  weight_decay=self.weight_decay, amsgrad=False)
		self.optim.dsc_opt.kwargs = dict(name='adam', betas=(0.9, 0.999), eps=1e-8,
										 weight_decay=self.weight_decay * 1e-3, amsgrad=False)


		# ==> trainer
		self.trainer.name = 'SimpleNetTrainer'
		self.trainer.logdir_sub = ''
		self.trainer.resume_dir = ''
		self.trainer.epoch_full = self.epoch_full
		self.trainer.scheduler_kwargs = dict(
			name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=self.lr / 1e2,
			warmup_lr=self.lr / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs, cooldown_epochs=0, use_iters=True,
			patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8), cycle_decay=0.1, decay_rate=0.1)
		self.trainer.mixup_kwargs = dict(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.0, switch_prob=0.5, mode='batch', correct_lam=True, label_smoothing=0.1)
		self.trainer.test_start_epoch = self.test_start_epoch
		self.trainer.test_per_epoch = self.test_per_epoch

		self.trainer.data.batch_size = self.batch_train
		self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

		# ==> loss
		self.loss.loss_terms = [
			dict(type='SumLoss', name='sum', lam=1.0),
		]

		# ==> logging
		self.logging.log_terms_train = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='data_t', fmt=':>5.3f'),
			dict(name='optim_t', fmt=':>5.3f'),
			dict(name='lr', fmt=':>7.6f'),
			dict(name='sum', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
		self.logging.log_terms_test = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='sum', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
