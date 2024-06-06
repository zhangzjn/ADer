import math
from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from configs.__base__ import *



class cfg(cfg_common, cfg_dataset_default, cfg_model_cflow):

	def __init__(self):
		cfg_common.__init__(self)
		cfg_dataset_default.__init__(self)
		cfg_model_cflow.__init__(self)

		self.seed = 42
		self.size = 256 # 512
		self.epoch_full = 30
		self.warmup_epochs = 0
		self.test_start_epoch = self.epoch_full * 10
		self.test_per_epoch = self.epoch_full
		self.batch_train = 32
		self.batch_test_per = 32
		self.lr = 2e-4

		self.weight_decay = 0.0
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
			dict(type='RandomRotation', degrees=5),
			dict(type='CenterCrop', size=(self.size, self.size)),
			dict(type='ToTensor'),
			dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		]

		self.data.test_transforms = [
			dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(self.size, self.size)),
			dict(type='ToTensor'),
			dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		]

		self.data.target_transforms = [
			dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(self.size, self.size)),
			dict(type='ToTensor'),
		]

		# ==> model
		self.model_backbone = Namespace()

		in_chas = [256, 512, 1024, 2048]
		name = 'timm_wide_resnet50_2'
		checkpoint_path = 'model/pretrain/wide_resnet50_2-95faca4d.pth'
		out_indices = [i + 1 for i in range(len(in_chas))]  # [1, 2, 3, 4]
		self.model_backbone.name = name
		self.model_backbone.device = 'cuda'
		self.model_backbone.dec_arch = 'freia-cflow'
		self.model_backbone.condition_vec = 128
		self.model_backbone.coupling_blocks = 8
		self.model_backbone.clamp_alpha = 1.9
		self.model_backbone.kwargs = dict(pretrained=True,
										 checkpoint_path='',
										 strict=False,
										 features_only=True, out_indices=out_indices)


		self.model = Namespace()
		self.model.name = 'cflow'
		self.model.pool_layers = 3
		self.model.N = 256
		self.model.kwargs = dict(
			pretrained=True, checkpoint_path='', strict=True,
			model_backbone=self.model_backbone, L=self.model.pool_layers, N=self.model.N)

		# ==> evaluator TODO to check
		# self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=[16, 16], max_step_aupro=100)
		self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=[16, 16], max_step_aupro=100, use_adeval=self.use_adeval)

		# ==> optimizer
		self.optim.lr = self.lr
		self.optim.kwargs = dict(name='adam', betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay, amsgrad=False)


		# ==> trainer
		self.trainer.name = 'CFLOWTrainer'
		self.trainer.logdir_sub = ''
		self.trainer.resume_dir = ''
		self.trainer.epoch_full = self.epoch_full
		
		self.trainer.lr = self.lr
		self.trainer.lr_warm_epochs = 2
		self.trainer.lr_cosine = True
		self.trainer.lr_warm = True
		self.trainer.lr_decay_rate = 0.1
		self.trainer.meta_epochs = self.epoch_full
		self.trainer.sub_epochs = 10
		# self.test_per_epoch = self.trainer.sub_epochs

		# self.trainer.lr_decay_epochs =  [i*self.trainer.meta_epochs//10 for i in [50,75,90]]
		self.trainer.lr_decay_epochs =  [i*self.trainer.meta_epochs//10 for i in [5,7,9]]
		self.trainer.lr_warmup_from = self.lr/10.0
		eta_min = self.lr * (self.trainer.lr_decay_rate ** 3)
		self.trainer.lr_warmup_to = eta_min + (self.lr - eta_min) * (
                    1 + math.cos(math.pi * self.trainer.lr_warm_epochs / self.trainer.meta_epochs)) / 2

		self.trainer.scheduler_kwargs = dict(
			name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=self.lr / 1e2,
			warmup_lr=self.lr / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs,
			cooldown_epochs=0, use_iters=True,
			patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8),
			cycle_decay=0.1, decay_rate=0.1)
		self.trainer.mixup_kwargs = dict(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.0, switch_prob=0.5, mode='batch', correct_lam=True, label_smoothing=0.1)
		self.trainer.test_start_epoch = self.test_start_epoch
		self.trainer.test_per_epoch = self.test_per_epoch
		
		self.trainer.data.batch_size = self.batch_train
		self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

		# ==> loss
		self.loss.clip_grad = None

		# ==> logging
		self.logging.log_terms_train = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='data_t', fmt=':>5.3f'),
			dict(name='optim_t', fmt=':>5.3f'),
			dict(name='lr', fmt=':>7.6f'),
			dict(name='pixel', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
		self.logging.log_terms_test = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='pixel', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]