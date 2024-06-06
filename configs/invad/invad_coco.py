from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from configs.__base__ import *


class cfg(cfg_common, cfg_dataset_default, cfg_model_invad):

	def __init__(self):
		cfg_common.__init__(self)
		cfg_dataset_default.__init__(self)
		cfg_model_invad.__init__(self)

		self.seed = 42
		self.size = 256
		self.epoch_full = 100
		self.warmup_epochs = 0
		self.test_start_epoch = self.epoch_full
		self.test_per_epoch = self.epoch_full // 10
		self.batch_train = 32
		self.batch_test_per = 32
		self.lr = 0.001 * self.batch_train / 8
		# self.lr = 1e-4 * self.batch_train / 8
		self.weight_decay = 0.0001
		self.metrics = [
			'mAUROC_sp_max', 'mAP_sp_max', 'mF1_max_sp_max',
			'mAUPRO_px',
			'mAUROC_px', 'mAP_px', 'mF1_max_px',
			'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1',
			'mIoU_max_px',
		]
		self.uni_am = True
		self.use_cos = True

		# ==> data
		self.data.type = 'DefaultAD'
		self.data.root = 'data/coco'
		self.data.meta = 'meta_20_0.json'
		self.data.cls_names = ['coco']

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

		# ==> timm_wide_resnet
		in_chas = [256, 512, 1024]
		name = 'timm_wide_resnet50_2'
		checkpoint_path = 'model/pretrain/wide_resnet50_racm-8234f177.pth'
		out_indices = [i + 1 for i in range(len(in_chas))]  # [1, 2, 3]

		out_cha = 64
		style_chas = [min(in_cha, out_cha) for in_cha in in_chas]
		in_strides = [2 ** (len(in_chas) - i - 1) for i in range(len(in_chas))]  # [4, 2, 1]
		latent_channel_size = 16
		self.model_encoder = Namespace()
		self.model_encoder.name = name
		self.model_encoder.kwargs = dict(pretrained=True,
										 checkpoint_path='',
										 strict=False,
										 features_only=True, out_indices=out_indices)

		self.model_fuser = dict(
			type='Fuser', in_chas=in_chas, style_chas=style_chas, in_strides=in_strides, down_conv=True, bottle_num=1,
			conv_num=1, lr_mul=0.01,
		)

		latent_spatial_size = self.size // (2 ** (1 + len(in_chas)))
		self.model_decoder = dict(in_chas=in_chas, style_chas=style_chas,
								  latent_spatial_size=latent_spatial_size, latent_channel_size=latent_channel_size,
								  blur_kernel=[1, 3, 3, 1], normalize_mode='LayerNorm',
								  lr_mul=0.01, small_generator=True, layers=[2] * len(in_chas))

		sizes = [self.size // (2 ** (2 + i)) for i in range(len(in_chas))]
		self.model_disor = dict(sizes=sizes, in_chas=in_chas)

		self.model = Namespace()
		self.model.name = 'invad'
		self.model.kwargs = dict(pretrained=False,
								 checkpoint_path='',
								 # checkpoint_path='runs/ablation_bs_invad_mvtec_multiple_class/InvADTrainer_configs_invad_bs_bs32_20230628-005543/net.pth',
								 strict=True,
								 model_encoder=self.model_encoder,
								 model_fuser=self.model_fuser,
								 model_decoder=self.model_decoder)

		# ==> evaluator
		self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=None, max_step_aupro=100)

		# ==> optimizer
		self.optim.lr = self.lr
		self.optim.kwargs = dict(name='adam', betas=(0, 0.99))

		# ==> trainer
		self.trainer.name = 'InvADTrainer'
		self.trainer.logdir_sub = ''
		self.trainer.resume_dir = ''
		self.trainer.epoch_full = self.epoch_full
		self.trainer.scheduler_kwargs = dict(
			name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=self.lr / 1e2,
			warmup_lr=self.lr / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs,
			cooldown_epochs=0, use_iters=True,
			patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8),
			cycle_decay=0.1, decay_rate=0.1)
		self.trainer.mixup_kwargs = dict(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.0,
										 switch_prob=0.5, mode='batch', correct_lam=True, label_smoothing=0.1)
		self.trainer.test_start_epoch = self.test_start_epoch
		self.trainer.test_per_epoch = self.test_per_epoch

		self.trainer.data.batch_size = self.batch_train
		self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

		# ==> loss
		self.loss.loss_terms = [
			dict(type='L2Loss', name='pixel', lam=1.0),
		]

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
