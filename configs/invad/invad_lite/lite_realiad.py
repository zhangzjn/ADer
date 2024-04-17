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
		self.test_per_epoch = self.epoch_full // 1
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
		self.data.type = 'RealIAD'
		self.data.root = 'data/realiad'
		self.data.use_sample = False
		self.data.views = []  # ['C1', 'C2', 'C3', 'C4', 'C5']
		self.data.cls_names = []

		# self.data.type = 'DefaultAD'
		# self.data.root = 'data/coco'
		# self.data.meta = 'meta_20_0.json'
		# self.data.cls_names = ['coco']

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

		multi_stride = 1
		# ==> timm_efficientnet_b4
		# in_chas = [32, 56, 160]
		# latent_channel_size = 64
		# self.model_encoder = Namespace()
		# self.model_encoder.name = 'timm_efficientnet_b4'
		# self.model_encoder.kwargs = dict(pretrained=False,
		# 								 checkpoint_path='model/pretrain/tf_efficientnet_b4_aa-818f208c.pth',
		# 								 strict=False,
		# 								 features_only=True, out_indices=[1, 2, 3])
		# self.model_fuser = Namespace()
		# self.model_fuser.kwargs = dict(in_chas=in_chas, in_strides=[4, 2, 1], down_conv=True,
		# 							   out_cha=latent_channel_size, out_stride=2, bottle_num=3)
		#
		# latent_spatial_size = self.size // (2 ** 4) // 2
		# self.model_decoder = Namespace()
		# self.model_decoder.kwargs = dict(in_chas=in_chas + [latent_channel_size],
		# 								 latent_spatial_size=latent_spatial_size,
		# 								 latent_channel_size=latent_channel_size,
		# 								 latent_multiplier=1, blur_kernel=[1, 3, 3, 1], normalize_mode='LayerNorm',
		# 								 lr_mul=0.01, small_generator=False, layers=[2] * 4)
		#
		# self.model = Namespace()
		# self.model.name = 'invad'
		# self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True,
		# 						 model_encoder=self.model_encoder,
		# 						 model_fuser=self.model_fuser,
		# 						 model_decoder=self.model_decoder, stages=[1, 4])

		# ==> timm_efficientnet_b4
		# in_chas = [32, 56, 160]
		# name = 'timm_efficientnet_b4'
		# checkpoint_path = 'model/pretrain/tf_efficientnet_b4_aa-818f208c.pth'
		# out_indices = [i + 1 for i in range(len(in_chas))]  # [1, 2, 3] + 1

		# ==> timm_wide_resnet
		# in_chas = [256, 512, 1024]
		# name = 'timm_wide_resnet50_2'
		# checkpoint_path = 'model/pretrain/wide_resnet50_racm-8234f177.pth'
		# name = 'timm_wide_resnet101_2'
		# checkpoint_path = 'model/pretrain/wide_resnet101_2-32ee1156.pth'
		# name = 'timm_resnet14t'
		# checkpoint_path = 'model/pretrain/resnet14t_176_c3-c4ed2c37.pth'
		# name = 'timm_resnext50_32x4d'
		# checkpoint_path = 'model/pretrain/resnext50_32x4d_a1h-0146ab0a.pth'
		# out_indices = [i + 1 for i in range(len(in_chas))]  # [1, 2, 3]

		# ==> timm_resnet50
		# in_chas = [256, 512, 1024]
		# name = 'timm_resnet50'
		# checkpoint_path = 'model/pretrain/semi_weakly_supervised_resnet50-16a12f1b.pth'
		# name = 'timm_resnet50_gn'
		# checkpoint_path = 'model/pretrain/resnet50_gn_a1h2-8fe6c4d0.pth'
		# out_indices = [i + 1 for i in range(len(in_chas))]  # [1, 2, 3]

		# ==> timm_resnet34
		in_chas = [64, 128, 256]
		name = 'timm_resnet34'
		checkpoint_path = 'model/pretrain/resnet34-43635321.pth'
		out_indices = [i + 1 for i in range(len(in_chas))]  # [1, 2, 3]

		# ==> timm_convnext_tiny
		# in_chas = [96, 192, 384]  # 96, 192, 384, 768
		# name = 'timm_convnext_tiny'
		# checkpoint_path = 'model/pretrain/convnext_tiny_22k_1k_224.pth'
		# out_indices = [i for i in range(len(in_chas))]  # [0, 1, 2]

		# ==> timm_convnextv2_tiny
		# in_chas = [96, 192, 384]  # 96, 192, 384, 768
		# name = 'timm_convnextv2_tiny'
		# checkpoint_path = 'model/pretrain/convnextv2_tiny_22k_224_ema.pt'
		# out_indices = [i for i in range(len(in_chas))]  # [0, 1, 2]

		# ==> timm_hrnet
		# in_chas = [128, 256, 512]
		# name = 'timm_hrnet_w18'
		# checkpoint_path = 'model/pretrain/hrnetv2_w18-8cb57bb9.pth'
		# name = 'timm_hrnet_w32'
		# checkpoint_path = 'model/pretrain/hrnetv2_w32-90d8c5fb.pth'
		# out_indices = [i + 1 for i in range(len(in_chas))]  # [1, 2, 3]

		# ==> timm_vgg16
		# in_chas = [128, 256, 512]
		# # name = 'timm_vgg16'
		# # checkpoint_path = 'model/pretrain/vgg16-397923af.pth'
		# name = 'timm_vgg16_bn'
		# checkpoint_path = 'model/pretrain/vgg16_bn-6c64b313.pth'
		# out_indices = [i + 1 for i in range(len(in_chas))]  # [1, 2, 3]
		# multi_stride = 2

		# ==> timm_swin_tiny_patch4_window7_224
		# in_chas = [96, 192, 384]  # 96, 192, 384, 768
		# name = 'timm_swin_tiny_patch4_window7_224'
		# checkpoint_path = 'model/pretrain/swin_tiny_patch4_window7_224.pth'
		# out_indices = [i for i in range(len(in_chas))]  # [0, 1, 2]

		# ==> timm_edgenext
		# in_chas = [24, 48, 88]
		# name = 'timm_edgenext_xx_small'
		# checkpoint_path = 'model/pretrain/edgenext_xx_small.pth'
		# out_indices = [i for i in range(len(in_chas))]  # [0, 1, 2]

		# ==> timm_ghostnet
		# in_chas = [24, 40, 80]
		# name = 'timm_ghostnet_100'
		# checkpoint_path = 'model/pretrain/ghostnet_1x.pth'
		# out_indices = [i + 1 for i in range(len(in_chas))]  # [1, 2, 3]

		# ==> timm_pvt_v2
		# in_chas = [64, 128, 320]
		# name = 'timm_pvt_v2_b1'
		# checkpoint_path = 'model/pretrain/pvt_v2_b1.pth'
		# out_indices = [i + 1 for i in range(len(in_chas))]  # [1, 2, 3]

		# ==> timm_poolformer
		# in_chas = [64, 128, 320]
		# name = 'timm_poolformer_s12'
		# checkpoint_path = 'model/pretrain/poolformer_s12.pth.tar'
		# out_indices = [i + 1 for i in range(len(in_chas))]  # [1, 2, 3]

		out_cha = 64
		style_chas = [min(in_cha, out_cha) for in_cha in in_chas]
		in_strides = [2 ** (len(in_chas) - i - 1) for i in range(len(in_chas))]  # [4, 2, 1]
		latent_channel_size = 16
		self.model_encoder = Namespace()
		self.model_encoder.name = name
		# self.pth = 'model/pretrain/semi_weakly_supervised_resnet50-16a12f1b.pth'
		self.pth = checkpoint_path
		self.model_encoder.kwargs = dict(pretrained=False,
										 checkpoint_path=self.pth,
										 strict=False,
										 features_only=True, out_indices=out_indices)

		self.model_fuser = dict(
			type='Fuser', in_chas=in_chas, style_chas=style_chas, in_strides=in_strides, down_conv=True, bottle_num=1, conv_num=1, lr_mul=0.01,
			# type='MultiScaleFuser', in_chas=in_chas, style_chas=style_chas, in_strides=[4, 2, 1], bottle_num=1, cross_reso=True
		)

		latent_spatial_size = self.size // (2 ** (1 + len(in_chas)))
		self.model_decoder = dict(in_chas=in_chas, style_chas=style_chas,
								  latent_spatial_size=latent_spatial_size, latent_channel_size=latent_channel_size,
								  blur_kernel=[1, 3, 3, 1], normalize_mode='LayerNorm',
								  lr_mul=0.01, small_generator=True, layers=[4] * len(in_chas))

		sizes = [self.size // (2 ** (2 + i)) for i in range(len(in_chas))]
		self.model_disor = dict(sizes=sizes, in_chas=in_chas)

		self.model = Namespace()
		self.model.name = 'invad'
		self.model.kwargs = dict(pretrained=False,
								 checkpoint_path='',
								 # checkpoint_path='runs/InvADTrainer_configs_invad_invad_mvtec_20230614-015946/net.pth',
								 strict=True,
								 model_encoder=self.model_encoder,
								 model_fuser=self.model_fuser,
								 model_decoder=self.model_decoder)

		# ==> evaluator
		self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=None, max_step_aupro=100)
		# self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=[16, 16], max_step_aupro=100)

		# ==> optimizer
		self.optim.lr = self.lr
		self.optim.kwargs = dict(name='adam', betas=(0, 0.99))
		# self.optim.kwargs = dict(name='adamw', betas=(0, 0.99), eps=1e-8, weight_decay=self.weight_decay, amsgrad=False)

		# ==> trainer
		self.trainer.name = 'InvADTrainer'
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
			dict(type='L2Loss', name='pixel', lam=1.0 * 5),
			# dict(type='CosLoss', name='pixel', flat=False, avg=False, lam=1.0),
			# dict(type='KLLoss', name='pixel', lam=1.0),
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
