from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F
from PIL import Image
from configs.__base__ import *


class cfg(cfg_common, cfg_dataset_default, cfg_model_realnet):

	def __init__(self):
		cfg_common.__init__(self)
		cfg_dataset_default.__init__(self)
		cfg_model_realnet.__init__(self)

		self.seed = 42
		self.size = 256
		self.epoch_full = 100
		self.warmup_epochs = 0
		self.test_start_epoch = self.epoch_full
		self.test_per_epoch = self.epoch_full // 10
		self.batch_train = 16
		self.batch_test_per = 16
		self.lr = 1e-4 # * self.batch_train / 8
		self.weight_decay = 0.0 # adam use 0.0 by default

		self.metrics = [
			'mAUROC_sp_max', 'mAP_sp_max', 'mF1_max_sp_max',
			'mAUPRO_px',
			'mAUROC_px', 'mAP_px', 'mF1_max_px',
			'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1',
			'mIoU_max_px',
		]
		self.use_adeval = True

		# ==> data
		self.data.type = 'Realnet'
		self.data.root = 'data/mvtec'
		self.data.meta = 'meta.json'
		self.data.cls_names = []

		self.data.anomaly_source_path = 'data/dtd/images/'
		self.data.resize_shape = [self.size, self.size]
		self.data.sdas_dir = 'data/sdas'

		self.data.use_sample = False
		self.data.views = []  # ['C1', 'C2', 'C3', 'C4', 'C5']

		self.data.cls_names = []
		self.data.resize = self.size
		self.data.dataset = 'mvtec'
		self.data.anomaly_types = {'normal':0.5, 'sdas':0.5}
		self.data.dtd_transparency_range = [0.2, 1.0]
		self.data.sdas_transparency_range = [0.5, 1.0]
		self.data.perlin_scale = 6
		self.data.perlin_noise_threshold = 0.5
		self.data.min_perlin_scale = 0

		self.data.train_transforms = [
			dict(type='ToTensor'),
			dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=False),
		]
		self.data.test_transforms = [
			dict(type='Resize', size=(self.size, self.size), interpolation=Image.BILINEAR),
			dict(type='ToTensor'),
			dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=False),
		]
		self.data.target_transforms = [
			dict(type='Resize', size=(self.size, self.size), interpolation=Image.NEAREST),
			dict(type='ToTensor'),
		]

		# ==> model
		self.model_backbone = Namespace()
		# name = 'wide_resnet50_2'
		checkpoint_path = 'model/pretrain/wide_resnet50_racm-8234f177.pth'
		self.model_backbone.name = 'timm_wide_resnet50_2'
		out_indices = []
		self.model_backbone.kwargs = dict(pretrained=True,
									checkpoint_path='',
									strict=False,
									features_only=True, out_indices=out_indices)
		
		self.model_afs = Namespace()
		self.model_afs.init_bsn = 64
		self.model_afs.device = 'cuda'
		self.model_afs.structure = [
			dict(name='block1', layers=[dict(idx='layer1', planes=256)],stride=4),
			dict(name='block2', layers=[dict(idx='layer2', planes=512)],stride=8),
			dict(name='block3', layers=[dict(idx='layer3', planes=512)],stride=16),
			dict(name='block4', layers=[dict(idx='layer4', planes=256)],stride=32)]

		self.model_recon = Namespace()
		self.model_recon.kwargs = dict(num_res_blocks=2, hide_channels_ratio=0.5, channel_mult=[1,2,4], attention_mult=[2,4])
		
		self.model_rrs = Namespace()
		self.model_rrs.kwargs = dict(modes = ['max','mean'], mode_numbers = [256,256], num_residual_layers = 2, stop_grad = False)
		
		self.model = Namespace()
		self.model.name = 'realnet'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_backbone=self.model_backbone, 
						   model_afs=self.model_afs, model_recon=self.model_recon, model_rrs=self.model_rrs, data_cfg = self)
		
		# ==> trainer
		self.trainer.name = 'RealNetTrainer'
		self.trainer.logdir_sub = ''
		self.trainer.resume_dir = ''
		self.trainer.epoch_full = self.epoch_full
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
		self.loss.loss_terms = [
			dict(type='SegmentCELoss', name='seg', weight=1.0),
			dict(type='L2Loss', name='pixel', lam=1.0),
		]
		self.loss.clip_grad = None


		# ==> evaluator
		self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=[16, 16], max_step_aupro=100, use_adeval=self.use_adeval)
		
		# ==> optimizer
		self.optim.lr = self.lr
		self.optim.kwargs = dict(name='adam', betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay, amsgrad=False)

		# ==> logging
		self.logging.log_terms_train = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='data_t', fmt=':>5.3f'),
			dict(name='optim_t', fmt=':>5.3f'),
			dict(name='lr', fmt=':>7.6f'),
			dict(name='seg', suffixes=[''], fmt=':>5.3f', add_name='avg'),
			dict(name='pixel', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
		self.logging.log_terms_test = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='seg', suffixes=[''], fmt=':>5.3f', add_name='avg'),
			dict(name='pixel', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]