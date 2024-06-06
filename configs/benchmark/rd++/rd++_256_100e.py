from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from configs.__base__ import *


class cfg(cfg_common, cfg_dataset_default, cfg_model_rd):

	def __init__(self):
		# super(cfg, self).__init__()
		cfg_common.__init__(self)
		cfg_dataset_default.__init__(self)
		cfg_model_rd.__init__(self)

		self.fvcore_b = 1
		self.fvcore_c = 3
		self.seed = 42
		self.size = 256
		self.epoch_full = 100
		self.warmup_epochs = 0
		self.test_start_epoch = self.epoch_full
		self.test_per_epoch = self.epoch_full // 10
		self.batch_train = 16  # official 16
		self.batch_test_per = 16
		self.lr = 0.001 * self.batch_train / 16
		self.weight_decay = 0.05
		self.metrics = [
			'mAUROC_sp_max', 'mAP_sp_max', 'mF1_max_sp_max',
			'mAUPRO_px',
			'mAUROC_px', 'mAP_px', 'mF1_max_px',
			'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1',
			'mIoU_max_px',
		]
		self.use_adeval = True

		# ==> data
		self.data.type = 'RDPPAD'
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
			dict(type='ToTensor'),
		]

		# ==> modal
		checkpoint_path = 'model/pretrain/wide_resnet50_racm-8234f177.pth'
		self.model_t = Namespace()
		self.model_t.name = 'timm_wide_resnet50_2'
		self.model_t.kwargs = dict(pretrained=True, checkpoint_path='',
								   strict=False, features_only=True, out_indices=[1, 2, 3])
		self.model_s = Namespace()
		self.model_s.name = 'de_wide_resnet50_2'
		self.model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False)
		self.model = Namespace()
		self.model.name = 'rdpp'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_t=self.model_t,
								 model_s=self.model_s)

		# ==> evaluator
		self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=None, max_step_aupro=100, use_adeval=self.use_adeval)

		# ==> optimizer
		self.optim.proj_opt = Namespace()
		self.optim.distill_opt = Namespace()
		self.optim.lr = self.lr
		self.optim.proj_opt.kwargs = dict(name='adam', betas=(0.5, 0.999))
		self.optim.distill_opt.kwargs = dict(name='adam', betas=(0.5, 0.999))

		# ==> trainer
		self.trainer.name = 'RDPTrainer'
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
		self.trainer.data.num_workers_per_gpu = 4

		# ==> loss
		self.loss.loss_terms = [
			dict(type='CosLoss', name='cos', avg=False, lam=1.0),
		]

		# ==> logging
		self.logging.log_terms_train = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='data_t', fmt=':>5.3f'),
			dict(name='optim_t', fmt=':>5.3f'),
			dict(name='lr', fmt=':>7.6f'),
			dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
		self.logging.log_terms_test = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
