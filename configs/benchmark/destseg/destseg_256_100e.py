from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
from configs.__base__ import *


class cfg(cfg_common, cfg_dataset_default, cfg_model_destseg):

	def __init__(self):
		cfg_common.__init__(self)
		cfg_dataset_default.__init__(self)
		cfg_model_destseg.__init__(self)

		self.seed = 42
		self.size = 256
		self.epoch_full = 100
		self.warmup_epochs = 0
		self.test_start_epoch = self.epoch_full
		self.test_per_epoch = self.epoch_full // 10
		self.batch_train = 32
		self.batch_test_per = 32
		self.lr = 0.01 * self.batch_train / 32
		self.metrics = [
			'mAUROC_sp_max','AUROC_sp', 'mAUROC_px', 'mAUPRO_px',
			'mAP_sp_max', 'mAP_px',
			'mF1_max_sp_max',
			'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1',
			'mF1_max_px', 'mIoU_max_px',
		]
		self.use_adeval = True

		# ==> data
		self.data.type = 'DeSTSeg'
		self.data.root = 'data/mvtec'
		self.data.meta = 'meta.json'
		self.data.cls_names = []

		self.data.anomaly_source_path = 'data/dtd/images/'
		self.data.resize_shape = [self.size, self.size]

		self.data.use_sample = False
		self.data.views = []  # ['C1', 'C2', 'C3', 'C4', 'C5']

		self.data.train_transforms = [
			dict(type='ToTensor'),
			dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		]
		self.data.test_transforms = self.data.train_transforms


		# ==> modal
		checkpoint_path = 'model/pretrain/resnet18-5c106cde.pth'
		self.model_t = Namespace()
		self.model_t.name = 'timm_resnet18'
		self.model_t.kwargs = dict(pretrained=True, checkpoint_path='', strict=False, features_only=True, out_indices=[1, 2, 3])
		self.model_s = Namespace()
		self.model_s.name = 'timm_resnet18'
		self.model_s.kwargs = dict(pretrained=False, checkpoint_path=None, strict=False, features_only=True, out_indices=[1, 2, 3, 4])
		self.model = Namespace()
		self.model.name = 'destseg'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_t=self.model_t, model_s=self.model_s)
		# ==> evaluator
		self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=None, max_step_aupro=100, use_adeval=self.use_adeval)

		# ==> optimizer
		self.optim.de_st = Namespace()
		self.optim.de_st.kwargs = dict(name='sgd', momentum=0.9, weight_decay=1e-4, nesterov=False,)

		# ==> trainer
		self.trainer.name = 'DeSTSegTrainer'
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
			dict(type='L1Loss', name='l1', lam=1.0),
			dict(type='CSUMLoss', name='csum', lam=1.0),
			dict(type='FFocalLoss', name='ffocal', gamma=4, lam=1.0),
		]

		# ==> logging
		self.logging.log_terms_train = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='data_t', fmt=':>5.3f'),
			dict(name='optim_t', fmt=':>5.3f'),
			dict(name='lr', fmt=':>7.6f'),
			dict(name='total', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
		self.logging.log_terms_test = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='total', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
