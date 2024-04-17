import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.plateau_lr import PlateauLRScheduler


def get_scheduler(cfg, optimizer):
	kwargs = {k: v for k, v in cfg.trainer.scheduler_kwargs.items()}
	name = kwargs.pop('name')
	use_iters = kwargs.pop('use_iters')
	if getattr(cfg.trainer, 'epoch_full', None):
		cfg.trainer.iter_full = cfg.data.train_size * cfg.trainer.epoch_full
	else:
		cfg.trainer.epoch_full = cfg.trainer.iter_full / cfg.data.train_size
	if kwargs['warmup_iters'] > -1:
		t_initial = cfg.trainer.iter_full
		warmup_t = kwargs['warmup_iters']
		decay_t = kwargs['decay_iters']
		patience_t = kwargs['patience_iters']
		if not use_iters:
			t_initial, warmup_t, decay_t, patience_t = [t / cfg.data.train_size for t in [t_initial, warmup_t, decay_t, patience_t]]
		cfg.trainer.iter_full += (kwargs['cooldown_iters'] + patience_t)
		cfg.trainer.epoch_full = cfg.trainer.iter_full / cfg.data.train_size
	elif kwargs['warmup_epochs'] > -1:
		t_initial = cfg.trainer.epoch_full
		warmup_t = kwargs['warmup_epochs']
		decay_t = kwargs['decay_epochs']
		patience_t = kwargs['patience_epochs']
		if use_iters:
			t_initial, warmup_t, decay_t, patience_t = [t * cfg.data.train_size for t in [t_initial, warmup_t, decay_t, patience_t]]
		cfg.trainer.epoch_full += (kwargs['cooldown_epochs'] + patience_t)
		cfg.trainer.iter_full = cfg.trainer.epoch_full * cfg.data.train_size
	else:
		raise Exception("invalid \'warmup_iters\' and \'warmup_epochs\'")

	if kwargs.get('lr_noise', None) is not None:
		lr_noise = kwargs.get('lr_noise')
		if isinstance(lr_noise, (list, tuple)):
			noise_range_t = [n * t_initial for n in lr_noise]
			if len(noise_range_t) == 1:
				noise_range_t = noise_range_t[0]
		else:
			noise_range_t = lr_noise * t_initial
	else:
		noise_range_t = None
		
	kwargs_common = dict(optimizer=optimizer,
						 warmup_lr_init=kwargs['warmup_lr'],
						 warmup_t=warmup_t,
						 noise_pct=kwargs.get('noise_pct', 0.67),
						 noise_std=kwargs.get('noise_std', 1.),
						 noise_seed=kwargs.get('noise_seed', 42),
						 noise_range_t=noise_range_t,
						 )
	if name == 'cosine':
		lr_scheduler = CosineLRScheduler(
			**kwargs_common,
			t_initial=t_initial,
			cycle_mul=kwargs.get('lr_cycle_mul', 1.),
			lr_min=kwargs['lr_min'],
			cycle_decay=kwargs['cycle_decay'],
			cycle_limit=kwargs.get('lr_cycle_limit', 1),
			t_in_epochs=True,
		)
	elif name == 'tanh':
		lr_scheduler = TanhLRScheduler(
			**kwargs_common,
			t_initial=t_initial,
			cycle_mul=kwargs.get('lr_cycle_mul', 1.),
			lr_min=kwargs['lr_min'],
			cycle_limit=kwargs.get('lr_cycle_limit', 1),
			t_in_epochs=True,
		)
	elif name == 'step':
		lr_scheduler = StepLRScheduler(
			**kwargs_common,
			decay_t=decay_t,
			decay_rate=kwargs['decay_rate'],
		)
	elif name == 'plateau':
		mode = 'min' if 'loss' in kwargs.get('eval_metric', '') else 'max'
		lr_scheduler = PlateauLRScheduler(
			**kwargs_common,
			decay_rate=kwargs['decay_rate'],
			patience_t=kwargs['patience_iters'],
			lr_min=kwargs['lr_min'],
			mode=mode,
			cooldown_t=0,
		)
	else:
		raise Exception(f'invalid scheduler: {name}')
	return lr_scheduler
