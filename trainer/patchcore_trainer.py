import os
import copy
import glob
import shutil
import datetime
import time

import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
	from apex import amp
	from apex.parallel import DistributedDataParallel as ApexDDP
	from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
	from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp


@TRAINER.register_module
class PatchCoreTrainer():
	def __init__(self, cfg):
		self.cfg = cfg
		self.master, self.logger, self.writer = cfg.master, cfg.logger, cfg.writer
		self.local_rank, self.rank, self.world_size = cfg.local_rank, cfg.rank, cfg.world_size
		log_msg(self.logger, '==> Running Trainer: {}'.format(cfg.trainer.name))
		# =========> model <=================================
		log_msg(self.logger, '==> Using GPU: {} for Training'.format(list(range(cfg.world_size))))
		log_msg(self.logger, '==> Building model')
		self.net = get_model(cfg.model)
		self.net.to('cuda:{}'.format(cfg.local_rank))
		self.net.eval()
		log_msg(self.logger, f"==> Load checkpoint: {cfg.model.kwargs['checkpoint_path']}") if cfg.model.kwargs[
			'checkpoint_path'] else None
		# print_networks([self.net], torch.randn(self.cfg.fvcore_b, self.cfg.fvcore_c, self.cfg.size, self.cfg.size).cuda(), self.logger) if self.cfg.fvcore_is else None
		self.dist_BN = cfg.trainer.dist_BN
		if cfg.dist and cfg.trainer.sync_BN != 'none':
			self.dist_BN = ''
			log_msg(self.logger, f'==> Synchronizing BN by {cfg.trainer.sync_BN}')
			syncbn_dict = {'apex': ApexSyncBN, 'native': torch.nn.SyncBatchNorm.convert_sync_batchnorm,
						   'timm': TIMMSyncBN}
			self.net = syncbn_dict[cfg.trainer.sync_BN](self.net)
		log_msg(self.logger, '==> Creating optimizer')
		# cfg.optim.lr *= cfg.trainer.data.batch_size / 512
		# cfg.trainer.scheduler_kwargs['lr_min'] *= cfg.trainer.data.batch_size / 512
		# cfg.trainer.scheduler_kwargs['warmup_lr'] *= cfg.trainer.data.batch_size / 512
		self.optim = get_optim(cfg.optim.kwargs, self.net, lr=cfg.optim.lr)
		self.amp_autocast = get_autocast(cfg.trainer.scaler)
		self.loss_scaler = get_loss_scaler(cfg.trainer.scaler)
		self.loss_terms = get_loss_terms(cfg.loss.loss_terms, device='cuda:{}'.format(cfg.local_rank))
		if cfg.trainer.scaler == 'apex':
			self.net, self.optim = amp.initialize(self.net, self.optim, opt_level='O1')
		if cfg.dist:
			if cfg.trainer.scaler in ['none', 'native']:
				log_msg(self.logger, '==> Native DDP')
				self.net = NativeDDP(self.net, device_ids=[cfg.local_rank],
									 find_unused_parameters=cfg.trainer.find_unused_parameters)
			elif cfg.trainer.scaler in ['apex']:
				log_msg(self.logger, '==> Apex DDP')
				self.net = ApexDDP(self.net, delay_allreduce=True)
			else:
				raise 'Invalid scaler mode: {}'.format(cfg.trainer.scaler)
		# =========> dataset <=================================
		cfg.logdir_train, cfg.logdir_test = f'{cfg.logdir}/show_train', f'{cfg.logdir}/show_test'
		makedirs([cfg.logdir_train, cfg.logdir_test], exist_ok=True)
		log_msg(self.logger, "==> Loading dataset: {}".format(cfg.data.type))
		self.train_loader, self.test_loader = get_loader(cfg)
		cfg.data.train_size, cfg.data.test_size = len(self.train_loader), len(self.test_loader)
		cfg.data.train_length, cfg.data.test_length = self.train_loader.dataset.length, self.test_loader.dataset.length
		self.cls_names = self.train_loader.dataset.cls_names
		self.mixup_fn = Mixup(**cfg.trainer.mixup_kwargs) if cfg.trainer.mixup_kwargs['prob'] > 0 else None
		self.scheduler = get_scheduler(cfg, self.optim)
		self.evaluator = get_evaluator(cfg.evaluator)
		self.metrics = self.evaluator.metrics
		self.adv = cfg.adv
		if self.adv:
			self.g_reg_every, self.d_reg_every = cfg.g_reg_every, cfg.d_reg_every
		if hasattr(cfg.trainer, 'metric_recorder'):
			self.metric_recorder = cfg.trainer.metric_recorder
		else:
			cfg.trainer.metric_recorder = dict()
			for idx, cls_name in enumerate(self.cls_names):
				for metric in self.metrics:
					cfg.trainer.metric_recorder.update({f'{metric}_{cls_name}': []})
					if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
						cfg.trainer.metric_recorder.update({f'{metric}_Avg': []})
			self.metric_recorder = cfg.trainer.metric_recorder
		self.iter, self.epoch = cfg.trainer.iter, cfg.trainer.epoch
		self.iter_full, self.epoch_full = cfg.trainer.iter_full, cfg.trainer.epoch_full
		if cfg.trainer.resume_dir:
			state_dict = torch.load(cfg.model.kwargs['checkpoint_path'], map_location='cpu')
			self.optim.load_state_dict(state_dict['optimizer'])
			self.scheduler.load_state_dict(state_dict['scheduler'])
			self.loss_scaler.load_state_dict(state_dict['scaler']) if self.loss_scaler else None
			self.cfg.task_start_time = get_timepc() - state_dict['total_time']
		# self.tmp_dir = f'/dev/shm/tmp/{cfg.logdir}'
		# tmp_dir = f'/dev/shm/tmp/tmp'
		tmp_dir = f'{cfg.trainer.checkpoint}/tmp'
		tem_i = 0
		while os.path.exists(f'{tmp_dir}/{tem_i}'):
			tem_i += 1
		self.tmp_dir = f'{tmp_dir}/{tem_i}'
		log_cfg(self.cfg)

	def set_input(self, inputs):
		self.imgs = inputs['img'].cuda()
		self.imgs_mask = inputs['img_mask'].cuda()
		self.cls_name = inputs['cls_name']
		self.anomaly = inputs['anomaly']
		self.img_path = inputs['img_path']
		self.bs = self.imgs.shape[0]

	def forward(self):
		self.net.net_patchcore.fit(self.train_loader)

	def reset(self, isTrain=True):
		self.net.train(mode=isTrain)
		self.log_terms, self.progress = get_log_terms(
			able(self.cfg.logging.log_terms_train, isTrain, self.cfg.logging.log_terms_test),
			default_prefix=('Train' if isTrain else 'Test'))

	def backward_term(self):
		pass

	def optimize_parameters(self):
		pass

	def scheduler_step(self,step):
		pass

	def _finish(self):
		log_msg(self.logger, 'finish training')
		self.writer.close() if self.master else None
		metric_list = []
		for idx, cls_name in enumerate(self.cls_names):
			for metric in self.metrics:
				metric_list.append(self.metric_recorder[f'{metric}_{cls_name}'])
				if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
					metric_list.append(self.metric_recorder[f'{metric}_Avg'])
		f = open(f'{self.cfg.logdir}/metric.txt', 'w')
		msg = ''
		for i in range(len(metric_list[0])):
			for j in range(len(metric_list)):
				msg += '{:3.5f}\t'.format(metric_list[j][i])
			msg += '\n'
		f.write(msg)
		f.close()

	def train(self):
		self.reset(isTrain=True)
		self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
		train_length = self.cfg.data.train_size
		train_loader = iter(self.train_loader)
		# ---------- data ----------
		t1 = get_timepc()
		t2 = get_timepc()
		update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
		# ---------- optimization ----------
		self.forward()
		t3 = get_timepc()
		update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
		update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
		# ---------- log ----------
		self.iter = train_length
		self.epoch = self.epoch_full
		if self.master:
			if self.iter % self.cfg.logging.train_log_per == 0:
				msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length,
												 self.iter_full / train_length), self.master, None)
				log_msg(self.logger, msg)
				if self.writer:
					for k, v in self.log_terms.items():
						self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
					self.writer.flush()
		if self.iter % self.cfg.logging.train_reset_log_per == 0:
			self.reset(isTrain=True)
		# ---------- update train_loader ----------
		if self.iter % train_length == 0:
			self.epoch += 1
			if self.cfg.dist and self.dist_BN != '':
				distribute_bn(self.net, self.world_size, self.dist_BN)
			self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None
			if self.epoch >= self.cfg.trainer.test_start_epoch or self.epoch % self.cfg.trainer.test_per_epoch == 0:
				self.test()
			else:
				self.test_ghost()
			self.cfg.total_time = get_timepc() - self.cfg.task_start_time
			total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
			eta_time_str = str(
				datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
			log_msg(self.logger,
					f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
			self.save_checkpoint()
			self.reset(isTrain=True)
			self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
		self._finish()

	@torch.no_grad()
	def test(self):
		if self.master:
			if os.path.exists(self.tmp_dir):
				shutil.rmtree(self.tmp_dir)
			os.makedirs(self.tmp_dir, exist_ok=True)
		self.reset(isTrain=False)
		imgs_masks, anomaly_maps, anomaly_scores, cls_names, anomalys = [], [], [], [], []
		batch_idx = 0
		test_length = self.cfg.data.test_size
		test_loader = iter(self.test_loader)
		while batch_idx < test_length:
			# if batch_idx == 10:
			# 	break
			t1 = get_timepc()
			batch_idx += 1
			test_data = next(test_loader)
			self.set_input(test_data)
			self.scores, self.preds = self.net.net_patchcore.predict(self.imgs)
			# self.forward()
			# self.net.predict()
			# loss_cos = self.loss_terms['sum'](self.true_loss, self.fake_loss)
			# update_log_term(self.log_terms.get('sum'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 1, self.master)
			# get anomaly maps
			# anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats_t, self.feats_s, [self.imgs.shape[2], self.imgs.shape[3]], uni_am=False, amap_mode='add', gaussian_sigma=4)
			anomaly_map = self.preds
			anomaly_score = self.scores
			self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
			if self.cfg.vis:
				if self.cfg.vis_dir is not None:
					root_out = self.cfg.vis_dir
				else:
					root_out = self.writer.logdir
				vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map, self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
			imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
			anomaly_maps.append(anomaly_map)
			anomaly_scores.append(anomaly_score)
			cls_names.append(np.array(self.cls_name))
			anomalys.append(self.anomaly.cpu().numpy().astype(int))
			t2 = get_timepc()
			update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
			print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
			# ---------- log ----------
			if self.master:
				if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
					msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
					log_msg(self.logger, msg)
		# merge results
		if self.cfg.dist:
			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, anomaly_scores=anomaly_scores, cls_names=cls_names, anomalys=anomalys)
			torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
			if self.master:
				results = dict(imgs_masks=[], anomaly_maps=[],anomaly_scores=[], cls_names=[], anomalys=[])
				valid_results = False
				while not valid_results:
					results_files = glob.glob(f'{self.tmp_dir}/*.pth')
					if len(results_files) != self.cfg.world_size:
						time.sleep(1)
					else:
						idx_result = 0
						while idx_result < self.cfg.world_size:
							results_file = results_files[idx_result]
							try:
								result = torch.load(results_file)
								for k, v in result.items():
									results[k].extend(v)
								idx_result += 1
							except:
								time.sleep(1)
						valid_results = True
		else:
			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, anomaly_scores=anomaly_scores, cls_names=cls_names, anomalys=anomalys)
		if self.master:
			results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
			msg = {}
			for idx, cls_name in enumerate(self.cls_names):
				metric_results = self.evaluator.run(results, cls_name, self.logger)
				msg['Name'] = msg.get('Name', [])
				msg['Name'].append(cls_name)
				avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
				msg['Name'].append('Avg') if avg_act else None
				# msg += f'\n{cls_name:<10}'
				for metric in self.metrics:
					metric_result = metric_results[metric] * 100
					self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
					max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
					max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
					msg[metric] = msg.get(metric, [])
					msg[metric].append(metric_result)
					msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
					msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
					if avg_act:
						metric_result_avg = sum(msg[metric]) / len(msg[metric])
						self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
						max_metric = max(self.metric_recorder[f'{metric}_Avg'])
						max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
						msg[metric].append(metric_result_avg)
						msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
			msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center", stralign="center", )
			log_msg(self.logger, f'\n{msg}')

	@torch.no_grad()
	def test_ghost(self):
		for idx, cls_name in enumerate(self.cls_names):
			for metric in self.metrics:
				self.metric_recorder[f'{metric}_{cls_name}'].append(0)
				if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
					self.metric_recorder[f'{metric}_Avg'].append(0)
	def save_checkpoint(self):
		if self.master:
			checkpoint_info = {'net': trans_state_dict(self.net.state_dict(), dist=False),
							   'optimizer': self.optim.state_dict(),
							   'scheduler': self.scheduler.state_dict(),
							   'scaler': self.loss_scaler.state_dict() if self.loss_scaler else None,
							   'iter': self.iter,
							   'epoch': self.epoch,
							   'metric_recorder': self.metric_recorder,
							   'total_time': self.cfg.total_time}
			save_path = f'{self.cfg.logdir}/ckpt.pth'
			torch.save(checkpoint_info, save_path)
			torch.save(checkpoint_info['net'], f'{self.cfg.logdir}/net.pth')
			if self.epoch % self.cfg.trainer.test_per_epoch == 0:
				torch.save(checkpoint_info['net'], f'{self.cfg.logdir}/net_{self.epoch}.pth')


	def run(self):
		log_msg(self.logger,
				f'==> Starting {self.cfg.mode}ing with {self.cfg.nnodes} nodes x {self.cfg.ngpus_per_node} GPUs')
		if self.cfg.mode in ['train']:
			self.train()
		elif self.cfg.mode in ['test']:
			self.test()
		else:
			raise NotImplementedError