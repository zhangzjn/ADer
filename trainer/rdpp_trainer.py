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
from model.rdpp import Revisit_RDLoss
from util.vis import vis_rgb_gt_amp

@TRAINER.register_module
class RDPTrainer(BaseTrainer):
	def __init__(self, cfg):
		super(RDPTrainer, self).__init__(cfg)
		self.optim.proj_opt = get_optim(cfg.optim.proj_opt.kwargs, self.net.proj_layer,	lr=cfg.optim.lr)
		proj_layer = self.net.proj_layer
		self.net.proj_layer = None
		self.optim.distill_opt = get_optim(cfg.optim.distill_opt.kwargs, self.net, lr=cfg.optim.lr * 5)
		self.net.proj_layer = proj_layer
		# self.proj_loss = Revisit_RDLoss()
		# self.optim.proj_opt = torch.optim.Adam(list(self.net.proj_layer.parameters()), lr=cfg.optim.lr, betas=(0.5, 0.999))
		# self.optim.distill_opt = torch.optim.Adam(list(self.net.net_s.parameters()) + list(self.net.mff_oce.parameters()), lr=cfg.optim.lr * 5,
		# 									 betas=(0.5, 0.999))

	def set_input(self, inputs):
		self.imgs = inputs['img'].cuda()
		self.imgs_mask = inputs['img_mask'].cuda()
		self.img_noise = inputs['img_noise'].cuda()
		self.cls_name = inputs['cls_name']
		self.anomaly = inputs['anomaly']
		self.img_path = inputs['img_path']
		self.bs = self.imgs.shape[0]

	def forward(self):
		self.feats_t, self.feats_s, self.L_proj = self.net(self.imgs, self.img_noise)

	def backward_term(self, loss_term, optim):
		optim.proj_opt.zero_grad()
		optim.distill_opt.zero_grad()
		if self.loss_scaler:
			self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(),
							 create_graph=self.cfg.loss.create_graph)
		else:
			loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
			if self.cfg.loss.clip_grad is not None:
				dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
			if (self.iter + 1) % 2 == 0:
				optim.proj_opt.step()
				optim.distill_opt.step()


	def optimize_parameters(self):
		if self.mixup_fn is not None:
			self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
		with self.amp_autocast():
			self.forward()
			loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s) + 0.2 * self.L_proj
		self.backward_term(loss_cos, self.optim)
		update_log_term(self.log_terms.get('cos'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 1, self.master)

	@torch.no_grad()
	def test(self):
		if self.master:
			if os.path.exists(self.tmp_dir):
				shutil.rmtree(self.tmp_dir)
			os.makedirs(self.tmp_dir, exist_ok=True)
		self.reset(isTrain=False)
		imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
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
			self.forward()
			loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
			update_log_term(self.log_terms.get('cos'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 1, self.master)
			# get anomaly maps
			anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats_t, self.feats_s, [self.imgs.shape[2], self.imgs.shape[3]], uni_am=False, amap_mode='add', gaussian_sigma=4)
			self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
			if self.cfg.vis:
				if self.cfg.vis_dir is not None:
					root_out = self.cfg.vis_dir
				else:
					root_out = self.writer.logdir
				vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map, self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
			imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
			anomaly_maps.append(anomaly_map)
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
			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
			torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
			if self.master:
				results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
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
			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
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
