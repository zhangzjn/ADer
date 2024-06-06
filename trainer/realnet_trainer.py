import os
import time
import copy
import glob
import shutil
import datetime
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

@TRAINER.register_module
class RealNetTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(RealNetTrainer, self).__init__(cfg)

        # model.afs.init_idxs(model, train_loader, distributed=True)
        # self.net.net_afs.init_idxs(self.net, self.train_loader, distributed=False)
        # self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net.net_afs.init_idxs(self.net, self.train_loader, distributed=False)
    
    def reset(self, isTrain=True):
        self.net.train(mode=isTrain)
        self.log_terms, self.progress = get_log_terms(able(self.cfg.logging.log_terms_train, isTrain, self.cfg.logging.log_terms_test), default_prefix=('Train' if isTrain else 'Test'))
        
    def scheduler_step(self, step):
        self.scheduler.step(step)
        update_log_term(self.log_terms.get('lr'), self.optim.param_groups[0]["lr"], 1, self.master)


    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        if 'gt_image' in inputs.keys():
            self.gt_imgs = inputs['gt_image'].cuda()
        else:
            self.gt_imgs = None
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.bs = self.imgs.shape[0]
    
    def forward(self):
        self.logit_mask, self.pred, self.recon_f, self.gt_f = self.net(self.imgs, self.gt_imgs)

    def backward_term(self, loss_term, optim):
        optim.zero_grad()
        if self.loss_scaler:
            self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(), create_graph=self.cfg.loss.create_graph)
        else:
            loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
            if self.cfg.loss.clip_grad is not None:
                dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
            optim.step()

    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        with self.amp_autocast():
            self.forward()
            loss_mse = self.loss_terms['pixel'](self.recon_f, self.gt_f)
            loss_seg = self.loss_terms['seg'](self.imgs_mask, self.logit_mask)
            total_loss = loss_mse + loss_seg

        self.backward_term(total_loss, self.optim)
        update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(), 1, self.master)
        update_log_term(self.log_terms.get('seg'), reduce_tensor(loss_seg, self.world_size).clone().detach().item(), 1, self.master)

    
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
        while self.epoch < self.epoch_full and self.iter < self.iter_full:
            self.scheduler_step(self.iter)
            # ---------- data ----------
            t1 = get_timepc()
            self.iter += 1
            train_data = next(train_loader)
            self.set_input(train_data)
            t2 = get_timepc()
            update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
            # ---------- optimization ----------
            self.optimize_parameters()
            t3 = get_timepc()
            update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
            update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
            # ---------- log ----------
            if self.master:
                if self.iter % self.cfg.logging.train_log_per == 0:
                    msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length, self.iter_full / train_length), self.master, None)
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
                eta_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
                log_msg(self.logger, f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
                self.reset(isTrain=True)
                self.save_checkpoint()
                self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
                train_loader = iter(self.train_loader)
        self._finish()


    @torch.no_grad()
    def test_ghost(self):
        for idx, cls_name in enumerate(self.cls_names):
            for metric in self.metrics:
                self.metric_recorder[f'{metric}_{cls_name}'].append(0)
                if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
                    self.metric_recorder[f'{metric}_Avg'].append(0)

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
            loss_mse = self.loss_terms['pixel'](self.recon_f, self.gt_f)
            loss_seg = self.loss_terms['seg'](self.imgs_mask, self.logit_mask)
            update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(), 1, self.master)
            update_log_term(self.log_terms.get('seg'), reduce_tensor(loss_seg, self.world_size).clone().detach().item(), 1, self.master)

            # get anomaly maps
            # anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats_t, self.feats_s, self.imgs.shape[-1], amap_mode='add', gaussian_sigma=4)
            anomaly_map = self.pred.cpu().numpy()
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
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
        
        # import pdb;pdb.set_trace()
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
