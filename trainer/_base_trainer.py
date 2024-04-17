import os
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

# from . import TRAINER


# @TRAINER.register_module
class BaseTrainer():
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
        print_networks([self.net], torch.randn(self.cfg.fvcore_b, self.cfg.fvcore_c, self.cfg.size, self.cfg.size).cuda(), self.logger) if self.cfg.fvcore_is else None
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

    def reset(self, isTrain=True):
        self.net.train(mode=isTrain)
        self.log_terms, self.progress = get_log_terms(
            able(self.cfg.logging.log_terms_train, isTrain, self.cfg.logging.log_terms_test),
            default_prefix=('Train' if isTrain else 'Test'))

    def scheduler_step(self, step):
        self.scheduler.step(step)
        update_log_term(self.log_terms.get('lr'), self.optim.param_groups[0]["lr"], 1, self.master)

    def set_input(self, inputs):
        pass

    def forward(self):
        pass

    def backward_term(self, loss_term, optim):
        optim.zero_grad()
        if self.loss_scaler:
            self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(),
                             create_graph=self.cfg.loss.create_graph)
        else:
            loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
            if self.cfg.loss.clip_grad is not None:
                dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
            optim.step()

    def optimize_parameters(self):
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
        pass

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

