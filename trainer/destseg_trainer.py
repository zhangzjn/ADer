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
import torch.nn.functional as F
from util.vis import vis_rgb_gt_amp

@TRAINER.register_module
class DeSTSegTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(DeSTSegTrainer, self).__init__(cfg)
        self.optim.de_st = get_optim(cfg.optim.de_st.kwargs, self.net.destseg.student_net,
                                       lr=cfg.lr*40)
        self.optim.seg_optimizer = torch.optim.SGD(
            [{"params": self.net.destseg.segmentation_net.res.parameters(), "lr": cfg.lr*10},
                {"params": self.net.destseg.segmentation_net.head.parameters(), "lr": cfg.lr},],
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=False,
        )
        print('optim finish!')

    def set_input(self, inputs):
        self.ori_imgs = inputs['img'].cuda()
        self.imgs = inputs.get('augmented_image', None)
        if self.imgs is None:
            self.imgs = inputs['img'].cuda()
        else:
            self.imgs = self.imgs.cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']

    def forward(self):
        self.output_segmentation, self.output_de_st, self.output_de_st_list, self.new_mask = self.net(self.imgs, self.ori_imgs, self.imgs_mask)

    def backward_term(self, loss_term, optim, params):
        optim.zero_grad()
        if self.loss_scaler:
            self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=params,
                             create_graph=self.cfg.loss.create_graph)
        else:
            loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
            if self.cfg.loss.clip_grad is not None:
                dispatch_clip_grad(params, value=self.cfg.loss.clip_grad)
            optim.step()

    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        if self.epoch < (self.epoch_full / 5):
            self.net.destseg.student_net.train()
            self.net.destseg.segmentation_net.eval()
        else:
            self.net.destseg.student_net.eval()
            self.net.destseg.segmentation_net.train()
        with self.amp_autocast():
            self.forward()
            loss_cos = self.loss_terms['csum'](self.output_de_st_list)
            loss_l1 = self.loss_terms['l1'](self.output_segmentation, self.new_mask)
            loss_focal = self.loss_terms['ffocal'](self.output_segmentation, self.new_mask)
        if self.epoch < (self.epoch_full / 5):
            optims = self.optim.de_st
            params = self.net.destseg.student_net.parameters()
            loss = loss_cos
        else:
            optims = self.optim.seg_optimizer
            params = list(self.net.destseg.segmentation_net.res.parameters()) + list(self.net.destseg.segmentation_net.head.parameters())
            loss = loss_l1 + loss_focal
        self.backward_term(loss, optims, params)
        update_log_term(self.log_terms.get('total'), reduce_tensor(loss, self.world_size).clone().detach().item(), 1,
                        self.master)


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
            self.output_segmentation, self.output_de_st, self.output_de_st_list, self.new_mask = self.net(self.ori_imgs)
            # get anomaly maps
            output_segmentation = F.interpolate(
                self.output_segmentation,
                size=self.imgs_mask.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            anomaly_map = output_segmentation[: ,0 ,: ,:].detach().cpu().numpy()
            # get anomaly scores
            output_segmentation_sample, _ = torch.sort(
                self.output_segmentation.view(self.output_segmentation.size(0), -1),
                dim=1,
                descending=True,
            )
            output_segmentation_sample = torch.mean(
                output_segmentation_sample[:, : 100], dim=1
            )
            anomaly_scores.append(output_segmentation_sample.detach().cpu().numpy())
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            if self.cfg.vis:
                if self.cfg.vis_dir is not None:
                    root_out = self.cfg.vis_dir
                else:
                    root_out = self.writer.logdir
                vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map,
                               self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
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
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, anomaly_scores=anomaly_scores,
                           cls_names=cls_names, anomalys=anomalys)
            torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
            if self.master:
                results = dict(imgs_masks=[], anomaly_maps=[], anomaly_scores=[], cls_names=[], anomalys=[])
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
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, anomaly_scores=anomaly_scores,
                           cls_names=cls_names, anomalys=anomalys)
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
            msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center",
                                    stralign="center", )
            log_msg(self.logger, f'\n{msg}')
