import logging
import os
import pickle
from collections import OrderedDict
import torchvision
import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

import copy
from typing import List
import scipy.ndimage as ndimage

# import metrics
# from utils import plot_segmentation_images
from model import get_model
from model import MODEL

LOGGER = logging.getLogger(__name__)


class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        return features.reshape(len(features), -1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, target_size=224):
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores, features):

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features)
            features = features.permute(0, 3, 1, 2)
            if self.target_size[0] * self.target_size[1] * features.shape[0] * features.shape[1] >= 2 ** 31:
                subbatch_size = int((2 ** 31 - 1) / (self.target_size[0] * self.target_size[1] * features.shape[1]))
                interpolated_features = []
                for i_subbatch in range(int(features.shape[0] / subbatch_size + 1)):
                    subfeatures = features[i_subbatch * subbatch_size:(i_subbatch + 1) * subbatch_size]
                    subfeatures = subfeatures.unsuqeeze(0) if len(subfeatures.shape) == 3 else subfeatures
                    subfeatures = F.interpolate(
                        subfeatures, size=self.target_size, mode="bilinear", align_corners=False
                    )
                    interpolated_features.append(subfeatures)
                features = torch.cat(interpolated_features, 0)
            else:
                features = F.interpolate(
                    features, size=self.target_size, mode="bilinear", align_corners=False
                )
            features = features.cpu().numpy()

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ], [
            feature
            for feature in features
        ]


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        # self.backbone = eval(backbone.name)
        self.train_backbone = train_backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = self.backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        # self.to(self.device)

    def forward(self, images, eval=True):
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                # The backbone will throw an Exception once it reached the last
                # layer to compute features from. Computation will stop there.
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape))
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        # if self.raise_exception_to_break:
        #     raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes=1536, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):

        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x


class TBWrapper:

    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class SimpleNet(torch.nn.Module):
    def __init__(self, backbone,layers_to_extract_from=('layer2','layer3'),input_shape=(3,256,256),
                 pretrain_embed_dimension=1536,  # 1536
                 target_embed_dimension=1536,  # 1536
                 patchsize=3,  # 3
                 patchstride=1,
                 embedding_size=256,  # 256
                 meta_epochs=40,  # 40
                 aed_meta_epochs=1,
                 gan_epochs=4,  # 4
                 noise_std=0.015,
                 mix_noise=1,
                 noise_type="GAU",
                 dsc_layers=2,  # 2
                 dsc_hidden=1024,  # 1024
                 dsc_margin=.5,  # .5
                 dsc_lr=0.0002,
                 train_backbone=False,
                 auto_noise=0.0,
                 cos_lr=False,
                 lr=1e-3,
                 pre_proj=1,  # 1
                 proj_layer_type=0):
        """anomaly detection class."""
        super().__init__()
        # super(SimpleNet, self).__init__()
        pid = os.getpid()
        self.backbone = backbone
        self.layers_to_extract_from = layers_to_extract_from
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.forward_modules = torch.nn.ModuleDict({})
        # feature_dimensions = [512,1024]
        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, train_backbone)
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] =  feature_aggregator

        preprocessing = Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = RescaleSegmentor(
            target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj,
                                             proj_layer_type)

        # Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.dsc_margin = dsc_margin

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

    def forward(self, input):
        # _ = self.forward_modules.eval()
        # if self.pre_proj > 0:
        #     self.pre_projection.train()
        # self.discriminator.train()
        # self.dsc_opt.zero_grad()
        # if self.pre_proj > 0:
        #     self.proj_opt.zero_grad()
        img = input
        img = img.to(torch.float)
        if self.pre_proj > 0:
            true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
        else:
            true_feats = self._embed(img, evaluation=False)[0]

        noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).cuda() # (N, K)
        noise = torch.stack([
            torch.normal(0, self.noise_std * 1.1 ** (k), true_feats.shape)
            for k in range(self.mix_noise)], dim=1).cuda()  # (N, K, C)
        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
        fake_feats = true_feats + noise

        scores = self.discriminator(torch.cat([true_feats, fake_feats]))
        true_scores = scores[:len(true_feats)]
        fake_scores = scores[len(fake_feats):]

        th = self.dsc_margin
        true_loss = torch.clip(-true_scores + th, min=0)
        fake_loss = torch.clip(fake_scores + th, min=0)

        return true_loss.mean(), fake_loss.mean()

    def set_model_dir(self, model_dir, dataset_name):

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)  # SummaryWriter(log_dir=tb_dir)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""

        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](
            features)  # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features)  # further pooling

        return features, patch_shapes

    # def test(self, training_data, test_data, save_segmentation_images):
    #
    #     ckpt_path = os.path.join(self.ckpt_dir, "models.ckpt")
    #     if os.path.exists(ckpt_path):
    #         state_dicts = torch.load(ckpt_path, map_location=self.device)
    #         if "pretrained_enc" in state_dicts:
    #             self.feature_enc.load_state_dict(state_dicts["pretrained_enc"])
    #         if "pretrained_dec" in state_dicts:
    #             self.feature_dec.load_state_dict(state_dicts["pretrained_dec"])
    #
    #     aggregator = {"scores": [], "segmentations": [], "features": []}
    #     scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
    #     aggregator["scores"].append(scores)
    #     aggregator["segmentations"].append(segmentations)
    #     aggregator["features"].append(features)
    #
    #     scores = np.array(aggregator["scores"])
    #     min_scores = scores.min(axis=-1).reshape(-1, 1)
    #     max_scores = scores.max(axis=-1).reshape(-1, 1)
    #     scores = (scores - min_scores) / (max_scores - min_scores)
    #     scores = np.mean(scores, axis=0)
    #
    #     segmentations = np.array(aggregator["segmentations"])
    #     min_scores = (
    #         segmentations.reshape(len(segmentations), -1)
    #         .min(axis=-1)
    #         .reshape(-1, 1, 1, 1)
    #     )
    #     max_scores = (
    #         segmentations.reshape(len(segmentations), -1)
    #         .max(axis=-1)
    #         .reshape(-1, 1, 1, 1)
    #     )
    #     segmentations = (segmentations - min_scores) / (max_scores - min_scores)
    #     segmentations = np.mean(segmentations, axis=0)
    #
    #     anomaly_labels = [
    #         x[1] != "good" for x in test_data.dataset.data_to_iterate
    #     ]
    #
    #     if save_segmentation_images:
    #         self.save_segmentation_images(test_data, segmentations, scores)
    #
    #     auroc = metrics.compute_imagewise_retrieval_metrics(
    #         scores, anomaly_labels
    #     )["auroc"]
    #
    #     # Compute PRO score & PW Auroc for all images
    #     pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
    #         segmentations, masks_gt
    #     )
    #     full_pixel_auroc = pixel_scores["auroc"]
    #
    #     return auroc, full_pixel_auroc

    # def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
    #
    #     scores = np.squeeze(np.array(scores))
    #     img_min_scores = scores.min(axis=-1)
    #     img_max_scores = scores.max(axis=-1)
    #     scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
    #     # scores = np.mean(scores, axis=0)
    #
    #     auroc = metrics.compute_imagewise_retrieval_metrics(
    #         scores, labels_gt
    #     )["auroc"]
    #
    #     if len(masks_gt) > 0:
    #         segmentations = np.array(segmentations)
    #         min_scores = (
    #             segmentations.reshape(len(segmentations), -1)
    #             .min(axis=-1)
    #             .reshape(-1, 1, 1, 1)
    #         )
    #         max_scores = (
    #             segmentations.reshape(len(segmentations), -1)
    #             .max(axis=-1)
    #             .reshape(-1, 1, 1, 1)
    #         )
    #         norm_segmentations = np.zeros_like(segmentations)
    #         for min_score, max_score in zip(min_scores, max_scores):
    #             norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
    #         norm_segmentations = norm_segmentations / len(scores)
    #
    #         # Compute PRO score & PW Auroc for all images
    #         pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
    #             norm_segmentations, masks_gt)
    #         # segmentations, masks_gt
    #         full_pixel_auroc = pixel_scores["auroc"]
    #
    #         pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)),
    #                                   norm_segmentations)
    #     else:
    #         full_pixel_auroc = -1
    #         pro = -1
    #
    #     return auroc, full_pixel_auroc, pro

    # def train(self, training_data, test_data):
    #     state_dict = {}
    #     ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
    #     if os.path.exists(ckpt_path):
    #         state_dict = torch.load(ckpt_path, map_location=self.device)
    #         if 'discriminator' in state_dict:
    #             self.discriminator.load_state_dict(state_dict['discriminator'])
    #             if "pre_projection" in state_dict:
    #                 self.pre_projection.load_state_dict(state_dict["pre_projection"])
    #         else:
    #             self.load_state_dict(state_dict, strict=False)
    #
    #         self.predict(training_data, "train_")
    #         scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
    #         auroc, full_pixel_auroc, anomaly_pixel_auroc = self._evaluate(test_data, scores, segmentations, features,
    #                                                                       labels_gt, masks_gt)
    #
    #         return auroc, full_pixel_auroc, anomaly_pixel_auroc
    #
    #     def update_state_dict(d):
    #
    #         state_dict["discriminator"] = OrderedDict({
    #             k: v.detach().cpu()
    #             for k, v in self.discriminator.state_dict().items()})
    #         if self.pre_proj > 0:
    #             state_dict["pre_projection"] = OrderedDict({
    #                 k: v.detach().cpu()
    #                 for k, v in self.pre_projection.state_dict().items()})
    #
    #     best_record = None
    #     for i_mepoch in range(self.meta_epochs):
    #
    #         self._train_discriminator(training_data)
    #
    #         # torch.cuda.empty_cache()
    #         scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
    #         auroc, full_pixel_auroc, pro = self._evaluate(test_data, scores, segmentations, features, labels_gt,
    #                                                       masks_gt)
    #         self.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
    #         self.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_mepoch)
    #         self.logger.logger.add_scalar("pro", pro, i_mepoch)
    #
    #         if best_record is None:
    #             best_record = [auroc, full_pixel_auroc, pro]
    #             update_state_dict(state_dict)
    #             # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
    #         else:
    #             if auroc > best_record[0]:
    #                 best_record = [auroc, full_pixel_auroc, pro]
    #                 update_state_dict(state_dict)
    #                 # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
    #             elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
    #                 best_record[1] = full_pixel_auroc
    #                 best_record[2] = pro
    #                 update_state_dict(state_dict)
    #                 # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
    #
    #         print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
    #               f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
    #               f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----")
    #
    #     torch.save(state_dict, ckpt_path)
    #
    #     return best_record

    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        LOGGER.info(f"Training discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    i_iter += 1
                    img = data_item["image"]
                    img = img.to(torch.float)
                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
                    else:
                        true_feats = self._embed(img, evaluation=False)[0]

                    noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
                    noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise)  # (N, K)
                    noise = torch.stack([
                        torch.normal(0, self.noise_std * 1.1 ** (k), true_feats.shape)
                        for k in range(self.mix_noise)], dim=1)  # (N, K, C)
                    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                    fake_feats = true_feats + noise

                    scores = self.discriminator(torch.cat([true_feats, fake_feats]))
                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(fake_feats):]

                    th = self.dsc_margin
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                    true_loss = torch.clip(-true_scores + th, min=0)
                    fake_loss = torch.clip(fake_scores + th, min=0)

                    self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                    self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)

                    loss = true_loss.mean() + fake_loss.mean()
                    self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    self.logger.step()

                    loss.backward()
                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()
                    self.dsc_opt.step()

                    loss = loss.detach().cpu()
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())

                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)

                if self.cos_lr:
                    self.dsc_schl.step()

                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)

    # def predict(self, data, prefix=""):
    #     if isinstance(data, torch.utils.data.DataLoader):
    #         return self._predict_dataloader(data, prefix)
    #     return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []
        from sklearn.manifold import TSNE

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks, _feats = self._predict(image)
                for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)

        return scores, masks, features, labels_gt, masks_gt

    def predict(self, inputs):
        """Infer score and mask for a batch of images."""
        images = inputs['img']
        images = images.cuda()
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            features, patch_shapes = self._embed(images,
                                                 provide_patch_shapes=True,
                                                 evaluation=True)
            if self.pre_proj > 0:
                features = self.pre_projection(features)

            patch_scores = image_scores = -self.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)
            preds = np.stack(masks)
        return image_scores, preds




    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            './output',
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                                s + 2 * padding - 1 * (self.patchsize - 1) - 1
                        ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x


class SIMPLENET(torch.nn.Module):
    def __init__(self, model_backbone, layers_to_extract_from, input_size):
        super(SIMPLENET, self).__init__()
        # self.net_backbone = NetworkFeatureAggregator(model_backbone,layers_to_extract_from=('layer2','layer3'))
        # self.prep = Preprocessing(input_dims=[512,1024],output_dim=1536)
        # self.net_proj = Projection(1536)
        # self.net_disc = Discriminator(in_planes=1536)
        self.model_backbone = get_model(model_backbone)
        self.net_simplenet = SimpleNet(self.model_backbone,  layers_to_extract_from=layers_to_extract_from, input_shape=input_size)
        # self.frozen_layers = ['net_backbone']
        self.frozen_layers = ['forward_modules']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.net_simplenet.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):
        true_loss, fake_loss = self.net_simplenet(imgs)
        return true_loss, fake_loss



@MODEL.register_module
def simplenet(pretrained=False, **kwargs):
    model = SIMPLENET(**kwargs)
    return model
