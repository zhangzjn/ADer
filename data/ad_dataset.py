import os
import glob
import json
import random
import pickle
from torch.utils.data import dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS
from util.data import get_img_loader
from data.utils import get_transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import torch.utils.data as data
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
import torch
import cv2
import math

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# from . import DATA
from data import DATA

# data
# ├── mvtec
#     ├── meta.json
#     ├── bottle
#         ├── train
#             └── good
#                 ├── 000.png
#         ├── test
#             ├── good
#                 ├── 000.png
#             ├── anomaly1
#                 ├── 000.png
#         └── ground_truth
#             ├── anomaly1
#                 ├── 000.png

@DATA.register_module
class DefaultAD(data.Dataset):
	def __init__(self, cfg, train=True, transform=None, target_transform=None):
		self.root = cfg.data.root
		self.train = train
		self.transform = transform
		self.target_transform = target_transform

		self.loader = get_img_loader(cfg.data.loader_type)
		self.loader_target = get_img_loader(cfg.data.loader_type_target)

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/{cfg.data.meta}', 'r'))
		name = self.root.split('/')[-1]
		if name in ['mvtec', 'coco', 'visa', 'medical']:
			meta_info = meta_info['train' if self.train else 'test']
		elif name in ['mvtec3d']:
			if self.train:
				meta_info, meta_info_val = meta_info['train'], meta_info['validation']
				for k in meta_info.keys():
					meta_info[k].extend(meta_info_val[k])
			else:
				meta_info = meta_info['test']

		self.cls_names = cfg.data.cls_names
		if not isinstance(self.cls_names, list):
			self.cls_names = [self.cls_names]
		self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names
		for cls_name in self.cls_names:
			self.data_all.extend(meta_info[cls_name])
		random.shuffle(self.data_all) if self.train else None
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly']
		img_path = f'{self.root}/{img_path}'
		img = self.loader(img_path)
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(self.loader_target(f'{self.root}/{mask_path}')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': img_path}


@DATA.register_module
class RealIAD(data.Dataset):
	def __init__(self, cfg, train=True, transform=None, target_transform=None):
		self.root = cfg.data.root
		self.train = train
		self.transform = transform
		self.target_transform = target_transform

		self.cls_names = cfg.data.cls_names
		if not isinstance(self.cls_names, list):
			self.cls_names = [self.cls_names]
		if len(self.cls_names) == 0:
			cls_names = os.listdir(self.root)
			real_cls_names = []
			for cls_name in cls_names:
				if cls_name.split('.')[0] not in real_cls_names:
					real_cls_names.append(cls_name.split('.')[0])
			real_cls_names.sort()
			self.cls_names = real_cls_names
		self.loader = get_img_loader(cfg.data.loader_type)
		self.loader_target = get_img_loader(cfg.data.loader_type_target)
		self.use_sample = cfg.data.use_sample

		meta_info = dict()
		for cls_name in self.cls_names:
			data_cls_all = []
			cls_info = json.load(open(f'{self.root}/{cls_name}.json', 'r'))
			data_cls = cls_info['train' if self.train else 'test']
			for data in data_cls:
				if data['anomaly_class'] == 'OK':
					info_img = dict(
						img_path=f"{cls_name}/{data['image_path']}",
						mask_path='',
						cls_name=cls_name,
						specie_name='',
						anomaly=0,
					)
				else:
					info_img = dict(
						img_path=f"{cls_name}/{data['image_path']}",
						mask_path=f"{cls_name}/{data['mask_path']}",
						cls_name=cls_name,
						specie_name=data['anomaly_class'],
						anomaly=1,
					)
				data_cls_all.append(info_img)
			meta_info[cls_name] = data_cls_all
		self.data_all = []
		for cls_name in self.cls_names:
			data_cls_all = meta_info[cls_name]
			if self.use_sample == True:
				sample_list = [data_cls_all[i:i + 5] for i in range(0, len(data_cls_all), 5)]
				self.data_all.extend(sample_list)
			else:
				if len(cfg.data.views) > 0:
					data_cls_all = [data for data in data_cls_all if data['img_path'].split('_')[-2] in cfg.data.views]
				self.data_all.extend(data_cls_all)
		random.shuffle(self.data_all) if self.train else None
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		if self.use_sample == True:
			sample = self.data_all[index]
			img_path, mask_path, cls_name, specie_name, anomaly = [], [], [], [], []
			img, img_mask=[], []
			for i in range(5):
				img_path.append(sample[i]['img_path'])
				mask_path.append(sample[i]['mask_path'])
				cls_name.append(sample[i]['cls_name'])
				specie_name.append(sample[i]['specie_name'])
				anomaly.append(sample[i]['anomaly'])
				img_path[i] = f'{self.root}/{img_path[i]}'
				img.append(self.loader(img_path[i]))
				if anomaly[i] == 0:
					img_mask.append(Image.fromarray(np.zeros((img[i].size[0], img[i].size[1])), mode='L'))
				else:
					img_mask.append(np.array(self.loader_target(f'{self.root}/{mask_path[i]}')) > 0)
					img_mask[i] = Image.fromarray(img_mask[i].astype(np.uint8) * 255, mode='L')
				img[i] = self.transform(img[i]) if self.transform is not None else img[i]
				img[i] = img[i].unsqueeze(0)
				img_mask[i] = self.target_transform(
					img_mask[i]) if self.target_transform is not None and img_mask[i] is not None else img_mask[i]
				img_mask[i] = [] if img_mask[i] is None else img_mask[i].unsqueeze(0)
			sample_anomaly = 0 if 1 not in anomaly else 1
			img= torch.cat(img, dim=0)
			img_mask = torch.cat(img_mask, dim=0)
			return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': img_path, 'sample_anomaly': sample_anomaly}
		else:
			data = self.data_all[index]
			img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly']
			img_path = f'{self.root}/{img_path}'
			img = self.loader(img_path)
			if anomaly == 0:
				img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
			else:
				img_mask = np.array(self.loader_target(f'{self.root}/{mask_path}')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
			img = self.transform(img) if self.transform is not None else img
			img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
			img_mask = [] if img_mask is None else img_mask
			return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': img_path}


@DATA.register_module
class CifarAD(data.Dataset):
	def __init__(self, cfg, train=True, transform=None, target_transform=None):
		self.root = cfg.data.root
		self.train = train
		self.transform = transform
		self.target_transform = target_transform

		self.cls_names = cfg.data.cls_names
		if not isinstance(self.cls_names, list):
			self.cls_names = [self.cls_names]
		# init splits
		if cfg.data.type_cifar == 'cifar10':
			self.root = f'{self.root}/cifar-10-batches-py'
			train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
			test_list = ['test_batch']
			cate_num = 10
		else:
			self.root = f'{self.root}/cifar-100-python'
			train_list = ['train']
			test_list = ['test']
			cate_num = 100
		cate_num_half = cate_num // 2
		if cfg.data.uni_setting:
			self.splits = [{'train': self.range(0, cate_num_half, 1), 'test': self.range(cate_num_half, cate_num, 1)},
						  {'train': self.range(cate_num_half, cate_num, 1), 'test': self.range(0, cate_num_half, 1)},
						  {'train': self.range(0, cate_num, 2), 'test': self.range(1, cate_num, 2)},
						  {'train': self.range(1, cate_num, 2), 'test': self.range(0, cate_num, 2)}, ]
		else:
			self.splits = []
			for idx in range(cate_num):
				cates = self.range(0, cate_num, 1)
				cates.remove(idx)
				if cfg.data.one_cls_train:
					self.splits.append({'train': [idx], 'test': cates})
				else:
					self.splits.append({'train': cates, 'test': [idx]})
		splits = self.splits[cfg.data.split_idx]
		# load data
		imgs, pseudo_cls_names, phases = [], [], []
		for idx, data_list in enumerate([train_list, test_list]):
			for file_name in data_list:
				file_path = f'{self.root}/{file_name}'
				with open(file_path, 'rb') as f:
					entry = pickle.load(f, encoding='latin1')
					imgs.append(entry['data'])
					pseudo_cls_names.extend(entry['labels'] if 'labels' in entry else entry['fine_labels'])
					phases.extend([idx] * len(entry['data']))
		self.imgs = np.vstack(imgs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
		self.pseudo_cls_names = np.array(pseudo_cls_names)
		self.phases = np.array(phases)
		# assign data
		self.data_all = []
		if self.train:
			for idx in range(len(self.imgs)):
				img, cls_name, phase = self.imgs[idx], self.pseudo_cls_names[idx], self.phases[idx]
				if cls_name in splits['train'] and phase == 0:
					self.data_all.append([img, self.cls_names[0], 0])
		else:
			if cfg.data.one_cls_train and cfg.data.type_cifar == 'cifar10':
				cls_cnt, max_cnt = [0] * 10, [111] * 10
				for cls_name in splits['train']:
					max_cnt[cls_name] = 1000
					max_cnt[(cls_name + 1) % 10] = 112
			for idx in range(len(self.imgs)):
				img, cls_name, phase = self.imgs[idx], self.pseudo_cls_names[idx], self.phases[idx]
				if cfg.data.uni_setting:
					if phase == 1:
						self.data_all.append([img, self.cls_names[0], 0 if cls_name in splits['train'] else 1])
				else:
					if cfg.data.one_cls_train and cfg.data.type_cifar == 'cifar10':
						if phase == 1:
							if cls_cnt[cls_name] < max_cnt[cls_name]:
								cls_cnt[cls_name] += 1
								self.data_all.append([img, self.cls_names[0], 0 if cls_name in splits['train'] else 1])
					else:
						if phase == 1:
							self.data_all.append([img, self.cls_names[0], 0 if cls_name in splits['train'] else 1])
						elif phase == 0:
							if cls_name in splits['test']:
								self.data_all.append([img, self.cls_names[0], 1])
		random.shuffle(self.data_all) if self.train else None
		self.length = len(self.data_all)

	@staticmethod
	def range(start, stop, step):
		return list(range(start, stop, step))

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		img, cls_name, anomaly = self.data_all[index]
		img = Image.fromarray(img)
		img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])) if anomaly == 0 else np.ones((img.size[0], img.size[1])), mode='L')
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': img_path}


@DATA.register_module
class TinyINAD(data.Dataset):
	def __init__(self, cfg, train=True, transform=None, target_transform=None):
		self.root = cfg.data.root
		self.train = train
		self.transform = transform
		self.target_transform = target_transform

		self.cls_names = cfg.data.cls_names
		if not isinstance(self.cls_names, list):
			self.cls_names = [self.cls_names]
		self.loader = get_img_loader(cfg.data.loader_type)
		# init splits
		cate_num = 200
		cate_num_half = cate_num // 2
		self.splits = [{'train': self.range(0, cate_num_half, 1), 'test': self.range(cate_num_half, cate_num, 1)},
					  {'train': self.range(cate_num_half, cate_num, 1), 'test': self.range(0, cate_num_half, 1)},
					  {'train': self.range(0, cate_num, 2), 'test': self.range(1, cate_num, 2)},
					  {'train': self.range(1, cate_num, 2), 'test': self.range(0, cate_num, 2)}, ]
		splits = self.splits[cfg.data.split_idx]
		# load data
		train_list, test_list = [], []
		pseudo_cls_names = os.listdir(f'{self.root}/train')
		pseudo_cls_names.sort()
		for i, pseudo_cls_name in enumerate(pseudo_cls_names):
			train_list.extend([[f'train/{pseudo_cls_name}/images/{file}', pseudo_cls_name] for file in os.listdir(f'{self.root}/train/{pseudo_cls_name}/images')])
			test_list.extend([[f'val/{pseudo_cls_name}/images/{file}', pseudo_cls_name] for file in os.listdir(f'{self.root}/val/{pseudo_cls_name}/images')])
		# assign data
		splits_train = [pseudo_cls_names[s] for s in splits['train']]
		splits_test = [pseudo_cls_names[s] for s in splits['test']]
		self.data_all = []
		if self.train:
			for data in train_list:
				if data[1] in splits_train:
					self.data_all.append([data[0], self.cls_names[0], 0])
		else:
			for data in test_list:
				if data[1] in splits_train:
					self.data_all.append([data[0], self.cls_names[0], 0])
				elif data[1] in splits_test:
					self.data_all.append([data[0], self.cls_names[0], 1])
				else:
					raise NotImplementedError
		random.shuffle(self.data_all) if self.train else None
		self.length = len(self.data_all)

	@staticmethod
	def range(start, stop, step):
		return list(range(start, stop, step))

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		img_path, cls_name, anomaly = self.data_all[index]
		img = self.loader(f'{self.root}/{img_path}')
		img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])) if anomaly == 0 else np.ones((img.size[0], img.size[1])), mode='L')
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': img_path}


@DATA.register_module
class Draem(data.Dataset):
	def __init__(self, cfg, train=True, transform=None, target_transform=None):
		self.root = cfg.data.root
		self.train = train
		self.transform = transform
		self.target_transform = target_transform

		self.anomaly_source_paths = sorted(glob.glob(cfg.data.anomaly_source_path + "/*/*.jpg"))
		self.resize_shape = cfg.data.resize_shape
		self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
						   iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
						   iaa.pillike.EnhanceSharpness(),
						   iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
						   iaa.Solarize(0.5, threshold=(32, 128)),
						   iaa.Posterize(),
						   iaa.Invert(),
						   iaa.pillike.Autocontrast(),
						   iaa.pillike.Equalize(),
						   iaa.Affine(rotate=(-45, 45))
						   ]
		self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

		self.cls_names = cfg.data.cls_names
		if not isinstance(self.cls_names, list):
			self.cls_names = [self.cls_names]
		self.loader = get_img_loader(cfg.data.loader_type)
		self.loader_target = get_img_loader(cfg.data.loader_type_target)

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/{cfg.data.meta}', 'r'))
		name = self.root.split('/')[-1]
		if name in ['mvtec', 'coco', 'visa']:
			meta_info = meta_info['train' if self.train else 'test']
		elif name in ['mvtec3d']:
			if self.train:
				meta_info, meta_info_val = meta_info['train'], meta_info['validation']
				for k in meta_info.keys():
					meta_info[k].extend(meta_info_val[k])
			else:
				meta_info = meta_info['test']

		self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names
		for cls_name in self.cls_names:
			self.data_all.extend(meta_info[cls_name])
		random.shuffle(self.data_all) if self.train else None
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def randAugmenter(self):
		aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
		aug = iaa.Sequential([
			self.augmenters[aug_ind[0]],
			self.augmenters[aug_ind[1]],
			self.augmenters[aug_ind[2]]])
		return aug

	def lerp_np(self, x, y, w):
		fin_out = (y - x) * w + x
		return fin_out

	def rand_perlin_2d_np(self, shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
		delta = (res[0] / shape[0], res[1] / shape[1])
		d = (shape[0] // res[0], shape[1] // res[1])
		grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

		angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
		gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
		tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

		tile_grads = lambda slice1, slice2: np.repeat(
			np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)
		dot = lambda grad, shift: (
				np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
						 axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

		n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
		n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
		n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
		n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
		t = fade(grid[:shape[0], :shape[1]])
		return math.sqrt(2) * self.lerp_np(self.lerp_np(n00, n10, t[..., 0]), self.lerp_np(n01, n11, t[..., 0]), t[..., 1])
	def augment_image(self, image, anomaly_source_path):
		aug = self.randAugmenter()
		perlin_scale = 6
		min_perlin_scale = 0
		anomaly_source_img = cv2.imread(anomaly_source_path)
		anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

		anomaly_img_augmented = aug(image=anomaly_source_img)
		perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
		perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

		perlin_noise = self.rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
		perlin_noise = self.rot(image=perlin_noise)
		threshold = 0.5
		perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
		perlin_thr = np.expand_dims(perlin_thr, axis=2)

		img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

		beta = torch.rand(1).numpy()[0] * 0.8

		augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
			perlin_thr)

		no_anomaly = torch.rand(1).numpy()[0]
		if no_anomaly > 0.5:
			image = image.astype(np.float32)
			return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
		else:
			augmented_image = augmented_image.astype(np.float32)
			msk = (perlin_thr).astype(np.float32)
			augmented_image = msk * augmented_image + (1 - msk) * image
			has_anomaly = 1.0
			if np.sum(msk) == 0:
				has_anomaly = 0.0
			return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

	def transform_image(self, image_path, anomaly_source_path):
		image = cv2.imread(image_path)
		image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

		do_aug_orig = torch.rand(1).numpy()[0] > 0.7
		if do_aug_orig:
			image = self.rot(image=image)

		image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
		augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
		augmented_image = np.transpose(augmented_image, (2, 0, 1))
		image = np.transpose(image, (2, 0, 1))
		anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
		return image, augmented_image, anomaly_mask, has_anomaly

	def __getitem__(self, index):
		if self.train:
			idx = torch.randint(0, len(self.data_all), (1,)).item()
			anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
			data = self.data_all[idx]
			img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
																  data['specie_name'], data['anomaly']
			image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(os.path.join(self.root, img_path),
																					 self.anomaly_source_paths[
																						 anomaly_source_idx])
			sample = {'img': image, "img_mask": anomaly_mask, 'cls_name': cls_name,
					  'augmented_image': augmented_image, 'anomaly': has_anomaly}
			return sample
		else:
			data = self.data_all[index]
			img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
																  data['specie_name'], data['anomaly']

			image_path = os.path.join(self.root, img_path)
			image = cv2.imread(image_path, cv2.IMREAD_COLOR)
			if mask_path != '':
				mask_path = os.path.join(self.root, mask_path)
				# mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
				mask = np.array(self.loader_target(mask_path)) > 0
				mask = mask.astype(np.uint8) * 255
			else:
				mask = np.zeros((image.shape[0], image.shape[1]))
			image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0])) / 255.0
			mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0])) / 255.0
			image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
			mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)
			image = np.transpose(image, (2, 0, 1))
			mask = np.transpose(mask, (2, 0, 1))
			return {'img': image, 'img_mask': mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': img_path}


@DATA.register_module
class DeSTSeg(data.Dataset):
	def __init__(self, cfg, train=True, transform=None, target_transform=None):
		self.root = cfg.data.root
		self.train = train
		self.transform = transform
		self.target_transform = target_transform

		self.anomaly_source_paths = sorted(glob.glob(cfg.data.anomaly_source_path + "/*/*.jpg"))
		self.resize_shape = cfg.data.resize_shape

		self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

		self.cls_names = cfg.data.cls_names
		if not isinstance(self.cls_names, list):
			self.cls_names = [self.cls_names]
		self.loader = get_img_loader(cfg.data.loader_type)
		self.loader_target = get_img_loader(cfg.data.loader_type_target)

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/{cfg.data.meta}', 'r'))
		name = self.root.split('/')[-1]
		if name in ['mvtec', 'coco', 'visa', 'medical']:
			meta_info = meta_info['train' if self.train else 'test']
		elif name in ['mvtec3d']:
			if self.train:
				meta_info, meta_info_val = meta_info['train'], meta_info['validation']
				for k in meta_info.keys():
					meta_info[k].extend(meta_info_val[k])
			else:
				meta_info = meta_info['test']

		self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names
		for cls_name in self.cls_names:
			self.data_all.extend(meta_info[cls_name])
		random.shuffle(self.data_all) if self.train else None
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def lerp_np(self, x, y, w):
		fin_out = (y - x) * w + x
		return fin_out

	def rand_perlin_2d_np(self, shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
		delta = (res[0] / shape[0], res[1] / shape[1])
		d = (shape[0] // res[0], shape[1] // res[1])
		grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

		angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
		gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
		tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

		tile_grads = lambda slice1, slice2: np.repeat(
			np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)
		dot = lambda grad, shift: (
				np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
						 axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

		n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
		n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
		n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
		n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
		t = fade(grid[:shape[0], :shape[1]])
		return math.sqrt(2) * self.lerp_np(self.lerp_np(n00, n10, t[..., 0]), self.lerp_np(n01, n11, t[..., 0]), t[..., 1])
	def augment_image(self, image, anomaly_source_path):
		perlin_scale = 6
		min_perlin_scale = 0
		anomaly_source_img = cv2.imread(anomaly_source_path)
		anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

		anomaly_img_augmented = anomaly_source_img
		perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
		perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

		perlin_noise = self.rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
		perlin_noise = self.rot(image=perlin_noise)
		threshold = 0.5
		perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
		perlin_thr = np.expand_dims(perlin_thr, axis=2)

		img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

		beta = torch.rand(1).numpy()[0] * 0.8

		augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
			perlin_thr)

		no_anomaly = torch.rand(1).numpy()[0]
		if no_anomaly > 1.0:
			image = image.astype(np.float32)
			return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
		else:
			augmented_image = augmented_image.astype(np.float32)
			msk = (perlin_thr).astype(np.float32)
			# augmented_image = msk * augmented_image + (1 - msk) * image
			has_anomaly = 1.0
			if np.sum(msk) == 0:
				has_anomaly = 0.0
			return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

	def transform_image(self, image_path, anomaly_source_path):
		image = cv2.imread(image_path)
		image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

		image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
		augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
		# augmented_image = np.transpose(augmented_image, (2, 0, 1))
		# image = np.transpose(image, (2, 0, 1))
		anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
		return image, augmented_image, anomaly_mask, has_anomaly

	def __getitem__(self, index):
		if self.train:
			idx = torch.randint(0, len(self.data_all), (1,)).item()
			anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
			data = self.data_all[idx]
			img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
																  data['specie_name'], data['anomaly']
			image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(os.path.join(self.root, img_path),
																					 self.anomaly_source_paths[
																						 anomaly_source_idx])
			image = self.transform(image) if self.transform is not None else image
			augmented_image = self.transform(augmented_image) if self.transform is not None else augmented_image
			sample = {'img': image, "img_mask": anomaly_mask, 'cls_name': cls_name,
					  'augmented_image': augmented_image, 'anomaly': has_anomaly}
			return sample
		else:
			data = self.data_all[index]
			img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
																  data['specie_name'], data['anomaly']

			image_path = os.path.join(self.root, img_path)
			image = cv2.imread(image_path, cv2.IMREAD_COLOR)
			if mask_path != '':
				mask_path = os.path.join(self.root, mask_path)
				# mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
				mask = np.array(self.loader_target(mask_path)) > 0
				mask = mask.astype(np.uint8) * 255
			else:
				mask = np.zeros((image.shape[0], image.shape[1]))
			image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0])) / 255.0
			mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0])) / 255.0
			image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
			mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)
			# image = np.transpose(image, (2, 0, 1))
			mask = np.transpose(mask, (2, 0, 1))
			image = self.transform(image) if self.transform is not None else image
			return {'img': image, 'img_mask': mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': img_path}


if __name__ == '__main__':
	from argparse import Namespace as _Namespace

	cfg = _Namespace()
	data = _Namespace()
	data.sampler = 'naive'
	# ========== MVTec ==========
	# data.root = 'data/mvtec'
	# data.meta = 'meta.json'
	# # data.cls_names = ['bottle']
	# data.cls_names = []
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# data.root = 'data/mvtec3d'
	# data.meta = 'meta.json'
	# # data.cls_names = ['bagel']
	# data.cls_names = []
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# data.root = 'data/coco'
	# data.meta = 'meta_20_0.json'
	# data.cls_names = ['coco']
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# data.root = 'data/visa'
	# data.meta = 'meta.json'
	# # data.cls_names = ['candle']
	# data.cls_names = []
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# ========== Cifar ==========
	# data.type = 'DefaultAD'
	# data.root = 'data/cifar'
	# data.type_cifar = 'cifar10'
	# data.cls_names = ['cifar']
	# data.uni_setting = True
	# data.one_cls_train = True
	# data.split_idx = 0
	# data_fun = CifarAD

	# ========== Tiny ImageNet ==========
	# data.root = 'data/tiny-imagenet-200'
	# data.cls_names = ['tin']
	# data.loader_type = 'pil'
	# data.split_idx = 0
	# data_fun = TinyINAD

	# ========== Real-IAD ==========
	data.root = 'data/realiad/explicit_full'
	# data.cls_names = ['audiojack']
	data.cls_names = []
	data.loader_type = 'pil'
	data.loader_type_target = 'pil_L'
	data.views = ['C1', 'C2']
	# data.views = []
	data.use_sample = True
	data_fun = RealIAD


	cfg.data = data
	data_debug = data_fun(cfg, train=True)
	# data_debug = data_fun(cfg, train=False)
	for idx, data in enumerate(data_debug):
		break
		if idx > 1000:
			break
		print()

