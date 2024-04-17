import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import accimage
import torchvision
import torchvision.transforms as transforms
from skimage import color


def pil_loader(path):
	return Image.open(path).convert('RGB')

def pil_loader_1(path):
	return Image.open(path).convert('1')

def pil_loader_L(path):
	return Image.open(path).convert('L')

def accimage_loader(path):
	return accimage.Image(path)
	
def get_img_loader(loader_type):
	if loader_type == 'pil':
		return pil_loader
	elif loader_type == 'pil_1':
		return pil_loader_1
	elif loader_type == 'pil_L':
		return pil_loader_L
	elif loader_type == 'accimage':
		torchvision.set_image_backend('accimage')
		return accimage_loader
	else:
		raise ValueError('invalid image loader type: {}'.format(loader_type))

# ---------- for visualization ----------
def rgb_vis(img, mean, std):
	"""
	Args:
		img     : tensor, rgb[-1.0, 1.0], [3, H, W]
	Returns:
		img     : numpy, rgb[0, 255]
	"""
	img = img.data.cpu().numpy()
	for i in range(3):
		img[i, :, :] = img[i, :, :] * std[i] + mean[i]
	img = np.transpose(img, (1, 2, 0)) * 255
	img = np.clip(img, 0, 255)
	img = img.astype(np.uint8)
	return img


def rgbs_vis(imgs, mean, std):
	"""
	Args:
		img     : tensor, rgb[-1.0, 1.0], [B, 3, H, W]
	Returns:
		img     : tensor, rgb[0.0, 1.0]
	"""
	bs = imgs.shape[0]
	imgs_tensor = []
	for i in range(bs):
		img = rgb_vis(imgs[i], mean, std)
		img = Image.fromarray(img)
		img = transforms.ToTensor()(img)
		imgs_tensor.append(img)
	imgs_tensor = torch.stack(imgs_tensor, dim=0)
	return imgs_tensor

# ---------- for multi-scale training ----------
def make_divisible(v, divisor=8, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


def get_scales(n_scale, base_h, base_w, min_h, max_h, min_w, max_w, check_scale_div_factor=32):
	hs = list(np.linspace(min_h, max_h, n_scale))
	if base_h not in hs:
		hs.append(base_h)
	ws = list(np.linspace(min_w, max_w, n_scale))
	if base_w not in ws:
		ws.append(base_w)
	scales = set()
	for h, w in zip(hs, ws):
		h = make_divisible(h, check_scale_div_factor)
		w = make_divisible(w, check_scale_div_factor)
		scales.add((h, w))
	scales = list(scales)
	return scales
