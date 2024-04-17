from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_dataset_default(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.data = Namespace()
		self.data.sampler = 'naive'

		self.data.loader_type = 'pil'
		self.data.loader_type_target = 'pil_L'

		# ---------- MUAD ----------
		self.data.type = 'DefaultAD'
		self.data.root = 'data/mvtec'  # ['mvtec', 'visa', 'mvtec3d', 'medical']
		self.data.meta = 'meta.json'
		self.data.cls_names = []

		# --> for COCO-AD
		# self.data.type = 'DefaultAD'
		# self.data.root = 'data/coco'
		# self.data.meta = 'meta_20_0.json'  # ['meta_20_0', 'meta_20_1', 'meta_20_2', 'meta_20_3']
		# self.data.cls_names = ['coco']

		# --> for RealIAD
		# self.data.type = 'RealIAD'
		# self.data.root = 'data/realiad'
		# self.data.use_sample = False
		# self.data.views = []  # ['C1', 'C2', 'C3', 'C4', 'C5']
		# self.data.cls_names = []

		# ---------- SUAD ----------
		# self.data.type = 'DefaultAD'
		# self.data.root = 'data/mvtec'  # ['mvtec', 'visa', 'mvtec3d', 'medical']
		# self.data.meta = 'meta.json'
		# self.data.cls_names = ['carpet', 'grid', 'leather']
		mvtec = [
			'carpet', 'grid', 'leather', 'tile', 'wood',
			'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
			'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
		]
		visa = [
			'pcb1', 'pcb2', 'pcb3', 'pcb4',
			'macaroni1', 'macaroni2', 'capsules', 'candle',
			'cashew', 'chewinggum', 'fryum', 'pipe_fryum',
		]
		mvtec3d = [
			'bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
			'foam', 'peach', 'potato', 'rope', 'tire',
		]
		medical = [
			'brain', 'liver', 'retinal',
		]

		# --> for RealIAD
		# self.data.type = 'RealIAD'
		# self.data.root = 'data/realiad'
		# self.data.use_sample = False
		# self.data.views = []  # ['C1', 'C2', 'C3', 'C4', 'C5']
		# self.data.cls_names = ['audiojack', 'bottle_cap', 'button_battery']
		realiad = [
			'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser',
			'fire_hood', 'mint', 'mounts', 'pcb', 'phone_battery',
			'plastic_nut', 'plastic_plug', 'porcelain_doll', 'regulator', 'rolled_strip_base',
			'sim_card_set', 'switch', 'tape', 'terminalblock', 'toothbrush',
			'toy', 'toy_brick', 'transistor1', 'u_block', 'usb',
			'usb_adaptor', 'vcpill', 'wooden_beads', 'woodstick', 'zipper',
		]

		# ---------- for one-class classification ----------
		# self.data.type = 'CifarAD'
		# self.data.root = 'data/cifar'
		# self.data.type_cifar = 'cifar10'
		# self.data.cls_names = ['cifar']
		# self.data.uni_setting = True
		# self.data.one_cls_train = True
		# self.data.split_idx = 0

		self.data.train_transforms = [
			dict(type='Resize', size=(256, 256), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(256, 256)),
			dict(type='ToTensor'),
			dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		]
		self.data.test_transforms = self.data.train_transforms
		self.data.target_transforms = [
			dict(type='Resize', size=(256, 256), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(256, 256)),
			dict(type='ToTensor'),
		]
		# self.data.train_transforms = [
		# 	dict(type='RandomResizedCrop', size=(self.size, self.size), scale=(0.25, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=F.InterpolationMode.BILINEAR),
		# 	dict(type='CenterCrop', size=(self.size, self.size)),
		# 	dict(type='RandomHorizontalFlip', p=0.5),
		# 	dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0),
		# 	dict(type='RandomRotation', degrees=(-17, 17), interpolation=F.InterpolationMode.BILINEAR, expand=False),
		# 	dict(type='CenterCrop', size=(self.size, self.size)),
		# 	dict(type='ToTensor'),
		# 	dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		# ]
		# self.data.test_transforms = self.data.train_transforms
		# self.data.target_transforms = [
		# 	dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
		# 	dict(type='CenterCrop', size=(self.size, self.size)),
		# 	dict(type='ToTensor'),
		# ]
