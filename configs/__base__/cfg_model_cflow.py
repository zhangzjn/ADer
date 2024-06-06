from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_model_cflow(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_backbone = Namespace()
		self.model_backbone.name = 'timm_wide_resnet_50_2'
		self.model_backbone.condition_vec = 128
		self.model_backbone.kwargs = dict(
											pretrained=False,
											checkpoint_path='../pretrain_models/wide_resnet50_racm-8234f177.pth',
											strict=False,
											hf=None, features_only=True, out_indices=[0, 1, 2, 3])


		self.model = Namespace()
		self.model.name = 'cflow'
		self.model.pool_layers = 3
		self.model.kwargs = dict(
			pretrained=False, checkpoint_path='', strict=True,
			model_backbone=self.model_backbone, L=self.model.pool_layers, N=256)
