from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_model_uniad(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_backbone = Namespace()
		self.model_backbone.name = 'timm_tf_efficientnet_b4'
		self.model_backbone.kwargs = dict(pretrained=False,
									 checkpoint_path='model/pretrain/tf_efficientnet_b4_aa-818f208c.pth', strict=False,
									 hf=None, features_only=True, out_indices=[0, 1, 2, 3])
		inplanes = [24, 32, 56, 160]
		self.model_decoder = dict(
			inplanes=inplanes,
			outplanes=[sum(inplanes)],
			instrides=[16],
			feature_size=[14, 14],
			neighbor_size=[7, 7],
		)
		self.model = Namespace()
		self.model.name = 'uniad'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_backbone=self.model_backbone, model_decoder=self.model_decoder)
