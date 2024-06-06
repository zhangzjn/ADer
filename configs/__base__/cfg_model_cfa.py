from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_model_cfa(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_backbone = Namespace()
		self.model_backbone.kwargs = dict(pretrained=True)
		# self.model_backbone.name = 'wide_resnet50_2'
		# self.model_backbone.kwargs = dict(pretrained=True,
		# 								  checkpoint_path='model/pretrain/tf_efficientnet_b4_aa-818f208c.pth',
		# 								  strict=False,
		# 								  hf=None, features_only=True, out_indices=[0, 1, 2, 3])
		
		self.model_dsvdd = Namespace()
		self.model_dsvdd.data_loader = None

		self.model_dsvdd.kwargs = dict(
			gamma_c = 1,
			gamma_d = 1,
			device = 'cuda',
			cnn = 'wrn50_2'
		)

		self.model = Namespace()
		self.model.name = 'cfa'
		self.model.kwargs = dict(pretrained=True,
								 checkpoint_path='', strict=True, 
								model_backbone=self.model_backbone,
								model_dsvdd=self.model_dsvdd)