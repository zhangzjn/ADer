from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_model_hrad(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_t = Namespace()
		self.model_t.name = 'timm_tf_efficientnet_b4'
		self.model_t.kwargs = dict(pretrained=False,
									 checkpoint_path='model/pretrain/tf_efficientnet_b4_aa-818f208c.pth', strict=False,
									 hf=None, features_only=True, out_indices=[0, 1, 2, 3])
		self.model_s = Namespace()
		self.model_s.name = 'de_wide_resnet50_2'
		self.model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False)
		self.model = Namespace()
		self.model.name = 'rd'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_t=self.model_t, model_s=self.model_s)
