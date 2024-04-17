from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_model_vitad(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_t = Namespace()
		self.model_t.name = 'vit_small_patch16_224_dino'
		self.model_t.kwargs = dict(pretrained=True, checkpoint_path='', strict=True,
								   img_size=256, teachers=[3, 6, 9], neck=[12])
		self.model_f = Namespace()
		self.model_f.name = 'fusion'
		self.model_f.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, dim=384, mul=1)
		self.model_s = Namespace()
		self.model_s.name = 'de_vit_small_patch16_224_dino'
		self.model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False,
								   img_size=256, students=[3, 6, 9], depth=9)
		self.model = Namespace()
		self.model.name = 'vitad'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_t=self.model_t,
								 model_f=self.model_f, model_s=self.model_s)
