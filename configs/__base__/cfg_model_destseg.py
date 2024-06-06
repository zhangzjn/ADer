from argparse import Namespace


class cfg_model_destseg(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_t = Namespace()
		self.model_t.name = 'timm_resnet18'
		self.model_t.kwargs = dict(pretrained=False, checkpoint_path='model/pretrain/resnet18-5c106cde.pth', strict=False, features_only=True, out_indices=[1, 2, 3])
		self.model_s = Namespace()
		self.model_s.name = 'timm_resnet18'
		self.model_s.kwargs = dict(pretrained=False, checkpoint_path=None, strict=False, features_only=True, out_indices=[1, 2, 3, 4])
		self.model = Namespace()
		self.model.name = 'destseg'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_t=self.model_t, model_s=self.model_s)
