from argparse import Namespace


class cfg_model_patchcore(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_backbone = Namespace()
		self.model_backbone.name = 'tv_wide_resnet50_2'
		self.model_backbone.kwargs = dict(pretrained=False, checkpoint_path='model/pretrain/wide_resnet50_2-95faca4d.pth', strict=False)
		self.model = Namespace()
		self.model.name = 'patchcore'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_backbone=self.model_backbone,)
