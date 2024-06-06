from argparse import Namespace


class cfg_model_pyramidflow(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_backbone = Namespace()
		self.model_backbone.name = 'tv_resnet18'
		self.model_backbone.kwargs = dict(pretrained=False, checkpoint_path='model/pretrain/resnet18-5c106cde.pth', strict=False)
		self.model = Namespace()
		self.model.name = 'pyramidflow'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_backbone=self.model_backbone,)
