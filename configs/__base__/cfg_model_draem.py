from argparse import Namespace


class cfg_model_draem(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model = Namespace()
		self.model.name = 'draem'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True)
