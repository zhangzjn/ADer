from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_model_realnet(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_backbone = Namespace()
		# name = 'wide_resnet50_2'
		checkpoint_path = '../pretrain_models/wide_resnet50_2-95faca4d.pth'
		self.model_backbone.name = 'timm_wide_resnet50_2'
		self.model_backbone.outlayers = []
		self.model_backbone.out_indices = []
		self.model_backbone.kwargs = dict(pretrained=False,
									checkpoint_path=checkpoint_path,
									strict=False,
									features_only=True, out_indices=self.model_backbone.out_indices)
		
		self.model_afs = Namespace()
		self.model_afs.init_bsn = 64
		self.model_afs.structure = [
			dict(name='block1', layers=[dict(idx='layer1', planes=256)],stride=4),
			dict(name='block2', layers=[dict(idx='layer2', planes=512)],stride=8),
			dict(name='block3', layers=[dict(idx='layer3', planes=512)],stride=16),
			dict(name='block4', layers=[dict(idx='layer4', planes=256)],stride=32)]

		self.model_recon = Namespace()
		self.model_recon.num_res_blocks=2
		self.model_recon.hide_channels_ratio=0.5
		self.model_recon.channel_mult=[1,2,4]
		self.model_recon.attention_mult=[2,4]
		
		self.model_rrs = Namespace()
		self.model_rrs.modes = ['max','mean']
		self.model_rrs.mode_numbers = [256,256] # dimensions of RRS, max=256,mean=256
		self.model_rrs.num_residual_layers = 2
		self.model_rrs.stop_grad = False
		
		self.model = Namespace()
		self.model.name = 'realnet'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_backbone=self.model_backbone, 
						   model_afs=self.model_afs, model_recon=self.model_recon, model_rrs=self.model_rrs)
		