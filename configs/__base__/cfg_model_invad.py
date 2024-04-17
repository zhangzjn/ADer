from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_model_invad(Namespace):

	def __init__(self):
		Namespace.__init__(self)

		size = 256
		in_chas=[256, 512, 1024]
		out_indices = [i + 1 for i in range(len(in_chas))]
		out_cha = 256
		style_chas = [min(in_cha, out_cha) for in_cha in in_chas]
		in_strides = [2 ** (len(in_chas) - i - 1) for i in range(len(in_chas))]  # [4, 2, 1]
		latent_channel_size = 64
		self.model_encoder = Namespace()
		self.model_encoder.name = 'timm_wide_resnet50_2'
		self.model_encoder.kwargs = dict(pretrained=False,
										 checkpoint_path='model/pretrain/wide_resnet50_racm-8234f177.pth',
										 strict=False, features_only=True, out_indices=out_indices)
		self.model_fuser = dict(type='Fuser', in_chas=in_chas, style_chas=style_chas, in_strides=in_strides, down_conv=True, bottle_num=1, conv_num=1, lr_mul=0.01)

		latent_spatial_size = size // (2 ** (1 + len(in_chas)))
		self.model_decoder = dict(in_chas=in_chas, style_chas=style_chas,
								  latent_spatial_size=latent_spatial_size, latent_channel_size=latent_channel_size,
								  blur_kernel=[1, 3, 3, 1], normalize_mode='LayerNorm',
								  lr_mul=0.01, small_generator=True, layers=[2] * len(in_chas))

		sizes = [size // (2 ** (2 + i)) for i in range(len(in_chas))]
		self.model_disor = dict(sizes=sizes, in_chas=in_chas)

		self.model = Namespace()
		self.model.name = 'invad'
		self.model.kwargs = dict(pretrained=False,
								 checkpoint_path='',
								 # checkpoint_path='runs/InvADTrainer_configs_invad_invad_mvtec_20230614-015946/net.pth',
								 strict=True,
								 model_encoder=self.model_encoder,
								 model_fuser=self.model_fuser,
								 model_decoder=self.model_decoder)
