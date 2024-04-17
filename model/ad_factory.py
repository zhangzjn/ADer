from timm.models._registry import _model_entrypoints
import torchvision
from . import MODEL


for timm_name, timm_fn in _model_entrypoints.items():
	MODEL.register_module(timm_fn, f'timm_{timm_name}')

for tv_name, tv_fn in torchvision.models.__dict__.items():
	if not tv_name.startswith('_'):
		MODEL.register_module(tv_fn, f'tv_{tv_name}')


if __name__ == '__main__':
	print()
