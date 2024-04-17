import glob
import importlib

from util.registry import Registry
LOSS = Registry('Loss')

files = glob.glob('loss/[!_]*.py')
for file in files:
	model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))


def get_loss_terms(loss_terms, device='cpu'):
	terms = {}
	for t in loss_terms:
		t = {k: v for k, v in t.items()}
		t_type = t.pop('type')
		t_name = t.pop('name')
		terms[t_name] = LOSS.get_module(t_type)(**t).to(device).eval()
	return terms
