import glob
import importlib
from configs.__base__.cfg_common import cfg_common
from configs.__base__.cfg_dataset_default import cfg_dataset_default


files = glob.glob('configs/__base__/[!_]*.py')
for file in files:
    model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))
    for obj_name in dir(model_lib):
        if obj_name.startswith("cfg_model"):
            globals()[obj_name] = getattr(model_lib, obj_name)
