import copy
import os
import sys
import time
import logging
import shutil
import argparse
import torch
from tensorboardX import SummaryWriter


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def run_pre(cfg):
    # from time
    if cfg.sleep > -1:
        for i in range(cfg.sleep):
            time.sleep(1)
            print('\rCount down : {} s'.format(cfg.sleep - 1 - i), end='')
    # from memory
    elif cfg.memory > -1:
        s_times = 0
        while True:
            os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Used > tmp')
            memory_used = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            if memory_used[0] < 3000:
                os.system('rm tmp')
                break
            else:
                s_times += 1
                time.sleep(1)
                print('\rWaiting for {} s'.format(s_times), end='')


def makedirs(dirs, exist_ok=False):
    if not isinstance(dirs, list):
        dirs = [dirs]
    for dir in dirs:
        os.makedirs(dir, exist_ok=exist_ok)


def init_checkpoint(cfg):
    def rm_zero_size_file(path):
        files = os.listdir(path)
        for file in files:
            path = '{}/{}'.format(cfg.logdir, file)
            size = os.path.getsize(path)  # unit:B
            if os.path.isfile(path) and size < 8:
                os.remove(path)

    os.makedirs(cfg.trainer.checkpoint, exist_ok=True)
    if cfg.trainer.resume_dir:
        cfg.logdir = '{}/{}'.format(cfg.trainer.checkpoint, cfg.trainer.resume_dir)
        checkpoint_path = cfg.model.kwargs['checkpoint_path']
        if checkpoint_path == '':
            cfg.model.kwargs['checkpoint_path'] = '{}/ckpt.pth'.format(cfg.logdir)
        else:
            cfg.model.kwargs['checkpoint_path'] = '{}/{}'.format(cfg.logdir, checkpoint_path.split('/')[-1])
        state_dict = torch.load(cfg.model.kwargs['checkpoint_path'], map_location='cpu')
        cfg.trainer.iter, cfg.trainer.epoch = state_dict['iter'], state_dict['epoch']
        cfg.trainer.metric_recorder = state_dict['metric_recorder']
    else:
        if cfg.master:
            logdir_sub = cfg.trainer.logdir_sub if cfg.trainer.logdir_sub != '' else time.strftime("%Y%m%d-%H%M%S")
            # logdir_exp = '{}_{}_{}_{}'.format(cfg.trainer.name, cfg.model.name, cfg.data.type, logdir_sub)
            logdir_exp = f"{cfg.trainer.name}_{cfg.cfg_path.replace('.', '_')}_{logdir_sub}"
            logdir = logdir_exp
            idx = 0
            while os.path.exists(logdir):
                logdir = f'{logdir_exp}_{idx}'
                idx += 1
            cfg.logdir = '{}/{}'.format(cfg.trainer.checkpoint, logdir)
            os.makedirs(cfg.logdir, exist_ok=True)
            shutil.copy(f"{cfg.cfg_path.replace('.', '/')}.py", '{}/{}.py'.format(cfg.logdir, cfg.cfg_path.split('.')[-1]))
        else:
            cfg.logdir = None
        cfg.trainer.iter, cfg.trainer.epoch = 0, 0
    cfg.logger = get_logger(cfg) if cfg.master else None
    cfg.writer = SummaryWriter(log_dir=cfg.logdir, comment='') if cfg.master else None
    log_msg(cfg.logger, f'==> Logging on master GPU: {cfg.logger_rank}')


# rm_zero_size_file(cfg.logdir) if cfg.master else None


def log_cfg(cfg):
    def _parse_Namespace(cfg, base_str=''):
        ret = {}
        if hasattr(cfg, '__dict__'):
            for key, val in cfg.__dict__.items():
                if not key.startswith('_'):
                    ret.update(_parse_Namespace(val, '{}.{}'.format(base_str, key).lstrip('.')))
        else:
            ret.update({base_str: cfg})
        return ret

    cfg_dict = _parse_Namespace(cfg)
    key_max_length = max(list(map(len, cfg_dict.keys())))
    excludes = ['writer.', 'logger.handlers']
    exclude_keys = []
    for k, v in cfg_dict.items():
        for exclude in excludes:
            if k.find(exclude) != -1:
                exclude_keys.append(k) if k not in exclude_keys else None
    # cfg_str = '\n'.join(
    # 	[(('{' + ':<{}'.format(key_max_length) + '} : {' + ':<{}'.format(key_max_length)) + '}').format(k, str(v)) for
    # 	 k, v in cfg_dict.items()])
    cfg_str = ''
    for k, v in cfg_dict.items():
        if k in exclude_keys:
            continue
        cfg_str += ('{' + ':<{}'.format(key_max_length) + '} : {' + ':<{}'.format(key_max_length) + '}').format(k, str(v))
        cfg_str += '\n'
    cfg_str = cfg_str.strip()
    cfg.cfg_dict, cfg.cfg_str = cfg_dict, cfg_str
    log_msg(cfg.logger, f'==> ********** cfg ********** \n{cfg.cfg_str}')


def get_logger(cfg, mode='a+'):
    log_format = '%(asctime)s - %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler('{}/log_{}.txt'.format(cfg.logdir, cfg.mode), mode=mode)
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)
    cfg.logger = logger
    return logger


def start_show(logger):
    logger.info('********************************************************************************')
    logger.info('==>   ======= ==    ==       ==     ============               ==            <==')
    logger.info('==>        == ==  ==           ==        ==           ====================   <==')
    logger.info('==>   ======= ===            ==  ==      ==           ==     ======     ==   <==')
    logger.info('==>   ==   ===========         ==        ==                    ==            <==')
    logger.info('==>   ======= ==                 ==      ==                    ==            <==')
    logger.info('==>     == == ==  ==           ==        ==                 == ==            <==')
    logger.info('==>        == ==    ==       ==     ============               ==            <==')
    logger.info('********************************************************************************')

    logger.info('********************************************************************************')
    logger.info('==>  =       =  =========  ========  =     =      =      =       =   ======  <==')
    logger.info('==>   =     =       =           ==   =     =     = =     = =     =  =    ==  <==')
    logger.info('==>    =   =        =         ==     =======    =====    =   =   =  =    ==  <==')
    logger.info('==>     = =         =       ==       =     =   =     =   =     = =  =        <==')
    logger.info('==>      =          =      ========  =     =  =       =  =       =   ======  <==')
    logger.info('********************************************************************************')


def able(ret, mark=False, default=None):
    return ret if mark else default


def log_msg(logger, msg, level='info'):
    if logger is not None:
        if msg is not None and level == 'info':
            logger.info(msg)


class AvgMeter(object):
    def __init__(self, name, fmt=':f', show_name='val', add_name=''):
        self.name = name
        self.fmt = fmt
        self.show_name = show_name
        self.add_name = add_name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '[{name} {' + self.show_name + self.fmt + '}'
        fmtstr += (' ({' + self.add_name + self.fmt + '})]' if self.add_name else ']')
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, meters, default_prefix=""):
        self.iter_fmtstr_iter = '{}: {:>3.2f}% [{}/{}]'
        self.iter_fmtstr_batch = ' [{:<.1f}/{:<3.1f}]'
        self.meters = meters
        self.default_prefix = default_prefix

    def get_msg(self, iter, iter_full, epoch=None, epoch_full=None, prefix=None):
        entries = [self.iter_fmtstr_iter.format(prefix if prefix else self.default_prefix, iter / iter_full * 100, iter, iter_full, epoch, epoch_full)]
        if epoch:
            entries += [self.iter_fmtstr_batch.format(epoch, epoch_full)]
        for meter in self.meters.values():
            entries.append(str(meter)) if meter.count > 0 else None
        return ' '.join(entries)


def get_log_terms(log_terms, default_prefix=''):
    terms = {}
    for t in log_terms:
        t = {k: v for k, v in t.items()}
        t_name = t['name']
        if t.get('suffixes', None) is not None:
            suffixes = t.pop('suffixes')
            for suffix in suffixes:
                t['name'] = t_name + suffix
                terms[t['name']] = AvgMeter(**t)
        else:
            terms[t_name] = AvgMeter(**t)
    progress = ProgressMeter(terms, default_prefix=default_prefix)
    return terms, progress


def update_log_term(term, val, n, master):
    term.update(val, n) if term and master else None


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk], [
        correct[:k].reshape(-1).float().sum(0) for k in topk] + [batch_size]


def get_timepc():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def get_net_params(net):
    num_params = 0
    for param in net.parameters():
        if param.requires_grad:
            num_params += param.numel()
    return num_params / 1e6
