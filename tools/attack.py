import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import register_all_modules 

def parse_args():
    parser = argparse.ArgumentParser(description= "attack detector")
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument("--attack", default='tsiaa')
    parser.add_argument("--step_size", type = int, default=1)
    parser.add_argument("--steps", type = int, default=20)
    parser.add_argument("--epsilon", type = int, default=16)
    parser.add_argument("--name", type=str, default='tsiaa_atss')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args        


if __name__ == "__main__":
    args = parse_args()
    
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])    
        
    cfg.load_from = args.checkpoint       
    
    attacker_default_args = dict(
        type = args.attack.upper() + "Attacker",
        img_mean = cfg.model.data_preprocessor.mean, 
        img_std = cfg.model.data_preprocessor.std,
        steps = args.steps,
        step_size = args.step_size,
        epsilon = args.epsilon,
    )
    cfg.attacker = attacker_default_args
   

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.name = args.name
    runner.attack()
    # print(f"There are {runner.attacker.count} invalid images")
    
