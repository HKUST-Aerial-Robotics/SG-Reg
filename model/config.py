import argparse
import yaml
from yaml import Loader
import os

def get_parser():
    parser = argparse.ArgumentParser(description='GNN for scene graph matching')
    parser.add_argument('--config', type=str, default='config/scannet.yaml', help='path to config file')

    args = parser.parse_args()
    assert args.config is not None
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # setattr(args, 'exp_path', os.path.join('exp', args.dataset, args.model_name, args.config.split('/')[-1][:-5]))

    return args

def create_cfg(cfg):
    
    cfg.backbone.init_radius = 0.5 * cfg.backbone.base_radius * cfg.backbone.init_voxel_size # 0.0625
    cfg.backbone.init_sigma  = 0.5 * cfg.backbone.base_sigma * cfg.backbone.init_voxel_size # 0.05
    
    return cfg

