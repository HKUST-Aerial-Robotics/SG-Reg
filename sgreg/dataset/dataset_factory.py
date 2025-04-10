import sys, os
import torch
import numpy as np
import open3d as o3d
from sgreg.dataset.scene_pair_dataset import (
    ScenePairDataset, 
)
from sgreg.dataset.stack_mode import (
    calibrate_neighbors_stack_mode,
    registration_collate_fn_stack_mode,
    sgreg_collate_fn_stack_mode,
    build_dataloader_stack_mode
)

def train_data_loader(cfg,distributed=False):
    train_dataset = ScenePairDataset(cfg.dataset.dataroot, 'train', cfg)    
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_dataset.neighbor_limits = neighbor_limits
    if 'verify_instance_points' in cfg.dataset:
        verify_instance_points = cfg.dataset.verify_instance_points
    else:
        verify_instance_points = False
    
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        sgreg_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        verify_instance_points,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )
    return train_loader,neighbor_limits

def val_data_loader(cfg,distributed=False):
    val_dataset = ScenePairDataset(cfg.dataset.dataroot, 'val', cfg)    
    neighbor_limits = calibrate_neighbors_stack_mode(
        val_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    print('Calibrated neighbor limits:',neighbor_limits)
    # neighbor_limits = [38, 36, 36, 38] # default setting in 3DMatch
    val_dataset.neighbor_limits = neighbor_limits
    
    if 'verify_instance_points' in cfg.dataset:
        verify_instance_points = cfg.dataset.verify_instance_points
    else:
        verify_instance_points = False
    
    val_loader = build_dataloader_stack_mode(
        val_dataset,
        sgreg_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        verify_instance_points,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        distributed=distributed,
    )
    return val_loader,neighbor_limits

if __name__=='__main__':
    import torch
    from omegaconf import OmegaConf
    conf = OmegaConf.load('config/pretrain.yaml')
    conf.backbone.init_radius = 0.5 * conf.backbone.base_radius * conf.backbone.init_voxel_size # 0.0625
    conf.backbone.init_sigma  = 0.5 * conf.backbone.base_sigma * conf.backbone.init_voxel_size # 0.05
    
    train_loader, train_neighbor_limits = train_data_loader(conf)
    val_loader, val_neighbor_limits = val_data_loader(conf)
    
    # print('train neighbor limits',train_neighbor_limits)
    # print('val neighbor limits',val_neighbor_limits)
    # exit(0)
    # data_dict = next(iter(train_loader))
    # src_graph = data_dict['src_graph']
    # print(data_dict.keys())
    # print(src_graph.keys())    
    
    warning_scans = []
    valid_scans = []
    
    for data_dict in val_loader:
        src_graph = data_dict['src_graph']
        src_scan = data_dict['src_scan'][0]
        ref_scan = data_dict['ref_scan'][0] 
        msg = '{}-{}: '.format(src_scan,ref_scan) 
        points_dict= data_dict['batch_points'][0]     
        # print(data_dict.keys())
        lengths = points_dict['lengths'] # [(Pl,Ql)]

        # if 'debug_flag' in data_dict:
        #     print('{} inconsistent instance {}-{} !!!'.format(
        #         data_dict['debug_flag'], src_scan,ref_scan, ))
        if data_dict['instance_matches'][0] is None:
            print('{}-{} no gt instance matches!!!'.format(src_scan,ref_scan))
            warning_scans.append('{} {}'.format(src_scan,ref_scan))
        else:
            valid_scans.append('{} {}'.format(src_scan,ref_scan))
        print(msg)
        # break
        
    print('************ Finished ************')
    print('{}/{} warning scans'.format(len(warning_scans),
                                       len(valid_scans)+len(warning_scans)))
    
    
    # outdir = '/data2/sgalign_data/splits/val_ours.txt'
    # with open(outdir,'w') as f:
    #     f.write('\n'.join(valid_scans))
    #     f.close()
