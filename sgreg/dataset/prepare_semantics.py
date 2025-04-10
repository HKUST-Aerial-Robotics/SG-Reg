'''
    Generate median features for the scene graph,
    - Semantic embeddings. Encoded from bert.
    - Hierarchical points features. Encoded in pair of sub-scenes.
'''


import os
import torch
import torch.utils.data
import pandas as pd
import time
# import numpy as np 
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from sgreg.bert.get_tokenizer import get_tokenlizer, get_pretrained_language_model
from sgreg.bert.bertwarper import generate_bert_fetures

# from model.ops.transformation import apply_transform
from sgreg.dataset.ScenePairDataset import ScenePairDataset
# from model.dataset.DatasetFactory import train_data_loader, val_data_loader
from sgreg.config import create_cfg
# from model.dataset.ScanNetPairDataset import read_scans,load_scene_graph

def generate_semantic_embeddings(data_dict):
    src_labels = data_dict['src_graph']['labels']
    ref_labels = data_dict['ref_graph']['labels']
    src_idxs = data_dict['src_graph']['idx2name']
    ref_idxs = data_dict['ref_graph']['idx2name']
    assert len(src_labels) == src_idxs.shape[0] and len(ref_labels) == ref_idxs.shape[0]
    
    labels = src_labels + ref_labels
    
    t0 = time.time()
    semantic_embeddings = generate_bert_fetures(tokenizer,bert,labels)
    t1 = time.time()
    print('encode {} labels takes {:.3f} msecs'.format(len(labels),1000*(t1-t0)))
    assert len(semantic_embeddings) == len(labels)

    src_semantic_embeddings = semantic_embeddings[:len(src_labels)].detach()
    ref_semantic_embeddings = semantic_embeddings[len(src_labels):].detach()

    return {'instance_idxs':src_idxs,'semantic_embeddings':src_semantic_embeddings},\
            {'instance_idxs':ref_idxs,'semantic_embeddings':ref_semantic_embeddings}

if __name__=='__main__':
    print('Save the semantic embeddings to accelerate the training process.')
    ##
    dataroot = '/data2/RioGraph' # '/data2/ScanNetGraph'
    split = 'val'
    cfg_file = '/home/cliuci/code_ws/SceneGraphNet/config/rio.yaml'
    middle_feat_folder = os.path.join(dataroot,'matches')
    from sgreg.dataset.DatasetFactory import prepare_hierarchy_points_feats

    ##
    tokenizer = get_tokenlizer('bert-base-uncased')
    bert = get_pretrained_language_model('bert-base-uncased') 
    bert.eval()
    bert.pooler.dense.weight.requires_grad = False
    bert.pooler.dense.bias.requires_grad = False
    
    # 
    conf = OmegaConf.load(cfg_file)
    conf = create_cfg(conf)
    conf.dataset.online_bert = True
    dataset = ScenePairDataset(dataroot,split,conf)
    
    neighbor_limits = [38, 36, 36, 38]
    print('neighbor_limits:',neighbor_limits)
    cfg_dict= {'num_stages': conf.backbone.num_stages, 
               'voxel_size': conf.backbone.init_voxel_size, 
               'search_radius': conf.backbone.init_radius, 
               'neighbor_limits': neighbor_limits}

    #
    N = len(dataset)
    print('Dataset size:',N)
    
    SEMANTIC_ON = True
    POINTS_ON = False
    CHECK_EDGES = False
    max_fine_points = []
    
    warn_scans = []
    
    for i in range(N):
        data_dict = dataset[i]
        scene_name = data_dict['src_scan'][:-1]
        src_subname = data_dict['src_scan'][-1]
        ref_subname = data_dict['ref_scan'][-1]
        print('---processing {} and {} -----'.format(data_dict['src_scan'],data_dict['ref_scan']))
        if data_dict['instance_ious'] is None:
            warn_scans.append((data_dict['src_scan'],data_dict['ref_scan']))
        
        if CHECK_EDGES:
            src_global_edges = data_dict['src_graph']['global_edge_indices']
            ref_global_edges = data_dict['ref_graph']['global_edge_indices']
            print('{} global src edges'.format(src_global_edges.shape[0]))
            
            if src_global_edges.shape[0]<1 or ref_global_edges.shape[0]<1:
                print('global_edges is None!')
                warn_scans.append((data_dict['src_scan'],data_dict['ref_scan']))
        
        if SEMANTIC_ON:
            src_semantic_embeddings, ref_semantic_embeddings = generate_semantic_embeddings(data_dict)
            torch.save(src_semantic_embeddings,
                    os.path.join(dataroot,split,data_dict['src_scan'],'semantic_embeddings.pth'))
            torch.save(ref_semantic_embeddings,
                    os.path.join(dataroot,split,data_dict['ref_scan'],'semantic_embeddings.pth'))
        if POINTS_ON:
            out_dict = prepare_hierarchy_points_feats(cfg_dict=cfg_dict,
                                                      ref_points = data_dict['ref_points'],
                                                      src_points=data_dict['src_points'],
                                                      ref_instances=data_dict['ref_instances'],
                                                      src_instances=data_dict['src_instances'],
                                                      ref_feats=data_dict['ref_feats'],
                                                      src_feats=data_dict['src_feats'])
            import numpy as np
            
            ref_instance_count = np.histogram(data_dict['ref_instances'],bins=np.unique(data_dict['ref_instances']))[0]
            print('ref instance list:',np.unique(data_dict['ref_instances']))
            print('ref_instance points count:',ref_instance_count)
            print('points number: ', data_dict['ref_points'].shape[0])

            ref_instance_list = np.unique(out_dict['ref_points_f_instances'])
            ref_instance_fine_count = np.histogram(out_dict['ref_points_f_instances'],bins=ref_instance_list)[0]
            print('ref instance list:',ref_instance_list)
            print('ref_instance fine points count:',ref_instance_fine_count)
            print('max fine points: ',ref_instance_fine_count.max())
            max_fine_points.append(ref_instance_fine_count.max())

            points_dict =  out_dict['points_dict']
            feats = out_dict['feats']
            ref_points_f_instances= out_dict['ref_points_f_instances']
            src_points_f_instances= out_dict['src_points_f_instances']
            assert 'instance_matches' in data_dict
            assert isinstance(ref_points_f_instances,torch.Tensor) and isinstance(src_points_f_instances,torch.Tensor)
            assert points_dict['points'][0].shape[0] == points_dict['lengths'][0].sum()
            # Export
            data_dict.update(out_dict)
            torch.save(data_dict,os.path.join(middle_feat_folder,scene_name,'data_dict_{}.pth'.format(src_subname+ref_subname)))
        
        # break
    print(warn_scans)
    print('finished {} scan pairs'.format(N))
    
    if len(max_fine_points)>0:   
        max_fine_points = np.array(max_fine_points)
        print(max_fine_points)
        print(max_fine_points.max())
        
    


