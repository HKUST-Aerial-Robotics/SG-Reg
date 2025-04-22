from functools import partial
import math, time
import numpy as np
import torch
from sgreg.ops import grid_subsample, radius_search
from sgreg.utils.torch import build_dataloader
from sgreg.dataset.scene_pair_dataset import check_instance_points
from torch_geometric.data import Batch
import torch


# Stack mode utilities
def precompute_data_stack_mode(points, insts, lengths, num_stages, voxel_size, radius, neighbor_limits):
    assert num_stages == len(neighbor_limits)

    points_list = []
    insts_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, insts, lengths = grid_subsample(points, insts, lengths, voxel_size=voxel_size)
        points_list.append(points)
        insts_list.append(insts)
        lengths_list.append(lengths)
        voxel_size *= 2

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search(
            cur_points,
            cur_points,
            cur_lengths,
            cur_lengths,
            radius,
            neighbor_limits[i],
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            subsampling_list.append(subsampling)

            upsampling = radius_search(
                cur_points,
                sub_points,
                cur_lengths,
                sub_lengths,
                radius * 2,
                neighbor_limits[i + 1],
            )
            upsampling_list.append(upsampling)

        radius *= 2

    return {
        'points': points_list,
        'insts': insts_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }
    
def stack_scene_graph_list(graph_dicts):
    graph_dict = {'scene_mask':[],'batch':[0]}
    for scene_id,graph in enumerate(graph_dicts):
        for key, value in graph.items():
            if key not in graph_dict:
                graph_dict[key] = []
            graph_dict[key].append(value)
            if key=='edge_indices' or key=='global_edge_indices': # (E,2) 
                graph_dict[key][-1] = graph_dict[key][-1] + torch.ones_like(graph_dict[key][-1]) * graph_dict['batch'][-1]

        graph_dict['scene_mask'].append(torch.tensor(scene_id).repeat(graph['centroids'].shape[0]))
        graph_dict['batch'].append(graph['centroids'].shape[0]+graph_dict['batch'][-1])
        
    for k,v in graph_dict.items():
        if k=='labels':
            import itertools
            graph_dict[k] = list(itertools.chain.from_iterable(v))
        elif k=='batch':
            graph_dict[k] = torch.tensor(v,dtype=torch.long)
        else:
            graph_dict[k] = torch.cat(v,dim=0)
    return graph_dict

def prepare_hierarchy_points_feats(cfg_dict, 
                                   ref_points, 
                                   src_points, 
                                   ref_instances, 
                                   src_instances, 
                                   ref_feats, 
                                   src_feats):
    r"""Prepare hierarchy points and features for the scene graph.
    It is a module built in GeoTransformer.
    It downsample the points and features into multiple-levels. We only use the fine-level of the points and features.
    The fine-level samples 1/2 of the input points.

    Args:
        scene_id (int)
        ref_points: (P,3)
        src_points: (Q,3)
        ref_instances: (P,)
        src_instances: (Q,)
        ref_feats: (P,1), initialized using torch.ones()
        src_feats: (Q,1), initialized using torch.ones()
    Returns:
        points_dict (Dict), all in torch.Tensor:
    """

    ref_length = ref_points.shape[0]
    src_length = src_points.shape[0]
    points = torch.cat([ref_points, src_points], dim=0)
    instances = torch.cat([ref_instances, src_instances], dim=0)
    feats = torch.cat([ref_feats, src_feats], dim=0).reshape(-1,1)
    lengths = torch.LongTensor([ref_length, src_length])
    t0 = time.time()
    points_dict = precompute_data_stack_mode(points, 
                                             instances,
                                             lengths, 
                                             cfg_dict['num_stages'],
                                             cfg_dict['voxel_size'],
                                             cfg_dict['search_radius'], 
                                             cfg_dict['neighbor_limits'])
    t1 = time.time()
    loader_duration = t1 - t0
    assert points.shape[0] == points_dict['points'][0].shape[0]
    assert feats.shape[0] == lengths.sum(), 'feats shape {} not match with lengths {}'.format(feats.shape,lengths.sum())
    
    # ref_length_f = points_dict['lengths'][1][0]
    # points_f = points_dict['points'][1]
    # ref_points_f = points_f[:ref_length_f] # (P',3)
    # src_points_f = points_f[ref_length_f:] # (Q',3)

    # ref_points_f_instances = associate_points_f_instances(
    #     ref_points_f.detach().numpy(),
    #     ref_points.detach().numpy(),
    #     ref_instances.detach().numpy()) # (P',)
    # src_points_f_instances = associate_points_f_instances(
    #     src_points_f.detach().numpy(),
    #     src_points.detach().numpy(),
    #     src_instances.detach().numpy()) # (Q',)

    return {'points_dict':points_dict,
            'feats':feats,
            # 'ref_points_f_instances':torch.from_numpy(ref_points_f_instances),
            # 'src_points_f_instances':torch.from_numpy(src_points_f_instances),
            'loader_duration':loader_duration}



def single_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for single point cloud in stack mode.

    Points are organized in the following order: [P_1, ..., P_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool=True)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: feats, points, normals
    if 'normals' in collated_dict:
        normals = torch.cat(collated_dict.pop('normals'), dim=0)
    else:
        normals = None
    feats = torch.cat(collated_dict.pop('feats'), dim=0)
    points_list = collated_dict.pop('points')
    #points_list = collated_dict.pop('insts')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)


    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    if normals is not None:
        collated_dict['normals'] = normals
    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        # voxelsize:0.05, search_radius:0.0625, neighbor_limits:[38, 36, 36, 38]
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def registration_collate_fn_stack_mode(data_dicts, 
                                       num_stages, 
                                       voxel_size, 
                                       search_radius, 
                                       neighbor_limits, 
                                       precompute_data=True, 
                                       points_limit=None
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    insts_list = collated_dict.pop('ref_instances') + collated_dict.pop('src_instances')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)
    insts = torch.cat(insts_list, dim=0).int()
    if 'sg' in collated_dict.keys():
        follow_batch=['x_q', 'x_t']
        exclude_keys=None
        sg_pair_batched = Batch.from_data_list(collated_dict['sg'], follow_batch, exclude_keys)
        collated_dict['sg'] = sg_pair_batched
        
    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            if key != 'sg' and key != 'sg_match':
                collated_dict[key] = value[0]

    collated_dict['features'] = feats
    import time
    
    t0 = time.time()
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, insts, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        voxel_size_ = voxel_size
        if points_limit > 0:
            while input_dict['points'][-1].shape[0] > points_limit:
                voxel_size_ = voxel_size_*math.sqrt(2)
                input_dict = precompute_data_stack_mode(points, insts, lengths, num_stages, voxel_size_, search_radius, neighbor_limits)
        collated_dict['voxel_size'] = voxel_size_
        collated_dict.update(input_dict)
        
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size
    collated_dict['stack_timing'] = [time.time()-t0]
    #print(collated_dict.keys())

    return collated_dict

def sgreg_collate_fn_stack_mode(data_dicts, 
                                num_stages, 
                                voxel_size, 
                                search_radius, 
                                neighbor_limits,
                                verify_instance_points=False,
                                point_limits=1500):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict]), each dict contains {ref_points,src_points,ref_feats,src_feats,transform}
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    ref_feats_list = collated_dict.pop('ref_feats')
    src_feats_list = collated_dict.pop('src_feats')
    ref_points_list = collated_dict.pop('ref_points')
    src_points_list = collated_dict.pop('src_points')
    ref_instances_list = collated_dict.pop('ref_instances')
    src_instances_list = collated_dict.pop('src_instances')
    src_graph_list = collated_dict.pop('src_graph')
    ref_graph_list = collated_dict.pop('ref_graph')    
    collated_dict['batch_points'] = []
    collated_dict['batch_features'] = []
    # collated_dict['batch_ref_instances'] = []
    # collated_dict['batch_src_instances'] = []    
    # assert isinstance(collated_dict['points_dict'], list)

    if 'points_dict' in collated_dict:
        # Load pre-computed data
        assert False, 'Abandoned'
        collated_dict['batch_points'] = collated_dict.pop('points_dict')
        collated_dict['batch_features'] = collated_dict.pop('feats')
        collated_dict['batch_ref_instances'] = collated_dict.pop('ref_points_f_instances')
        collated_dict['batch_src_instances'] = collated_dict.pop('src_points_f_instances')

    cfg_dict = {'num_stages':num_stages,
                'voxel_size':voxel_size,
                'search_radius':search_radius,
                'neighbor_limits':neighbor_limits}

    for scene_id in range(batch_size):
        out_dict = prepare_hierarchy_points_feats(
            cfg_dict=cfg_dict,
            ref_points=ref_points_list[scene_id],
            src_points=src_points_list[scene_id],
            ref_instances=ref_instances_list[scene_id],
            src_instances=src_instances_list[scene_id],
            ref_feats=ref_feats_list[scene_id],
            src_feats=src_feats_list[scene_id])
        if out_dict['points_dict']['lengths'][-1].sum() > point_limits:
            print('[WARNING] {} superpoints. Require large memory.'.format(out_dict['points_dict']['lengths'][-1].sum()))
    
        collated_dict['batch_points'].append(out_dict['points_dict'])
        collated_dict['batch_features'].append(out_dict['feats'])

        if verify_instance_points: # verify each instance has valid points
            instances_f = out_dict['points_dict']['insts'][1]
            lengths_f = out_dict['points_dict']['lengths'][1]
            ref_instances_f = instances_f[:lengths_f[0]]
            src_instances_f = instances_f[lengths_f[0]:]
            ref_idxs = ref_graph_list[scene_id]['idx2name']
            src_idxs = src_graph_list[scene_id]['idx2name']
            check_instance_points(torch.arange(ref_idxs.shape[0]),
                                  ref_instances_f.clone().squeeze(),
                                  collated_dict['src_scan'][scene_id])
            check_instance_points(torch.arange(src_idxs.shape[0]),
                                    src_instances_f.clone().squeeze(),
                                    collated_dict['ref_scan'][scene_id])

        if False: # check the points_f_instances
            raw_src_instace_list = torch.unique(src_instances_list[scene_id])
            src_instance_f_list = torch.unique(out_dict['src_points_f_instances'])
            src_id2name = src_graph_list[0]['idx2name']
            if src_instance_f_list.shape[0] != raw_src_instace_list.shape[0]:
                # print('{}-{} inconsistent instance number!!!'.format(
                #     collated_dict['src_scan'][scene_id],
                #     collated_dict['ref_scan'][scene_id]))
                
                # find the inconsistent instance idx
                inconsistent_id = []
                for id in raw_src_instace_list:
                    if id not in src_instance_f_list:
                        # inst_idx = 
                        inconsistent_id.append(id)
                print('inconsistent instance id:',inconsistent_id)
                collated_dict['debug_flag'] = raw_src_instace_list.shape[0] - src_instance_f_list.shape[0]
    
    # Stack scene graphs      
    collated_dict['src_graph'] = stack_scene_graph_list(src_graph_list) 
    collated_dict['ref_graph'] = stack_scene_graph_list(ref_graph_list)

    # Stack others
    stack_transforms = [t_.unsqueeze(0) for t_ in collated_dict['transform']]
    collated_dict['transform'] = torch.cat(stack_transforms,dim=0) # [B,4,4]              
    collated_dict['batch_size'] = batch_size
    collated_dict['loader_duration'] = out_dict['loader_duration']

    return collated_dict


def calibrate_neighbors_stack_mode(dataset, 
                                   collate_fn, 
                                   num_stages, 
                                   voxel_size, 
                                   search_radius, 
                                   keep_ratio=0.8, 
                                   sample_threshold=20000, 
                                   points_limit=-1):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i, sample in enumerate(dataset):
        data_dict = collate_fn([sample], 
                               num_stages, 
                               voxel_size, 
                               search_radius, 
                               max_neighbor_limits, 
                               precompute_data=True, 
                               points_limit=points_limit)

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    search_radius,
    neighbor_limits,
    verify_instance_points,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    distributed=False,
    # precompute_data=True,
    # reproducibility=True,
    # point_limits=None
):

    # if reproducibility:
    #     g = torch.Generator()
    #     g.manual_seed(0)
    # else:
    #     g = None

    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            search_radius=search_radius,
            neighbor_limits=neighbor_limits,
            verify_instance_points = verify_instance_points,
            # precompute_data=precompute_data,
            # point_limits=point_limits
        ),
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader


# def get_dataloader(train_dataset, val_dataset, cfg, args):

#     '''
#     if isinstance(train_dataset, torch.utils.data.IterableDataset):
#         neighbor_limits = cfg.dataset.neighbor_limits
#     else:
#         neighbor_limits = calibrate_neighbors_stack_mode(
#         train_dataset,
#         registration_collate_fn_stack_mode,
#         cfg.backbone.num_stages,
#         cfg.backbone.init_voxel_size,
#         cfg.backbone.init_radius
#     )
#     '''
#     neighbor_limits = cfg.dataset.neighbor_limits

#     train_loader = build_dataloader_stack_mode(
#         train_dataset,
#         registration_collate_fn_stack_mode,
#         cfg.backbone.num_stages,
#         cfg.backbone.init_voxel_size,
#         cfg.backbone.init_radius,
#         neighbor_limits,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         shuffle=True,
#         distributed=False,
#         reproducibility=args.reproducibility,
#         point_limits=cfg.dataset.max_c_points
#     )

#     val_loader = build_dataloader_stack_mode(
#         val_dataset,
#         registration_collate_fn_stack_mode,
#         cfg.backbone.num_stages,
#         cfg.backbone.init_voxel_size,
#         cfg.backbone.init_radius,
#         neighbor_limits,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         shuffle=False,
#         distributed=False,
#         reproducibility=args.reproducibility,
#         point_limits=cfg.dataset.max_c_points
#     )
#     return train_loader, val_loader


# def collate_fn_pointnet(data_dicts):
#     batch_size = len(data_dicts)
#     # merge data with the same key from different samples into a list
#     collated_dict = {}
#     for data_dict in data_dicts:
#         for key, value in data_dict.items():
#             if isinstance(value, np.ndarray):
#                 value = torch.from_numpy(value)
#             if key not in collated_dict:
#                 collated_dict[key] = []
#             collated_dict[key].append(value)
    
#     if 'sg' in collated_dict.keys():
#         follow_batch=['x_q', 'x_t']
#         exclude_keys=None
#         sg_pair_batched = Batch.from_data_list(collated_dict['sg'], follow_batch, exclude_keys)
#         collated_dict['sg'] = sg_pair_batched
        
#     if batch_size == 1:
#         # remove wrapping brackets if batch_size is 1
#         for key, value in collated_dict.items():
#             if key != 'sg' and key != 'sg_match':
#                 collated_dict[key] = value[0]
    
#     collated_dict['batch_size'] = batch_size
#     return collated_dict

# def get_dataloader_pointnet(train_dataset, val_dataset, cfg, args):

#     g = torch.Generator()
#     g.manual_seed(0)

#     train_loader = torch.utils.data.DataLoader(
#             train_dataset,
#             batch_size=args.batch_size,
#             num_workers=args.num_workers,
#             generator=g,
#             shuffle=True,
#             collate_fn=collate_fn_pointnet,
#             pin_memory=False,
#             drop_last=False,
#     )
    
#     val_loader = torch.utils.data.DataLoader(
#             val_dataset,
#             batch_size=args.batch_size,
#             num_workers=args.num_workers,
#             generator=g,
#             shuffle=True,
#             collate_fn=collate_fn_pointnet,
#             pin_memory=False,
#             drop_last=False,
#     )

#     return train_loader, val_loader
