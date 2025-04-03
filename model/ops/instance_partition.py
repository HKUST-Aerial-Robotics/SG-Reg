import torch

@torch.no_grad()
def point_to_instance_partition(
    points: torch.Tensor,
    points_instance: torch.Tensor,
    instances: torch.Tensor,
    point_limit: int = 128,
):
    '''
    Sample point_limit points for each instance
    Input,
        - points: (N, 3), fine-level points
        - points_instances: (N, 1), fine points instances
        - instances: (M,), instance idx
    '''
    
    assert points.shape[0] == points_instance.shape[0]
    M = instances.shape[0]
    instance_points_masks = instances.unsqueeze(1).repeat(1, points.shape[0]) - points_instance.unsqueeze(0).repeat(M,1)  # (M, N)
    instance_points_masks = instance_points_masks == 0
    instance_points_count = instance_points_masks.sum(dim=1)
    
    instance_masks = instance_points_count> point_limit  # (M,)
    assert torch.all(instance_points_count>1), 'Some instances are not assigned to any fine points.'
    
    instance_knn_indices = torch.tensor(points.shape[0]).repeat((M, point_limit)).to(points.device)  # (M, K)
    small_instances = torch.where(instance_masks==False)[0]
    for idx in small_instances:
        f_points_count = instance_points_masks[idx,:].sum()
        instance_knn_indices[idx,:f_points_count] = torch.where(instance_points_masks[idx,:])[0] 
    instance_knn_indices[instance_masks] = torch.multinomial(instance_points_masks[instance_masks].float(), point_limit)  # (M, K)
    instance_knn_masks = instance_knn_indices < points.shape[0]  # (M, K)
    instance_knn_count = instance_knn_masks.sum(1)  # (M,)
    
    assert instance_knn_indices.min() < points.shape[0], 'Some instances are not assigned to any points.'
    
    return instance_knn_indices.to(torch.int64), instance_knn_masks, instance_knn_count

@torch.no_grad()
def instance_f_points_batch(
    points_instance: torch.Tensor,
    instances_list: torch.Tensor,
    point_limit: int = 1024,
):
    '''
    Input:
        - points_instances: (N, 1), fine points instances
        - instances: (M,), instance idx
    Output:
        - instance_f_points_indices: (M, K), fine points indices
    '''
    M = instances_list.shape[0]
    # instance_count = torch.histogram(points_instance.clone().detach().cpu(), 
                                    #  bins=M,range=(instances_list.min().item(),instances_list.max().item()+1))[0]
    # K = instance_count.max().int().item()
    K = point_limit
    instance_f_points_indices = torch.zeros((M, K), dtype=torch.int64).cuda()

    for id in instances_list:
        instance_f_points_masks = points_instance == id # (N, 1)
        count = instance_f_points_masks.sum()
        assert count>1, 'An instance has none fine points.'
        instance_f_points_indices[id] = torch.multinomial(instance_f_points_masks.float(), K, replacement=True)
        
        # instance_f_points_indices[id][:count] = torch.where(instance_f_points_masks)[0]
        # if count<K:
        #     instance_f_points_indices[id][count:] = torch.multinomial(instance_f_points_masks.float(), K-count, replacement=True)

    return instance_f_points_indices

@torch.no_grad
def sample_instance_from_points(
    points_instance: torch.Tensor,
    instances_list: torch.Tensor,
    K: int = 1024,
    invalid: int = -1):
    '''
    Sample K points for each instance. 
    If the number of points is less than K, fill the rest with invalid value.
    Input:
        - points_instances: (X, 1), fine points instances
        - instances: (N,), instance idx
        - K: int, number of samples
        - invalid: int, for the instances less than K, fill the rest with invalid value
    Output:
        - instance_f_points_indices: (N, K), fine points indices
        - instance_f_points_count: (N,), number of fine points
    '''
    N = instances_list.shape[0]
    
    # Sample K points for each instance
    instance_f_points_masks = instances_list.unsqueeze(1).repeat(1, points_instance.shape[0]) \
        - points_instance.T.repeat(N,1)  # (N, X)
    instance_f_points_masks = instance_f_points_masks == 0
    instance_f_points_count = instance_f_points_masks.sum(dim=1) # (N,)
    invalid_instances = instance_f_points_count<1
    if torch.any(invalid_instances):
        # print('Some instances are not assigned to any fine points.')
        instance_f_points_indices = invalid*torch.ones((N, K), dtype=torch.int64).to(points_instance.device)
    else:
        instance_f_points_indices = torch.multinomial(instance_f_points_masks.float(), 
                                                    K, 
                                                    replacement=True) # (N, K)
    
    return instance_f_points_indices, invalid_instances

@torch.no_grad
def sample_all_instance_points(
    instance_f_points_indices: torch.Tensor,
    invalid: int,
):
    '''
    An instance contain invalid indices due to its number of points less than K.
    Fill the invalid indices by random sampling from the valid indices.
    '''
    
    out_instance_f_points_indices = instance_f_points_indices.clone()
    instance_points_mask = instance_f_points_indices< invalid # (N,K)
    instance_points_count = instance_points_mask.sum(dim=1) # (N,)
    small_instances = instance_points_count<instance_f_points_indices.shape[1] # (N,)
    if torch.any(small_instances):
        small_indices_select_indices = torch.multinomial(
            instance_points_mask[small_instances].float(), instance_f_points_indices.shape[1],replacement=True)
        # tmp = out_instance_f_points_indices[small_instances][small_indices_select_indices] # cause indexing error
        small_instance_list = torch.where(small_instances)[0]
        for i, idx in enumerate(small_instance_list):
            new_indices = instance_f_points_indices[idx][small_indices_select_indices[i]]
            out_instance_f_points_indices[idx] = new_indices

    return out_instance_f_points_indices

@torch.no_grad
def sample_k_instance_points(
    instance_f_points_indices: torch.Tensor,
    max_point_index: int,
    K: int
):
    '''
    An instance contain invalid indices due to its limited number of points.
    For each instance, sample K points from the valid indices.
    If the number of points is less than K, fill the rest with max_point_index.
    '''
    instances_number, points_number = instance_f_points_indices.shape
    assert K<=points_number, 'K should be less than the number of points.'
    out_instance_f_points_indices = torch.zeros((instances_number, K), dtype=torch.int64).to(instance_f_points_indices.device)
    out_instance_f_points_indices.fill_(max_point_index)
    instance_points_mask = instance_f_points_indices< max_point_index # (N,K)
    instance_points_count = instance_points_mask.sum(dim=1) # (N,)
    assert torch.all(instance_points_count>1), 'Some instances are not assigned to any fine points.'
    # print(instance_points_count.clone().detach().cpu().numpy())
    for id in torch.arange(instances_number):
        # mask_count = instance_points_count[id]<K
        if instance_points_count[id]<K:
            out_instance_f_points_indices[id][:instance_points_count[id]] = \
                instance_f_points_indices[id][instance_points_mask[id]]
        else:
            valid_indices = torch.multinomial(instance_points_mask[id].float(), K)
            out_instance_f_points_indices[id] = instance_f_points_indices[id][valid_indices]
    
    return out_instance_f_points_indices

@torch.no_grad
def sample_adaptive_k_instance_points(
    instance_f_points_indices: torch.Tensor,
    max_point_index: int,
    K: int,
    sample_ratio: float = 0.5
):
    # todo: test the module
    instances_number, points_number = instance_f_points_indices.shape
    assert K<=points_number, 'K should be less than the number of points.'
    out_instance_f_points_indices = torch.zeros((instances_number, K), dtype=torch.int64).to(instance_f_points_indices.device)
    out_instance_f_points_indices.fill_(max_point_index)
    instance_points_mask = instance_f_points_indices< max_point_index # (M,K)
    instance_points_count = instance_points_mask.sum(dim=1) # (M,)
    assert torch.all(instance_points_count>1), 'Some instances are not assigned to any fine points.'
    
    for scene_id in torch.arange(instances_number):
        sampling_number = max(int(instance_points_count[scene_id]*sample_ratio),1)
        valid_indices = torch.multinomial(
            instance_points_mask[scene_id].float(), min(sampling_number, K))
        out_instance_f_points_indices[scene_id] = instance_f_points_indices[scene_id][valid_indices]
    
    return out_instance_f_points_indices

def extract_instance_f_feats(
    fused_feats_dict:dict,
    instances_knn_dict: dict,
    batch_size: int,
):
    '''
    replace the f_feats in instances_knn_dict with the fused f_feats

    fused_feats_dict:
        - features: (P, C), fine-level features
        - features_batch: (B+1,), features batch ranges
    instances_knn_dict:
        - instances_batch: (N, 1), instance batch
        - instances_f_indices: (N, T), fine points indices
    '''
    features = fused_feats_dict['feats_f']
    features_batch = fused_feats_dict['feats_f_batch']
    fused_instance_knn_feats = [] # (N, T, C)
    
    for scene_id in torch.arange(batch_size):
        scene_mask = instances_knn_dict['instances_batch']==scene_id
        instance_knn_indices = instances_knn_dict['instances_f_indices'][scene_mask,:] # (N_i, T)
        scene_features = features[features_batch[scene_id]:features_batch[scene_id+1],:] # (P_i, C)
        padded_scene_features = torch.cat(
            [scene_features, torch.zeros((1, scene_features.shape[1])).to(scene_features.device)], dim=0) # (P_i+1, C)
        
        # instance_valid = instance_knn_indices<scene_features.shape[0] # (N_i, T)
        instance_knn_features = padded_scene_features[instance_knn_indices] # (N_i, T, C)
        fused_instance_knn_feats.append(instance_knn_features)
    
    fused_instance_knn_feats = torch.cat(fused_instance_knn_feats, dim=0) # (N, T, C)
    instances_knn_dict['instance_f_feats'] = fused_instance_knn_feats

    return instances_knn_dict
    
