import os
import torch
import torch.utils.data
import pandas as pd
import numpy as np 
import open3d as o3d
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from sgreg.ops.transformation import apply_transform
from sgreg.utils.utils import read_scan_pairs

def associate_points_f_instances(points_f:np.ndarray,xyz:np.ndarray,instances:np.ndarray,radius=0.2):
    P = points_f.shape[0]
    raw_pcd = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(xyz)
    )
    raw_pcd_kdtree = o3d.geometry.KDTreeFlann(raw_pcd)
    points_f_vec = o3d.utility.Vector3dVector(points_f)
    points_f_instances = -np.ones(points_f.shape[0])
    count = 0
    
    for i in np.arange(P):
        [k, idx, _] = raw_pcd_kdtree.search_radius_vector_3d(points_f_vec[i], radius)
        # [k, idx, _] = raw_pcd_kdtree.search_knn_vector_3d(o3d.utility.Vector3dVector(points_f[i,:]), 1)
        if k>0:
            points_f_instances[i] = instances[idx[0]]
            count +=1
            dist = np.linalg.norm(points_f[i] - xyz[idx[0]])
            assert dist<=radius
    assert points_f_instances.shape[0] == points_f.shape[0]
    # print('{}/{} fine points find instances label'.format(count,points_f.shape[0]))
    return points_f_instances

def check_instance_points(idxs:torch.Tensor, instances:torch.Tensor, debug_prefix=''):
    '''
    Input,
        - idxs: (N,), instance index
        - instances: (X,), instance label of each point
    '''
    assert isinstance(instances,torch.Tensor) and isinstance(idxs,torch.Tensor), 'input should be tensor'
    N = idxs.shape[0]
    X = instances.shape[0]
    assert N>0 and X>0, 'empty input'
    mask = idxs.unsqueeze(1).repeat(1,X) == instances.unsqueeze(0).repeat(N,1) # (N,X)
    instance_count = mask.sum(dim=1) # (N,)
    if torch.any(instance_count<1):
        invalid_instance = idxs[instance_count==0]
        invalid_instance = invalid_instance.cpu().numpy()
        # print('{} Instance {} contain no points'.format(debug_prefix,
        #                                                 invalid_instance))
        # print('{} instances, {} points'.format(N,X))
        print('{} find {} invalid instances'.format(debug_prefix, 
                                                    invalid_instance.shape[0]))
        print(invalid_instance)
        assert False, '{} contain {} invalid instances'.format(debug_prefix, invalid_instance.shape[0])


def filter_invalid_match(matches:torch.Tensor,
                         src_labels:list,
                         ref_labels:list,IGNORE_TYPES:list):
    '''
    Input,
        - matches: tensor of shape (num_matches,2), [src_id,tar_id]
        - src_labels: list of string
        - ref_labels: list of string
    '''
    
    src_mask = torch.tensor([src_labels[idx] not in IGNORE_TYPES for idx in matches[:,0]]) # (num_matches,)
    ref_mask = torch.tensor([ref_labels[idx] not in IGNORE_TYPES for idx in matches[:,1]])
    
    assert src_mask.shape[0]==matches.shape[0]
    
    valid = src_mask & ref_mask
    return matches[valid]

class ScenePairDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot:str, split:str, conf) -> None:
        super().__init__()
        self.dataroot = dataroot
        self.split = split
        self.conf = conf
        self.metadata = []
        
        if 'load_transform' in conf.dataset:
            self.load_transform = conf.dataset.load_transform
        else:
            self.load_transform = False    
        
        scene_pairs = read_scan_pairs(os.path.join(dataroot,'splits','{}.txt'.format(self.split)))
        for pair in scene_pairs:
            self.metadata.append({'src_scan':pair[0],'ref_scan':pair[1],'iou':0.5})
        
        print('Init {} {} set {} scene pairs with {} samples'.format(self.dataroot,
                                                                     self.split,
                                                                     len(scene_pairs),
                                                                     len(self)))

    def __len__(self):
        return len(self.metadata)
    
    def load_matches_data(self,
                          dir:str,
                          name2idx_s:str,
                          name2idx_r:str):
        '''
        instance_matches: (num_matches,3), [src_id,tar_id,iou]
        ious_map: dict, {'iou':iou_mat, 
                        'src_names':src_names, 
                        'tar_names':tar_names}
        global_iou: scaler
        '''
        assert os.path.exists(dir), 'gt match file NOT FOUND.'
        
        match_data = torch.load(dir)
        if len(match_data)==4:
            (instance_matches, ious_map, _, global_iou) = match_data
        elif len(match_data)==3:
            (instance_matches, ious_map, global_iou) = match_data
        else:
            raise ValueError('match data not correct')
        assert ious_map['iou'] is not None, '{} ious_map is None'.format(dir)
        
        # (instance_matches, ious_map, _, global_iou) = torch.load(dir)
        # torch.load(os.path.join(self.graph_rootdir,'matches',scan,'matches_ab.pth'))
        # assert point_matches.shape[1] == 4
        if instance_matches.shape[0]<1: # or point_matches.shape[0] < 1: 
            return None, None
        valid_matches = instance_matches[:,2] > self.conf.dataset.min_iou
        instance_matches = instance_matches[valid_matches,:2].int()
        instance_matches[:,0] = name2idx_s[instance_matches[:,0]]
        instance_matches[:,1] = name2idx_r[instance_matches[:,1]]
        if valid_matches.sum() < 1: 
            print('No valid gt matches in {}'.format(dir))
        assert valid_matches.sum()>0, '{} contain empty gt matches'.format(dir)
        assert instance_matches.shape[0]>0, 'contain empty gt matches'
        assert instance_matches.min()>=0, 'contain invalid node name'
        
        #
        instances_iou_mat = ious_map['iou']
        src_names = torch.tensor(ious_map['src_names']).long()
        ref_names = torch.tensor(ious_map['tar_names']).long()
        if not isinstance(instances_iou_mat,torch.Tensor):
            instances_iou_mat = torch.tensor(instances_iou_mat).float()
        assert torch.all(name2idx_s[src_names]>=0) and torch.all(name2idx_r[ref_names]>=0), \
            'src or ref nodes indices names inconsistent!'
        
        # point_matches[:,0] = name2idx_s[point_matches[:,0].int()]
        # point_matches[:,1] = name2idx_r[point_matches[:,1].int()]
        return instance_matches, instances_iou_mat.to(torch.float32)
    
    def load_scene_graph(self,scan_dir,aug_transform=False):
        instance_nodes = {} # {idx:node_info}
        
        # Nodes
        nodes_data = pd.read_csv(os.path.join(scan_dir,'nodes.csv'))
        max_node_id = 0
        gt_drift = np.eye(4)
        if aug_transform:
            gt_drift[:3,:3] = R.from_euler('z', np.random.uniform(-180,180,1), degrees=True).as_matrix()
            gt_drift[:3,3] = np.random.uniform(-10.0,10.0,3)
        gt_drift = torch.tensor(gt_drift).float()
        
        for idx, label, score, center, quat, extent, cloud_dir in zip(nodes_data['node_id'],
                                                                      nodes_data['label'],
                                                                      nodes_data['score'],
                                                                      nodes_data['center'],
                                                                      nodes_data['quaternion'],
                                                                      nodes_data['extent'],
                                                                      nodes_data['cloud_dir']):
            centroid = np.fromstring(center, dtype=float, sep=',')
            quaternion = np.fromstring(quat, dtype=float, sep=',') # (x,y,z,w)
            extent = np.fromstring(extent, dtype=float, sep=',')
            
            # incorporate random translation and rotation
            quaternion_ = torch.tensor(R.from_quat(quaternion).as_matrix()).float()
            if aug_transform:
                centroid = apply_transform(torch.tensor(centroid).float(),gt_drift)
            else:
                centroid = torch.tensor(centroid).float()
            
            if np.isnan(extent).any() or np.isnan(quaternion).any() or np.isnan(centroid).any() or np.isnan(idx):
                continue
            if '_' in label: label = label.replace('_',' ')
            instance_nodes[idx] = {'idx':idx,
                                'label':label,
                                'score':score,
                                'centroid':centroid,
                                'quaternion':quaternion_,
                                'box_dim':torch.from_numpy(extent).float()
                                }
            
            if idx>max_node_id:
                max_node_id = idx

        # Semantic embeddings
        semantic_dir = os.path.join(scan_dir,'semantic_embeddings.pth')
        if self.conf.dataset.online_bert: 
            pre_semantic_data = None
        elif os.path.exists(semantic_dir):
            pre_semantic_data = torch.load(semantic_dir)
        else:
            pre_semantic_data = None
            print('Set onliner bert to false! \n No semantic embeddings found in {}'.format(scan_dir))

        # Instance Point Cloud
        xyzi = torch.load(os.path.join(scan_dir,'xyzi.pth'))
        instances = xyzi[:,-1].to(torch.int32)
        xyz = xyzi[:,:3].to(torch.float32)
        if aug_transform:
            xyz = apply_transform(xyz,gt_drift)
        
        # Edges
        edges_data = torch.load(os.path.join(scan_dir,'edges.pth'))
        edges = edges_data['edges'].long()
        global_edges_data = edges_data['global_edges']
        if global_edges_data.shape != torch.Size([0]):
            assert global_edges_data.shape[1]==3, '{} global edges data not correct'.format(scan_dir)
            global_edges_valid = global_edges_data[:,2]>self.conf.dataset.global_edges_dist
            global_edges = global_edges_data[global_edges_valid,:2].long()
        else:
            global_edges = torch.tensor([]).long()
            
        if len(instance_nodes) >1: # and edges.shape[0] >1:
            out= {'nodes':instance_nodes,       
                    'pre_semantic_data': pre_semantic_data,
                    'edges':edges,
                    'global_edges':global_edges,
                    'max_node':max_node_id, 
                    'xyz':xyz, 
                    'instances':instances,
                    'augmented_drift':gt_drift,
                    }
            return out
        else: return None

    def construct_deep_graph(self,unproc_graph): 
        '''
        instance_names: [0,3,5,...max_instance_idx], the index is not continuous
        idxs: [0,1,2,3...num_instances-1], the index is continuous
        '''       
        # pyg graph
        labels = [] # list of string
        N = len(unproc_graph['nodes'])
        idx2name = torch.tensor(-1).repeat(N).int() # (N,), -1 is invalid
        name2idx = torch.tensor(-1).repeat(unproc_graph['max_node']+1).int() # (max_node_name+1,)
        
        boxes = torch.zeros((N,3))  #((unproc_graph['max_node']+1, 3))
        centroids = torch.zeros((N, 3))
        valid = torch.zeros(N).int()
    
        # nodes
        idx = 0
        for inst_name, instance in unproc_graph['nodes'].items():
            # assert instance['idx']>0, 'node idx should be positive'
            labels.append(instance['label'])
            # quaternion_ = torch.tensor(instance['quaternion'])
            boxes[idx] = instance['box_dim']
            centroids[idx] = instance['centroid']
            name2idx[int(inst_name)] = idx
            idx2name[idx] = int(inst_name)
            valid[idx] = 1
            idx += 1
        assert idx2name.min()>=0, 'contain invalid node name'

        # encode batch semantics
        assert len(labels) == N, 'semantic number not match'
        if unproc_graph['pre_semantic_data'] is not None:
            assert unproc_graph['pre_semantic_data']['instance_idxs'].shape[0] == N, 'semantic instance idx not match'
            assert (idx2name - unproc_graph['pre_semantic_data']['instance_idxs']).sum() == 0, 'semantic instance idx not match'
            semantic_embeddings = unproc_graph['pre_semantic_data']['semantic_embeddings']
        
        # edges 
        edge_indices_ud = name2idx[unproc_graph['edges']] # (e,2)
        assert edge_indices_ud.min()>=0, 'contain invalid node name in the edge'
        edge_indices_bd = torch.cat([edge_indices_ud,edge_indices_ud[:,[1,0]]],dim=0) # (2e,2)        
        if unproc_graph['global_edges'].shape[0]>0:
            global_edge_indices = name2idx[unproc_graph['global_edges']] # (e,2)
        else:
            global_edge_indices = torch.tensor([]).long()

        
        #
        xyz = unproc_graph['xyz'] # (X, 3)
        instances = unproc_graph['instances']
        instances = name2idx[instances] # (X,)
        
        out_dict = {
            'labels':labels,
            'boxes':boxes,
            'centroids':centroids,
            'edge_indices': edge_indices_bd,
            'global_edge_indices': global_edge_indices,
            'idx2name':idx2name,
            'name2idx':name2idx,         
            'xyz':xyz,
            'instances':instances
        }
        
        if unproc_graph['pre_semantic_data'] is not None:
            out_dict['semantic_embeddings'] = semantic_embeddings
        
        return out_dict
        

    def __getitem__(self, index):
        metadata = self.metadata[index]

        data_dict = {}
        if self.load_transform: src_aug_transform = False
        else: src_aug_transform = True
        src_graph_unproc = self.load_scene_graph(os.path.join(self.dataroot,self.split,metadata['src_scan']),
                                                    aug_transform=src_aug_transform)
        ref_graph_unproc = self.load_scene_graph(os.path.join(self.dataroot,self.split,metadata['ref_scan']),
                                                    aug_transform=False)
        pair_name = '{}-{}'.format(metadata['src_scan'],metadata['ref_scan'])            
        
        assert src_graph_unproc is not None and ref_graph_unproc is not None, 'Invalid pair {}, {}'.format(metadata['src_scan'],metadata['ref_scan'])
        src_dict = self.construct_deep_graph(src_graph_unproc)
        ref_dict = self.construct_deep_graph(ref_graph_unproc)

        # Gt data
        if self.load_transform:
            gt_transform = np.loadtxt(os.path.join(self.dataroot,'gt',pair_name+'.txt'))
            gt_transform = torch.from_numpy(gt_transform).float()
        else: # augmented transform
            assert 'augmented_drift' in src_graph_unproc, 'augment drift transformation if not prior gt transformation'
            gt_transform = torch.linalg.inv(src_graph_unproc['augmented_drift'])
            
        instance_matches, instances_iou_mat = self.load_matches_data(
            os.path.join(self.dataroot, 'matches', '{}.pth'.format(pair_name)),
            src_dict['name2idx'],
            ref_dict['name2idx'])
        
        # Pack data
        ## Data that will be collate in list
        data_dict['src_scan'] = metadata['src_scan'] # string
        data_dict['ref_scan'] = metadata['ref_scan'] # string
        data_dict['instance_matches'] = instance_matches # tensor
        data_dict['instance_ious'] = instances_iou_mat # tensor
        ## 
        data_dict['transform'] = gt_transform # (4,4)
        
        ## Data that 
        data_dict['src_points'] = src_dict.pop('xyz') # (P,3)
        data_dict['ref_points'] = ref_dict.pop('xyz') # (Q,3)
        data_dict['src_instances'] = src_dict.pop('instances') # (P,)
        data_dict['ref_instances'] = ref_dict.pop('instances') # (Q,)
        data_dict['src_feats'] = torch.ones_like(data_dict['src_points'][:,0]).reshape(-1,1) # (P,1)
        data_dict['ref_feats'] = torch.ones_like(data_dict['ref_points'][:,0]).reshape(-1,1) # (Q,1)
        data_dict['src_graph'] = src_dict
        data_dict['ref_graph'] = ref_dict
            
        assert data_dict['src_graph']['edge_indices'].shape[1] == 2    
        # print('get dataset item!')
        return data_dict

if __name__=='__main__':    
    # dataroot = '/data2/ScanNetGraph'
    split = 'val'
    conf = OmegaConf.load('/home/cliuci/code_ws/SceneGraphNet/config/realsense.yaml')
    conf.backbone.init_radius = conf.backbone.base_radius * conf.backbone.init_voxel_size
    
    dataset = ScenePairDataset(conf.dataset.dataroot,split,conf)
    
    data_dict = dataset[0]
    print('data dict keys:',data_dict.keys())
    print('scene graph keys:',data_dict['src_graph'].keys())
    print(data_dict['ref_graph']['edge_indices'].shape)
    print(data_dict['ref_graph']['global_edge_indices'].shape)   
    print(data_dict['src_scan'],data_dict['ref_scan'])
    # print(data_dict['instance_matches'])
    # print(data_dict['instance_ious'])

