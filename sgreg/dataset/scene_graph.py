import os
import torch
import pandas as pd
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from open3d.geometry import OrientedBoundingBox as OBB

class Instance:
    def __init__(self,idx:int,
                 cloud:o3d.geometry.PointCloud|np.ndarray,
                 label:str,
                 score:float):
        self.idx = idx
        self.label = label
        self.score = score
        self.cloud = cloud
        self.cloud_dir = None
    def load_box(self,box:o3d.geometry.OrientedBoundingBox):
        self.box = box

def load_raw_scene_graph(folder_dir:str,
                     voxel_size:float=0.02,
                     ignore_types:list=['ceiling']):
    ''' graph: {'nodes':{idx:Instance},'edges':{idx:idx}}
    '''
    # load scene graph
    nodes = {}
    boxes = {}
    invalid_nodes = []
    xyzi = []
    global_cloud = o3d.geometry.PointCloud()
    # IGNORE_TYPES = ['floor','carpet','wall']
    # IGNORE_TYPES = ['ceiling']
    
    # load instance boxes
    with open(os.path.join(folder_dir,'instance_box.txt')) as f:
        count=0
        for line in f.readlines():
            line = line.strip()
            if'#' in line:continue
            parts = line.split(';')
            idx = int(parts[0])
            center = np.array([float(x) for x in parts[1].split(',')])
            rotation = np.array([float(x) for x in parts[2].split(',')])
            extent = np.array([float(x) for x in parts[3].split(',')])
            o3d_box = o3d.geometry.OrientedBoundingBox(center,rotation.reshape(3,3),extent)
            o3d_box.color = (0,0,0)
            # if'nan' in line:invalid_nodes.append(idx)
            if 'nan' not in line:
                boxes[idx] = o3d_box
                # nodes[idx].load_box(o3d_box)
                count+=1
        f.close()
        print('load {} boxes'.format(count))    
        
    # load instance info
    with open(os.path.join(folder_dir,'instance_info.txt')) as f:
        for line in f.readlines():
            line = line.strip()
            if'#' in line:continue
            parts = line.split(';')
            idx = int(parts[0])
            if idx not in boxes: continue
            label_score_vec = parts[1].split('(')
            label = label_score_vec[0]
            score = float(label_score_vec[1].split(')')[0])
            if label in ignore_types: continue
            # print('load {}:{}, {}'.format(idx,label,score))
            
            cloud = o3d.io.read_point_cloud(os.path.join(folder_dir,'{}.ply'.format(parts[0])))
            cloud = cloud.voxel_down_sample(voxel_size)
            xyz = np.asarray(cloud.points)
            # if xyz.shape[0]<50: continue
            xyzi.append(np.concatenate([xyz,idx*np.ones((len(xyz),1))],
                                       axis=1))
            global_cloud = global_cloud + cloud
            nodes[idx] = Instance(idx,cloud,label,score)
            nodes[idx].cloud_dir = '{}.ply'.format(parts[0])
            nodes[idx].load_box(boxes[idx])

        f.close()
        print('Load {} instances '.format(len(nodes)))
    if len(xyzi)>0:
        xyzi = np.concatenate(xyzi,axis=0)
        
    return {'nodes':nodes,
            'edges':[],
            'global_cloud':global_cloud, 
            'xyzi':xyzi}

def load_processed_scene_graph(scan_dir:str):
    
    instance_nodes = {} # {idx:node_info}
    
    # Nodes
    nodes_data = pd.read_csv(os.path.join(scan_dir,'nodes.csv'))
    max_node_id = 0

    for idx, label, score, center, quat, extent, _ in zip(nodes_data['node_id'],
                                                                    nodes_data['label'],
                                                                    nodes_data['score'],
                                                                    nodes_data['center'],
                                                                    nodes_data['quaternion'],
                                                                    nodes_data['extent'],
                                                                    nodes_data['cloud_dir']):
        centroid = np.fromstring(center, dtype=float, sep=',')
        quaternion = np.fromstring(quat, dtype=float, sep=',') # (x,y,z,w)
        rot = R.from_quat(quaternion)
        extent = np.fromstring(extent, dtype=float, sep=',')
                
        if np.isnan(extent).any() or np.isnan(quaternion).any() or np.isnan(centroid).any() or np.isnan(idx):
            continue
        if '_' in label: label = label.replace('_',' ')
        instance_nodes[idx] = Instance(idx, None, label, score)
        instance_nodes[idx].load_box(OBB(centroid, 
                                         rot.as_matrix(),
                                         extent))
        
        if idx>max_node_id:
            max_node_id = idx

    # Instance Point Cloud
    xyzi = torch.load(os.path.join(scan_dir,'xyzi.pth')).numpy()
    instances = xyzi[:,-1].astype(np.int32)
    xyz = xyzi[:,:3].astype(np.float32)
    assert max_node_id == instances.max(), 'Instance ID mismatch'
    assert np.unique(instances).shape[0] == len(instance_nodes), 'Instance ID mismatch'
    
    colors = np.zeros_like(xyz)
    for idx, instance in instance_nodes.items():  
        inst_mask = instances== idx
        assert inst_mask.sum()>0
        inst_color = 255*np.random.rand(3)
        colors[inst_mask] = np.floor(inst_color).astype(np.int32)
        instance.cloud = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(xyz[inst_mask]))
    
    # Global Point Cloud
    global_pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(xyz))
                # colors=o3d.utility.Vector3dVector(colors))  
    global_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print('Load {} instances'.format(len(instance_nodes)))
    
    return {'nodes':instance_nodes,
            'edges':[],
            'global_cloud':global_pcd}
    