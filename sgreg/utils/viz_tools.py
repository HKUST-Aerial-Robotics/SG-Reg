import open3d as o3d
import numpy as np
import torch

def build_o3d_points(points:np.ndarray,colors:np.ndarray=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def build_correspondences_lines(corres_s, corres_t, corres_pos=None):
    line_points = np.concatenate([corres_s,corres_t],axis=0) # (2C,3)
    line_indices = np.stack([np.arange(len(corres_s)),np.arange(len(corres_s),2*len(corres_s))],axis=0) # (2,C)
    line_colors = np.zeros((corres_s.shape[0],3))
    if corres_pos is None:
        line_colors += np.array([0,0,1])
    else:
        line_colors[corres_pos] = np.array([0,1,0])
        line_colors[~corres_pos] = np.array([1,0,0])
    
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(line_points),
        lines = o3d.utility.Vector2iVector(line_indices.T))
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set

def build_instance_centroids(graph:dict,pos_indices=np.array([]),neg_indices=np.array([]),radius=0.1):
    centroids = []
    for idx, instance in graph['nodes'].items():
        centroid = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        if instance.idx in pos_indices:
            centroid.paint_uniform_color(np.array([0,1,0]))
        elif instance.idx in neg_indices:
            centroid.paint_uniform_color(np.array([1,0,0]))
        # else:
        #     continue
        centroid.translate(instance.cloud.get_center())
        centroids.append(centroid)

    return centroids

def build_centroids_from_points(points:np.ndarray,radius=0.1):
    centroids = []
    for point in points:
        centroid = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        centroid.paint_uniform_color(np.array([0,1,0]))
        centroid.translate(point)
        centroids.append(centroid)
    return centroids

def generate_instance_color(instances):
    # ref_instances = data_dict['ref_points_f_instances']
    ref_instance_list = torch.unique(instances)
    instance_colors = np.zeros((instances.shape[0],3))
    for idx in ref_instance_list:
        color = np.random.uniform(0,1,3)
        instance_mask = instances==idx
        instance_colors[instance_mask] = color
    return instance_colors