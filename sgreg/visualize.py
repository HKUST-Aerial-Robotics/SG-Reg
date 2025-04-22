import os 
import numpy as np
import open3d as o3d
import rerun as rr
import argparse, json
from scipy.spatial.transform import Rotation as R

from sgreg.dataset.scene_graph import load_processed_scene_graph, transform_scene_graph
from sgreg.utils.io import read_pred_nodes, read_dense_correspondences

from os.path import join as osp

def render_point_cloud(entity_name:str,
                        cloud:o3d.geometry.PointCloud, 
                        radius=0.1,
                       color=None):
    """
    Render a point cloud with a specific color and point size.
    """
    if color is not None:
        viz_colors = color
    else:   
        viz_colors = np.asarray(cloud.colors)

    rr.log(entity_name,
           rr.Points3D(
               np.asarray(cloud.points),
               colors=viz_colors,
               radii=radius,
           )
           )

def render_node_centers(entity_name:str,
                        nodes:dict,
                        radius=0.1,
                        color=[0,0,0]):
    """
    Render the centers of nodes in the scene graph.
    """
    
    centers = []
    semantic_labels = []
    for node in nodes.values():
        if isinstance(node, o3d.geometry.OrientedBoundingBox):
            centers.append(node.center)
        else:
            centers.append(node.cloud.get_center())
        semantic_labels.append(node.label)
    centers = np.array(centers)
    if color is None:
        raise NotImplementedError('todo: color by node pcd')
    else:
        viz_colors = color
    rr.log(entity_name,
           rr.Points3D(
               centers,
               colors=viz_colors,
               radii=radius,
               labels=semantic_labels,
               show_labels=False
           )
           )

def render_node_bboxes(entity_name:str,
                       nodes:dict,
                       show_labels:bool=True,
                       radius=0.01):
    
    for idx, node in nodes.items():
        rr.log('{}/{}'.format(entity_name,idx),
               rr.Boxes3D(half_sizes=0.5*node.box.extent,
                          centers=node.box.center,
                          radii=radius,
                          labels=node.label,
                          show_labels=show_labels,
                          colors=None)
               )
               
    
def render_semantic_scene_graph(scene_name:str,
                                scene_graph:dict,
                                voxel_size:float=0.05,
                                origin:np.ndarray=np.eye(4),
                                box:bool=False
                                ):
    render_point_cloud(scene_name+'/global_cloud',
                       scene_graph['global_cloud'],
                       voxel_size)
    render_node_centers(scene_name+'/centroids',
                        scene_graph['nodes'])
    
    if box:
        render_node_bboxes(scene_name+'/nodes',
                            scene_graph['nodes'],
                            show_labels=True)
        
    quad = R.from_matrix(origin[:3,:3]).as_quat()
    rr.log(scene_name+'/local_origin',
            rr.Transform3D(translation=origin[:3,3],
                            quaternion=quad)
            )
        
def render_correspondences(entity_name:str,
                           src_points:np.ndarray,
                           ref_points:np.ndarray,
                           transform:np.ndarray=None,
                           gt_mask:np.ndarray=None,
                           radius=0.01):
    
    N = src_points.shape[0]
    assert N==ref_points.shape[0], 'src and ref points should have the same number of points'
    line_points = []
    line_colors = []

    for i in range(N):
        src = src_points[i]
        ref = ref_points[i]
        if transform is not None:
            src = transform[:3,:3] @ src + transform[:3,3]
            
        if gt_mask[i]:
            line_colors.append([0,255,0])
        else:
            line_colors.append([255,0,0])
    
        line_points.append([src,ref])
        
    
    line_points = np.concatenate(line_points,axis=0)
    line_points = line_points.reshape(-1,2,3)
    line_colors = np.array(line_colors)
    rr.log(entity_name,
           rr.LineStrips3D(line_points,
                           radii=radius,
                           colors=line_colors)
           )

def render_registration(entity_name:str,
                        src_cloud:o3d.geometry.PointCloud,
                        ref_cloud:o3d.geometry.PointCloud,
                        transform:np.ndarray):
    
    src_cloud.transform(transform)
    src_points = np.asarray(src_cloud.points)
    ref_points = np.asarray(ref_cloud.points)
    src_color = [0,180,180]
    ref_color = [180,180,0]
    rr.log(entity_name+'/src',
           rr.Points3D(src_points,
                       colors=src_color,
                       radii=0.01)
           )
    rr.log(entity_name+'/ref',
           rr.Points3D(ref_points,
                       colors=ref_color,
                       radii=0.01)
           )
           

def get_parser_args():
    def float_list(string):
        return [float(x) for x in string.split(',')]
    
    parser = argparse.ArgumentParser(description='Visualize scene graph')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='path to dataset root')
    parser.add_argument('--src_scene', type=str, default='scene0108_00c',
                        help='source scene name')
    parser.add_argument('--ref_scene', type=str, default='scene0108_00a',
                        help='reference scene name')
    parser.add_argument('--result_folder', type=str, default='output/sgnet_scannet_0080',
                        help='a relative path to the result folder')
    parser.add_argument('--viz_mode', type=int, required=True,
                        help='0: no viz, 1: local viz, 2: remote viz, 3: save rrd')
    parser.add_argument('--remote_rerun_add', type=str, help='IP:PORT')
    parser.add_argument('--find_gt', action='store_true',
                        help='align the scene graphs for bettter visualization')
    parser.add_argument('--augment_transform', action='store_true',
                        help='only enable it for ScanNet scenes.')
    parser.add_argument('--viz_translation', type=json.loads,
                        default='[0,0,0]',
                        help='translation to viz the scene graphs')
    
    parser.add_argument('--voxel_size', type=float, default=0.05,
                        help='voxel size for downsampling')    
    
    return parser.parse_args()

if __name__=='__main__':
    print('*'*60)
    print('This script reads the data association and registration results.')
    print('*'*60)
    
    ############ Args ############
    args = get_parser_args()
    SPLIT = 'val'
    RESULT_FOLDER = osp(args.dataroot, 
                        args.result_folder, 
                        '{}-{}'.format(args.src_scene,args.ref_scene))
    print('Visualize {}-{} scene graph'.format(args.src_scene,args.ref_scene))
    ##############################

    # Load scene graphs
    src_sg = load_processed_scene_graph(osp(args.dataroot,SPLIT,args.src_scene))
    ref_sg = load_processed_scene_graph(osp(args.dataroot,SPLIT,args.ref_scene))
    src_cloud = o3d.geometry.PointCloud(src_sg['global_cloud'].points)
    
    # Load SG-Reg results
    if os.path.exists(RESULT_FOLDER):
        node_matches = read_pred_nodes(osp(RESULT_FOLDER,'node_matches.txt'))
        point_correspondences = read_dense_correspondences(osp(RESULT_FOLDER,'corr_src.ply'),
                                                            osp(RESULT_FOLDER,'corr_ref.ply'),
                                                            osp(RESULT_FOLDER,'point_matches.txt'))
        pred_transformation = np.loadtxt(osp(RESULT_FOLDER,'svds_estimation.txt'))
    
    # Transform for better visualization
    transform = np.eye(4)
    if args.find_gt:
        if 'ScanNet' in args.dataroot:
            assert False, 'ScanNet scenes are already aligned. Remove the option.'
        gt_dir = osp(args.dataroot, 'gt', '{}-{}.txt'.format(args.src_scene,args.ref_scene))
        assert os.path.exists(gt_dir), 'gt file not found'
        gt = np.loadtxt(gt_dir)
        print('Load gt transformations from ',gt_dir)
        transform = gt

    transform[:3,3] += args.viz_translation
    transform_scene_graph(src_sg, transform)

    # Stream to rerun
    rr.init("SGReg")
    render_semantic_scene_graph('src',src_sg,
                                args.voxel_size,
                                transform)
    render_semantic_scene_graph('ref',ref_sg,
                                args.voxel_size,
                                np.eye(4),
                                True)
    
    if args.augment_transform:
        if 'RioGraph' in args.dataroot: 
            assert False, 'RIO dataset does not require augment transform. Remove the option.'
        drift_dir = osp(RESULT_FOLDER,'gt_transform.txt')
        transform = np.loadtxt(drift_dir)
        assert os.path.exists(drift_dir), 'drift transform file not found'
        src_cloud.transform(np.linalg.inv(transform))
        transform[:3,3] += args.viz_translation
          
    if os.path.exists(RESULT_FOLDER):
        render_correspondences('node_matches',
                            node_matches['src_centroids'],
                            node_matches['ref_centroids'],
                            transform=transform,
                            gt_mask=node_matches['gt_mask'])
        render_correspondences('point_correspondences',
                            point_correspondences['src_corrs'],
                            point_correspondences['ref_corrs'],
                            transform=transform,
                            gt_mask=point_correspondences['corr_masks'])
        render_registration('registration',
                            src_cloud,
                            ref_sg['global_cloud'],
                            pred_transformation)
    
        # Eval message
        msg = '{}/{} TP node matches, '.format(node_matches['gt_mask'].sum(),
                                            node_matches['gt_mask'].shape[0])
        msg += 'Inlier ratio: {:.2f}%'.format(point_correspondences['corr_masks'].mean()*100)
        print(msg)
    
    # Render on rerun
    if args.viz_mode==1:
        rr.spawn()
    elif args.viz_mode==2:
        assert args.remote_rerun_add is not None, \
            'require a remote address for rendering, (eg. 143.89.38.169:9876)'
        print('--- Render rerun at a remote machine ',args.remote_rerun_add, '---')
        rr.connect_tcp(args.remote_rerun_add)
    elif args.viz_mode==3:
        rr.save(osp(RESULT_FOLDER,'result.rrd'))
        print('Save rerun data to ',osp(RESULT_FOLDER,'result.rrd'))
    else:
        print('No visualization')