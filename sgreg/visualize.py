import os 
import numpy as np
import open3d as o3d
import rerun as rr

from sgreg.dataset.scene_graph import load_processed_scene_graph

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

if __name__=='__main__':
    print('*'*60)
    print('This script reads the data association and registration results.')
    print('*'*60)
    
    ############ Args ############
    DATAROOT = '/data2/RioGraph'
    SPLIT = 'val'
    SRC_SCENE = 'scene0108_00c'
    REF_SCENE = 'scene0108_00a'
    RESULT_FOLDER = osp(DATAROOT, 'output', 'sgnet_init_layer2')
    VIZ_MODE = 2 # 0: no viz, 1: local viz, 2: remote viz, 3: save rrd
    RERUN_ADDRESS = '143.89.38.169:9876'
    ##############################

    # Load scene graphs
    src_sg = load_processed_scene_graph(osp(DATAROOT,SPLIT, SRC_SCENE))

    # Visualize
    rr.init("SGReg")
    render_point_cloud('src/global_cloud',
                       src_sg['global_cloud'],
                       0.05)
    
    if VIZ_MODE==1:
        rr.spawn()
    elif VIZ_MODE==2:
        print('Render rerun at a remote machine: ',RERUN_ADDRESS)
        rr.connect_tcp(RERUN_ADDRESS)
    elif VIZ_MODE==3:
        rr.save(osp(RESULT_FOLDER,'result.rrd'))
    else:
        print('No visualization')