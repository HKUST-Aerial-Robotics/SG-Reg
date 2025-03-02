import os, glob
import sys
import torch

def read_frame_data(data_dir:str):
    print('  ----- Reading data from {} -----'.format(os.path.basename(data_dir)[:12]))
    data_dict = torch.load(data_dir)
    
    features = data_dict['features'] # (N,128)
    xyz = data_dict['xyz'] # (P,3)
    instances = data_dict['instances'] # (P,)
    N = features.shape[0]
    P = xyz.shape[0]
    
    print('  Read {} features and {} points'.format(N, P))
    
    print('  Demonstrate how to read node-wise points:')
    for i in range(N):
        masks = instances==i
        node_points = xyz[masks]
        print('  Node {} has {} points'.format(i, masks.sum()))
        assert masks.sum() > 0, 'Node {} has no points'.format(i)

def process_scene_sequence(scene_folder:str):
    print('************ Processing {} ************'.format(os.path.basename(scene_folder)))
    frame_files = glob.glob(os.path.join(scene_folder, 'frame-*.pth'))
    frame_files = sorted(frame_files)
    
    # Read the features and points for each frame
    for frame_file in frame_files:
        read_frame_data(frame_file)

if __name__=='__main__':
    ################# SET ARGS #################
    RAG_DATAROOT = '/data2/ScanNetRag'
    ############################################
    
    scene = 'scene0025_00'
    process_scene_sequence(os.path.join(RAG_DATAROOT, 'val', scene))
    
    
