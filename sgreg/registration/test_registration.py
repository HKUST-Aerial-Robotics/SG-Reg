import os, sys
import argparse
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

sys.path.append('/home/cliuci/code_ws/LiDAR-Registration-Benchmark')
from sgreg.dataset.generate_gt_association import process_scene, transform_scene_graph, get_geometries
from sgreg.process.viz_match_results import load_matches, load_match_new
from sgreg.utils import read_scans

from sgreg.registration.evaluate import EvalResult, compute_residual

from misc.config import cfg_from_yaml_file, cfg
from misc.registration import fpfh_teaser, fpfh_teaser_new

def save_errors(scans,errors,dir):
    with open(dir,'w') as f:
        f.write('# scan, rot_error, t_error\n')
        for scan, error in zip(scans,errors):
            f.write('{}, {:.3f}, {:.3f}\n'.format(scan,error[0],error[1]))
        f.close()

def compute_instance_residuals():
    msg = ''
    for i_m, pair in enumerate(pred): # compute instance-wise residual
        check_equal = np.sum((gt[:,:2]-pair)==0,axis=1)
        true_positive = check_equal.max()==2
        if pair[0] not in src_graph['nodes'] or pair[1] not in tar_graph['nodes']: continue

        inst_s = src_graph['nodes'][pair[0]]
        inst_t = tar_graph['nodes'][pair[1]]
        if inst_s.label=='floor' or inst_s.label=='carpet':continue
        
        mask = inst_corrs[:,0]==i_m
        residual = compute_residual(A_corr[mask],B_corr[mask],pred_T)
        eval_instances.append({'tp':int(true_positive),'residual':residual})
        
        msg +='{} pair ({}_{},{}_{}) find {} inliners, residual: {:.3f} \n'.format(true_positive,pair[0],inst_s.label,pair[1],inst_t.label,np.sum(mask),residual)
    msg = 'mean residual {:.3f}'.format(compute_residual(A_corr,B_corr,pred_T))
    print(msg)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Registration test')
    parser.add_argument('--graphroot', type=str, default='/data2/ScanNetGraph')
    parser.add_argument('--prediction', type=str, default='lap', help='Prediction folder under graph root')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--global_register', action='store_true', help='Global map registration')
    args = parser.parse_args()
    
    VOXEL_SIZE = 0.05
    cfg_from_yaml_file('/home/cliuci/code_ws/LiDAR-Registration-Benchmark/configs/dataset.yaml', cfg)

    # Enable webrtc streaming
    os.environ.setdefault('WEBRTC_IP', '143.89.46.75')
    os.environ.setdefault('WEBRTC_PORT', '8020')
    o3d.visualization.webrtc_server.enable_webrtc()
    gt_folder= os.path.join(args.graphroot,'matches')

    #
    scans = read_scans(os.path.join(args.graphroot, 'splits', 'val' + '.txt'))
    if args.samples<len(scans):
        sample_scans = np.random.choice(scans,args.samples)    
    else:
        sample_scans = scans
    # sample_scans = ['scene0025_00']
    
    #
    eval_instances =[]
    eval_metrics = EvalResult()
    global_reg_errors = []
    inst_reg_errors = []
    warn_scans = []
    executed_scans = []
    
    for scan in sample_scans:
        scan_dir = os.path.join(args.graphroot,args.split,scan)
        pred_folder = os.path.join(args.graphroot,'pred',args.prediction,'{}'.format(scan))
        if not os.path.exists(pred_folder):
            print('skip {}'.format(scan))
            continue
        print('processing {}'.format(scan))
        out = process_scene(scan_dir,None,0.3,save_nodes=False,compute_matches=False)
        src_graph, tar_graph = out['src'], out['tar'] # load graphs
        gt = load_match_new(os.path.join(gt_folder,scan,'matches.pth')) # (\bar{M},3)
        
        pred = np.loadtxt(os.path.join(pred_folder,'instances.txt'),delimiter=',')
        if pred.ndim ==1:
            pred = [pred.astype(np.int32)]
        else:
            pred = [pair for pair in pred.astype(np.int32)]

        drift_translation = np.random.uniform(-10.0,10.0,3)
        drift_rot = R.from_euler('z', np.random.uniform(-180,180,1), degrees=True)
        drift_T = np.concatenate([drift_rot.as_matrix().squeeze(),drift_translation.reshape(3,1)],axis=1)
        drift_T = np.concatenate([drift_T,np.array([0,0,0,1]).reshape(1,4)],axis=0)
        gt_T = np.linalg.inv(drift_T)
        
        transform_scene_graph(src_graph,drift_T)
        src_geometries = get_geometries(src_graph)
        tar_geometries = get_geometries(tar_graph)

        msg = ''
        corr_lines = []
        src_cloud = o3d.geometry.PointCloud()
        tar_cloud = o3d.geometry.PointCloud()
        src_xyzi = []
        tar_xyzi = []     
        for pair in pred:
            check_equal = np.sum((gt[:,:2]-pair)==0,axis=1)
            true_positive = check_equal.max()==2
            if pair[0] not in src_graph['nodes'] or pair[1] not in tar_graph['nodes']:
                print('[WARN] predicted instances in {} not exist'.format(scan))
                # warn_scans.append(scan)
                continue
            inst_s = src_graph['nodes'][pair[0]]
            inst_t = tar_graph['nodes'][pair[1]]
            # if inst_s.label=='floor' or inst_s.label=='carpet':continue
            src_cloud = src_cloud + inst_s.cloud
            tar_cloud = tar_cloud + inst_t.cloud

            cloud_ds_s = inst_s.cloud.voxel_down_sample(VOXEL_SIZE)
            cloud_ds_t = inst_t.cloud.voxel_down_sample(VOXEL_SIZE)
            n_src = len(cloud_ds_s.points)
            n_tar = len(cloud_ds_t.points)         
            src_xyzi.append(np.concatenate([np.asarray(cloud_ds_s.points),pair[0]*np.ones((n_src,1))],axis=1))
            tar_xyzi.append(np.concatenate([np.asarray(cloud_ds_t.points),pair[1]*np.ones((n_tar,1))],axis=1))
            # continue
            
        src_xyzi = np.concatenate(src_xyzi,axis=0)
        tar_xyzi = np.concatenate(tar_xyzi,axis=0)

        
        # Global map registration with instance correspondences
        pred_T, inliners_map, inliners, A_corr, B_corr, _ = fpfh_teaser_new(src_xyzi,tar_xyzi, pred)
        inliners = inliners_map[inliners]
        assert A_corr.shape[0]==B_corr.shape[0]
        rot_error, t_error = eval_metrics.evaluate(pred_T,gt_T)
        inst_reg_errors.append(np.array([rot_error,t_error]))
        print('[{}] instance map registration, {}/{} inliners, rotation error {:.3f}deg, translation error {:.3f}m'.format(scan,len(inliners),A_corr.shape[0],rot_error,t_error))
        
        if args.global_register:
            pred_T, inliners_map, inliners, A_corr, B_corr = fpfh_teaser(np.asarray(src_graph['global_cloud'].points),np.asarray(tar_graph['global_cloud'].points),False)
            
            rot_error, t_error = eval_metrics.evaluate(pred_T,gt_T)
            global_reg_errors.append(np.array([rot_error,t_error]))

            if t_error>5.0: 
                warn_scans.append(scan)
            print('[{}] global map registration, {}/{} inliners, rotation error {:.3f}deg, translation error {:.3f}m'.format(scan,len(inliners),A_corr.shape[0],rot_error,t_error))
        
        executed_scans.append(scan)
        # continue
        
        # visualization
        src_cloud.paint_uniform_color([0,0.65,0.92])
        tar_cloud.paint_uniform_color([1,0.71,0.0])
        src_cloud.transform(pred_T)
        viz_geometries = [src_cloud,tar_cloud]
        # viz_geometries = corrected_geometries + tar_geometries

        #
        # viz_geometries = src_geometries + tar_geometries+ corr_lines
        o3d.visualization.draw(viz_geometries)
        
    exit(0)
    import json
    # with open(os.path.join(args.graphroot,'pred',args.prediction,'eval.json'),'w') as jf:
    #     json.dump(eval_instances,jf)
    #     jf.close()
    
    save_errors(executed_scans,inst_reg_errors,os.path.join(args.graphroot,'pred',args.prediction,'errors_instance_register.txt'))
    if args.global_register:
        save_errors(executed_scans,global_reg_errors,os.path.join(args.graphroot,'pred',args.prediction,'errors_global_register.txt'))
        
    exit(0)
    with open(os.path.join(args.graphroot,'pred',args.prediction,'warn_scans.txt'),'w') as f:
        for scan in warn_scans:
            f.write('{}\n'.format(scan))
        f.close()
