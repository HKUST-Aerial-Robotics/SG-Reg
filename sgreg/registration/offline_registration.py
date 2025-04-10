import os 
import torch
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf

from sgreg.loss.eval import Evaluator
from sgreg.utils.utils import read_scan_pairs
from sgreg.utils.io import read_pred_nodes, read_corr_scores

from os.path import join as osp

def read_points(dir:str):
    pcd = o3d.io.read_point_cloud(dir)
    points = np.asarray(pcd.points) # (N,3)
    return points

if __name__=='__main__':
    print('*'*60)
    print('This script reads the data association from neural network.')
    print('It estimates a relative transformation between scene grpahs.')
    print('Lastly, save and evaluate the results.')
    print('*'*60)
    
    ############ Args ############
    DATAROOT = '/data2/RioGraph'
    SPLIT = 'val'
    RESULT_FOLDER = osp(DATAROOT, 'output', 'sgnet_init_layer2')
    ##############################
    
    cfg = OmegaConf.create({'eval': {'acceptance_overlap': 0.0, 
                                    'acceptance_radius': 0.1, 
                                    'rmse_threshold': 0.2}})
    eval = Evaluator(cfg)
    summary_rmse = []
    summary_recall = []
    
    scene_pairs = read_scan_pairs(osp(DATAROOT, 'splits', SPLIT+'.txt'))
    print('Read {} scene pairs'.format(len(scene_pairs)))
    
    for scene_pair in scene_pairs:
        print('----------- {}-{} ------------'.format(scene_pair[0],scene_pair[1]))
        scene_result_folder = os.path.join(RESULT_FOLDER, '{}-{}'.format(scene_pair[0],scene_pair[1]))
        
        # 1. load src points, node_matches, and correspondence points
        src_points = read_points(osp(scene_result_folder,'src_instances.ply'))
        _, _, src_centroids, ref_centroids, _ = read_pred_nodes(osp(scene_result_folder,'node_matches.txt'))
        corr_src_points = read_points(osp(scene_result_folder,'corr_src.ply'))
        corr_ref_points = read_points(osp(scene_result_folder,'corr_ref.ply'))
        _, corr_scores, _ = read_corr_scores(osp(scene_result_folder,'point_matches.txt'))
        assert corr_src_points.shape[0] == corr_scores.shape[0]
        print('Read {} node matches and {} point corrs'.format(src_centroids.shape[0], 
                                                               corr_src_points.shape[0]))
        
        # 2. load gt
        T_ref_src = np.loadtxt(osp(scene_result_folder, 'gt_transform.txt'))
        
        # TODO 3. estimate transformation
        T_fake_pose = np.eye(4)
        
        # 4. Eval       
        precision, corr_errors \
            = eval.evaluate_fine({'src_corr_points':torch.from_numpy(corr_src_points).float(),
                                'ref_corr_points':torch.from_numpy(corr_ref_points).float()},
                                {'transform':torch.from_numpy(T_ref_src).float()})
        rre, rte, rmse, recall =\
            eval.evaluate_registration({'src_points':torch.from_numpy(src_points).float(),
                                    'estimated_transform':torch.from_numpy(T_fake_pose).float()},
                                    {'transform':torch.from_numpy(T_ref_src).float()})
        
        # print('Inlier ratio: {:.2f}'.format(precision.item()))
        msg = 'Inlier ratio: {:.3f}%, '.format(precision.item()*100)
        msg += 'RRE: {:.2f} deg, RTE: {:.2f}m'.format(rre.item(), rte.item())
        msg += ', RMSE: {:.2f}m'.format(rmse.item())
        print(msg)
        
        summary_rmse.append(rmse.item())
        summary_recall.append(recall.item())
        
        # break

    print('************************** Summary ***************************')
    # In the C++ version of registration code (https://github.com/glennliu/OpensetFusion/blob/master/src/Test3RscanRegister.cpp). 
    # The result is, Registration recall: 0.790(79/100), RMSE: 0.102m
    if len(summary_rmse)>0:
        summary_rmse = np.array(summary_rmse)
        summary_recall = np.array(summary_recall)    
        print('Average RMSE: {:.2f}m'.format(np.mean(summary_rmse)))
        print('Registration Recall: {:.2f}% ({}/{})'.format(np.mean(summary_recall)*100,
                                                        np.sum(summary_recall).astype(int),
                                                        summary_recall.shape[0]))