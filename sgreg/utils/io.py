import os, sys
import numpy as np
import open3d as o3d

def write_pred_nodes(dir:str,
                     pred_src_instances:np.ndarray,
                     pred_ref_instances:np.ndarray,
                     pred_src_centroids:np.ndarray,
                     pred_ref_centroids:np.ndarray,
                     gt_mask:np.ndarray):
    
    with open(dir,'w') as f:
        f.write('# src_id, ref_id, gt_mask, src_centroid, ref_centroid\n')
        for i in range(pred_src_instances.shape[0]):
            f.write('{} {} {} '.format(pred_src_instances[i],
                                      pred_ref_instances[i],
                                      gt_mask[i]))
            f.write('{:.3f} {:.3f} {:.3f} '.format(pred_src_centroids[i,0],
                                                   pred_src_centroids[i,1],
                                                   pred_src_centroids[i,2]))
            f.write('{:.3f} {:.3f} {:.3f}\n'.format(pred_ref_centroids[i,0],
                                                    pred_ref_centroids[i,1],
                                                    pred_ref_centroids[i,2]))
        f.close()
        # print('write pred nodes to {}'.format(dir))
        
def write_registration_results(registration_dict:dict,
                               dir:str):
    corres_points = registration_dict['points'].detach().cpu().numpy() # (C,6)
    corres_instances = registration_dict['instances'].detach().cpu().numpy() # (C,1), [m]
    corres_scores = registration_dict['scores'].detach().cpu().numpy() # (C,)
    # corres_rmse = registration_dict['errors'].detach().cpu().numpy() # (C,)
    corres_masks = registration_dict['corres_masks'].int().detach().cpu().numpy() # (C,)
    estimated_transforms = registration_dict['estimated_transform'].squeeze().detach().cpu().numpy() # (4,4)

    # correspondences
    corre_ref_pcd = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(corres_points[:,:3]))
    corre_src_pcd = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(corres_points[:,3:]))
    o3d.io.write_point_cloud(os.path.join(dir,'corr_ref.ply'),corre_ref_pcd)
    o3d.io.write_point_cloud(os.path.join(dir,'corr_src.ply'),corre_src_pcd)
    
    # correspondences info
    with open(os.path.join(dir,'point_matches.txt'),'w') as f:
        f.write('# match, score, tp_mask\n')
        for i in range(corres_points.shape[0]):
            f.write('{} {:.2f} {}\n'.format(corres_instances[i],
                                            corres_scores[i],
                                            corres_masks[i]))
        f.close()
    
    # estimated transform
    np.savetxt(os.path.join(dir,'svds_estimation.txt'),
               estimated_transforms,
               fmt='%.6f')

def read_pred_nodes(dir:str):
    pred_src_instances = []
    pred_ref_instances = []
    pred_src_centroids = []
    pred_ref_centroids = []
    gt_mask = []
    
    with open(dir,'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().split(' ')
            pred_src_instances.append(int(line[0]))
            pred_ref_instances.append(int(line[1]))
            gt_mask.append(int(line[2]))
            pred_src_centroids.append([float(line[3]),float(line[4]),float(line[5])])
            pred_ref_centroids.append([float(line[6]),float(line[7]),float(line[8])])
        f.close()
    
    return {'src_instances':np.array(pred_src_instances),
            'ref_instances':np.array(pred_ref_instances),
            'src_centroids':np.array(pred_src_centroids),
            'ref_centroids':np.array(pred_ref_centroids),
            'gt_mask':np.array(gt_mask)}
    
    # return np.array(pred_src_instances), \
    #         np.array(pred_ref_instances), \
    #         np.array(pred_src_centroids), \
    #         np.array(pred_ref_centroids), \
    #         np.array(gt_mask)

def read_corr_scores(dir:str):
    corr_instances = []
    corr_scores = []
    corr_masks = []
    
    with open(dir,'r') as f:
        lines = f.readlines()
        for line in lines:
            if '#' in line:
                continue
            line = line.strip().split(' ')
            corr_instances.append(int(line[0]))
            corr_scores.append(float(line[1]))
            corr_masks.append(int(line[2]))
        f.close()
        return np.array(corr_instances), \
                np.array(corr_scores), \
                np.array(corr_masks)

def read_dense_correspondences(src_corr_dir:str,
                               ref_corr_dir:str,
                               score_file_dir:str):
    src_corr_pcd = o3d.io.read_point_cloud(src_corr_dir)
    ref_corr_pcd = o3d.io.read_point_cloud(ref_corr_dir)
    src_corr_points = np.asarray(src_corr_pcd.points) # (N,3)
    ref_corr_points = np.asarray(ref_corr_pcd.points) # (N,3)

    corr_inst, corr_scores, corr_masks = read_corr_scores(score_file_dir)
    assert src_corr_points.shape[0] == corr_scores.shape[0]
    assert src_corr_points.shape[0] == ref_corr_points.shape[0]
    assert src_corr_points.shape[0] == corr_masks.shape[0]
    
    return {'src_corrs':src_corr_points,
            'ref_corrs':ref_corr_points,
            'corr_scores':corr_scores,
            'corr_masks':corr_masks}