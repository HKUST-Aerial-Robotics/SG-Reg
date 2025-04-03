import torch
import torch.nn as nn
import numpy as np
from model.ops import apply_transform
from model.registration.metrics import isotropic_transform_error


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict["ref_points_c"].shape[0]
        src_length_c = output_dict["src_points_c"].shape[0]
        gt_node_corr_overlaps = output_dict["gt_node_corr_overlaps"]
        gt_node_corr_indices = output_dict["gt_node_corr_indices"]
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict["ref_node_corr_indices"]
        src_node_corr_indices = output_dict["src_node_corr_indices"]

        precision = gt_node_corr_map[
            ref_node_corr_indices, src_node_corr_indices
        ].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict["transform"]
        ref_corr_points = output_dict["ref_corr_points"]
        src_corr_points = output_dict["src_corr_points"]
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision, corr_distances

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict["transform"]
        est_transform = output_dict["estimated_transform"]
        src_points = output_dict["src_points"]

        rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        # c_precision = self.evaluate_coarse(output_dict)
        # f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        return {
            # 'PIR': c_precision,
            # 'IR': f_precision,
            "RRE": rre,
            "RTE": rte,
            "RMSE": rmse,
            "RR": recall,
        }


def eval_instance_match(pred: torch.Tensor, gt: torch.Tensor):
    """background (floors, carpets) are considered.
    pred: (a,2), gt: (b,2)
    return: true_pos, false_pos
    """
    true_pos_mask = torch.zeros_like(pred[:, 0]).bool()

    for row, pred_pair in enumerate(pred):
        check_equal = (gt - pred_pair) == 0
        if check_equal.sum(dim=1).max() == 2:
            true_pos_mask[row] = True

    tp = true_pos_mask.sum().cpu().numpy()
    fp = pred.shape[0] - tp

    return tp, fp, true_pos_mask.detach().cpu().numpy().astype(np.int32)

def eval_instance_match_new(gt_matrix: torch.Tensor,
                        pred: torch.Tensor,
                        min_iou: float):
    """background (floors, carpets) are considered.
    - gt_matrix: (n,m)
    - pred: (a,2), [i,j] where i in [0,n), j in [0,m)
    - return: true_pos, false_pos
    """
    
    true_pos_mask = torch.zeros_like(pred[:, 0]).bool()
    
    # gt matches
    row_max = torch.zeros_like(gt_matrix).to(torch.int32)
    col_max = torch.zeros_like(gt_matrix).to(torch.int32)
    row_max[torch.arange(gt_matrix.shape[0]), gt_matrix.argmax(dim=1)] = 1
    col_max[gt_matrix.argmax(dim=0), torch.arange(gt_matrix.shape[1])] = 1
    valid_mask = row_max * col_max
    gt_tp_matrix = torch.gt(gt_matrix, min_iou) * valid_mask
    
    # pred matches. 
    # If there are multiple tp pairs, they are all considered as tp.
    for row, pred_pair in enumerate(pred):
        pred_iou = gt_matrix[pred_pair[0], pred_pair[1]]
        if pred_iou >= min_iou:
            true_pos_mask[row] = True
    
    tp = true_pos_mask.sum().cpu().numpy()
    fp = pred.shape[0] - tp
    true_pos_mask = true_pos_mask.detach().cpu().numpy().astype(np.int32)
    gt_pairs = gt_tp_matrix.sum().item()
    
    return tp, fp, gt_pairs, true_pos_mask

            


def is_recall(
    source: np.ndarray, T_est: np.ndarray, T_gt: np.ndarray, threshold: float
):
    """check if the registration is successful
    source: (N,3), T_est: (4,4), T_gt: (4,4)
    """

    source = np.hstack([source, np.ones((source.shape[0], 1))])
    realignment_transform = np.linalg.inv(T_gt) @ T_est
    realigned_src_points_f = source @ realignment_transform.T
    rmse = np.linalg.norm(realigned_src_points_f[:, :3] - source[:, :3], axis=1).mean()
    recall = rmse < threshold
    return recall, rmse

def compute_node_matching(metric_dict:dict):
    recall = metric_dict['nodes_tp'] / metric_dict['nodes_gt']
    precision = metric_dict['nodes_tp'] / (metric_dict['nodes_tp'] + metric_dict['nodes_fp'])

    recall = 100 * recall
    precision = 100 * precision
    return recall, precision

def compute_registration(metric_dict:dict):
    rmse = metric_dict['rmse'] / metric_dict['scenes']
    recall = metric_dict['recall'] / metric_dict['scenes']  
    recall = 100 * recall  
    return rmse, recall
