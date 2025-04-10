import os, glob
import torch
import torch.nn.functional as F
import numpy as np
from time import perf_counter


class RecallMetrics:
    def __init__(self):
        self.num_pos = 0
        self.num_neg = 0
        self.num_gt = 0

    # def update(self,tp_,fp_,gt_):
    def update(self, data_dict):
        self.num_pos += data_dict["tp"]
        self.num_neg += data_dict["fp"]
        self.num_gt += data_dict["gt"]

    def get_metrics(self, precentage=True, recall_only=False):
        if self.num_pos < 1:
            recall = 0.0
            precision = 0.0
        else:
            recall = self.num_pos / (self.num_gt + 1e-6)
            precision = self.num_pos / (self.num_pos + self.num_neg + 1e-6)

        if precentage:
            recall = recall * 100
            precision = precision * 100
        if recall_only:
            return {"R": recall}
        else:
            return {"R": recall, "P": precision}


class TicToc:
    def __init__(self):
        self.time_dict = {}

    def tic(self, key_name):
        if key_name not in self.time_dict:
            self.time_dict[key_name] = {"time": 0, "number": 0, "t0": perf_counter()}
        else:
            self.time_dict[key_name]["t0"] = perf_counter()

    def toc(self, key_name, verbose=False):
        if key_name not in self.time_dict:
            raise ValueError(f"No timer started for {key_name}")
        t1 = perf_counter()
        elapsed_time = (t1 - self.time_dict[key_name]["t0"]) * 1000
        self.time_dict[key_name]["time"] += elapsed_time
        self.time_dict[key_name]["number"] += 1
        if verbose:
            print(f"Time for {key_name}: {elapsed_time:.2f} ms")
        return elapsed_time

    def print_summary(self):
        for k, v in self.time_dict.items():
            average_time = v["time"] / v["number"] if v["number"] > 0 else 0
            print(f"Average Time for {k}: {average_time:.2f} ms")


timer = TicToc()

def summary_dict(input_dict: dict, output_dict: dict):
    for k, v in input_dict.items():
        if k not in output_dict:
            output_dict[k] = v
        else:
            output_dict[k] += v

def update_dict(total_dict: dict, sub_dict: dict, name: str):
    for k, v in sub_dict.items():
        total_dict[name + "_" + k] = v
    return total_dict


def read_scans(dir):
    with open(dir, "r") as f:
        scans = [line.strip() for line in f.readlines()]
        print("Find {} scans to load".format(len(scans)))
        return scans


def create_mask_from_edges(edge_index, min_nodes, max_nodes):
    nodes_number = max_nodes - min_nodes
    mask = torch.zeros(nodes_number, nodes_number).bool().to(edge_index.device)
    valid_edges = edge_index[
        :,
        (edge_index[0, :] >= min_nodes)
        & (edge_index[0, :] < max_nodes)
        & (edge_index[1, :] >= min_nodes)
        & (edge_index[1, :] < max_nodes),
    ]  # (2,e)

    valid_edges = valid_edges - min_nodes
    mask[valid_edges[0, :], valid_edges[1, :]] = True
    return mask

def mask_valid_labels(src_labels:list,
                      ignore_labels:str):
    mask = np.ones(len(src_labels),dtype=bool)
    for i, label in enumerate(src_labels):
        if label in ignore_labels: mask[i] = False
    
    return mask

def read_scan_pairs(dir):
    with open(dir, "r") as f:
        pairs = [line.strip().split(" ") for line in f.readlines()]
        return pairs

def scanpairs_2_scans(scan_pairs):
    scans = []
    for pair in scan_pairs:
        scans.append(pair[0])
        scans.append(pair[1])
    return scans


def write_scan_pairs(scan_pairs, dir):
    with open(dir, "w") as f:
        n = len(scan_pairs)
        for i, pair in enumerate(scan_pairs):
            f.write("{} {}".format(pair[0], pair[1]))
            if i < n - 1:
                f.write("\n")


def load_checkpoint(path: str, model: torch.nn.Module, keyname: str = "pointnet2"):
    if os.path.exists(path):
        print("load checkpoint {} from {}".format(keyname, path))
        ckpt = torch.load(path)
        model.load_state_dict(ckpt[keyname])
        return True
    else:
        print("checkpoint {} not found".format(path))
        return False

    MODEL_STATE = "model_state"
    OPTIMIZER_STATE = "optimizer_state"
    SCHEDULER_STATE = "scheduler_state"
    # ckpt_weights ={'pointnet2':ckpt[MODEL_STATE]}
    # torch.save(ckpt_weights, path.replace('64','model_epoch64'))

    # if optimizer is not None and OPTIMIZER_STATE in ckpt:
    #     optimizer.load_state_dict(ckpt[OPTIMIZER_STATE])
    # if scheduler is not None and SCHEDULER_STATE in ckpt:
    #     scheduler.load_state_dict(ckpt[SCHEDULER_STATE])

    # epoch = os.path.basename(path).split('.')[0]
    # print('load checkpoint from {}, epoch {}'.format(path,epoch))

    # return int(epoch) + 1

def load_pretrain_weight(dir):
    model_dict = torch.load(dir)
    if "model" in model_dict:
        model_dict = model_dict["model"]
    elif "model_state" in model_dict:
        model_dict = model_dict["model_state"]
    else:
        raise ValueError("No model state found in the checkpoint")
        
    return model_dict

def load_submodule_state_dict(model_state_dir, submodule_names=["backbone"]):
    model_dict = torch.load(model_state_dir)
    if "model" in model_dict:
        model_dict = model_dict["model"]
    elif "model_state" in model_dict:
        model_dict = model_dict["model_state"]
    elif 'state_dict' in model_dict:
        model_dict = model_dict['state_dict']

    submodule_dicts = {}
    for submodule_name in submodule_names:

        submodule_dicts[submodule_name] = {
            k[len(submodule_name) + 1 :]: v
            for k, v in model_dict.items()
            if submodule_name==k.split(".")[0]
        }

    return submodule_dicts
