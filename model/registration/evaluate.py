import numpy as np

class EvalResult:
    def __init__(self, success=0, total=0, t: float=0):
        self.success = success
        self.total = total
        self.time = t

    def __add__(self, other):
        self.success += other.success
        self.total += other.total
        self.time += other.time
        return self

    def evaluate(self, tf, gt):
        rot_est = tf[:3, :3]
        rot_gt = gt[:3, :3]
        trace = np.trace(np.dot(rot_est, rot_gt.T))
        tmp = np.clip((trace - 1) / 2, -1, 1)
        rot_error = np.arccos(tmp) * 180 / np.pi
        # rot_succ = np.arccos(tmp) * 180 / np.pi < cfg.evaluation.rot_thd

        trans_est = tf[:3, 3]
        trans_gt = gt[:3, 3]
        trans_error = np.linalg.norm(trans_gt-trans_est)
        # trans_succ = np.linalg.norm(trans_gt - trans_est) < cfg.evaluation.trans_thd
        return rot_error, trans_error

    def print(self):
        msg = "Success rate: %.2f%%\n" % (self.success / self.total * 100)
        msg +="Average time: %.2fms" % (self.time / self.total * 1000)


def compute_residual(src_cloud:np.ndarray,tar_cloud:np.ndarray,tf:np.ndarray):
    src_cloud = np.concatenate([src_cloud,np.ones((src_cloud.shape[0],1))],axis=1)
    src_cloud = np.dot(tf,src_cloud.T).T
    residual = np.linalg.norm(src_cloud[:,:3]-tar_cloud[:,:3],axis=1)
    return residual.mean()