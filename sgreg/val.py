import os
import yaml, argparse
from omegaconf import OmegaConf
import numpy as np
import torch

from sgreg.dataset.dataset_factory import val_data_loader
from sgreg.config import create_cfg
from sgreg.utils.torch import to_cuda, checkpoint_restore
from sgreg.sg_reg import SGNet, SGNetDecorator
from train import val_epoch, create_model

def test_epoch(data_loader, model, test_model_fn, save_dir=''):
    # todo: inference without gt evaluation
    print('test start')
    memory_array = []
    time_record = []
    time_analysis = []
    count = 0
    
    for i, batch_data in enumerate(data_loader):
        batch_data = to_cuda(batch_data)
        output_dict, eval_dict = test_model_fn(batch_data,model)
        

        assert batch_data['batch_size'] == 1
        with torch.no_grad():
            if save_dir != '':
                src_scan = batch_data['src_scan'][0]
                ref_scan = batch_data['ref_scan'][0]
                src_subix = src_scan[-1]
                ref_subix = ref_scan[-1]
                scene_name = src_scan[:-1]
                if os.path.exists(os.path.join(save_dir,scene_name)) == False:
                    os.makedirs(os.path.join(save_dir,scene_name))
                    
                src_idx2name = batch_data['src_graph']['idx2name'].squeeze()
                ref_idx2name = batch_data['ref_graph']['idx2name'].squeeze()
                pred_nodes = output_dict['instance_matches']['pred_nodes'][:,1:] # (M,2)
                src_centroids = batch_data['src_graph']['centroids'].squeeze()
                ref_centroids = batch_data['ref_graph']['centroids'].squeeze()
                pred_src_centroids = src_centroids[pred_nodes[:,0].long()] # (M,3)
                pred_ref_centroids = ref_centroids[pred_nodes[:,1].long()] # (M,3)
                
                pred_nodes[:,0] = src_idx2name[pred_nodes[:,0].long()]
                pred_nodes[:,1] = ref_idx2name[pred_nodes[:,1].long()]
                
                scan_output_dict = {
                    'ref_instances':src_idx2name.detach().cpu().numpy(),
                    'src_instances':ref_idx2name.detach().cpu().numpy(),
                    'pred_nodes':pred_nodes.detach().cpu().numpy(),
                    'gt_transform': batch_data['transform'].squeeze().detach().cpu().numpy(),
                    'pred_scores': output_dict['instance_matches']['pred_scores'].squeeze().detach().cpu().numpy(),
                }
                print('{},{}: pred nodes {}'.format(src_scan,ref_scan,scan_output_dict['pred_nodes'].shape[0]))
                
                if 'registration' in output_dict:
                    corres_points = output_dict['registration']['points'].detach().cpu().numpy() # (C,6)
                    scan_output_dict['corres_ref_points'] = corres_points[:,:3]
                    scan_output_dict['corres_src_points'] = corres_points[:,3:]
                    scan_output_dict['corres_ref_centroids'] = pred_ref_centroids.detach().cpu().numpy()
                    scan_output_dict['corres_src_centroids'] = pred_src_centroids.detach().cpu().numpy()
                    scan_output_dict['corres_scores'] = output_dict['registration']['scores'].detach().cpu().numpy() # (C,)
                    scan_output_dict['corres_rmse'] = output_dict['registration']['errors'].detach().cpu().numpy() # (C,)
                    scan_output_dict['estimated_transforms'] = output_dict['registration']['estimated_transform'].squeeze().detach().cpu().numpy() # (4,4)
                    # scan_output_dict['ref_instance_points'] = output_dict['ref_instance_points'].squeeze().detach().cpu().numpy()
                    # scan_output_dict['src_instance_points'] = output_dict['src_instance_points'].squeeze().detach().cpu().numpy()
                    print('PIR:{:.3f}% in {} correspondence points'.format(100*eval_dict['PIR'].squeeze(),corres_points.shape[0]))
                    
                torch.save(scan_output_dict,os.path.join(save_dir,scene_name,'output_dict_{}{}.pth'.format(src_subix,ref_subix)))

    print('finished test')
    
def run_rag_epoch(data_loader, model, model_fn, save_dir):
    assert save_dir != ''
    print('Inference the dataset for RAG. Save result to {}'.format(save_dir))
    
    scene_pairs = []
    scene_size = []
    sum_metric_dict = {}
    count_scenes = 0
    
    for i, batch_data in enumerate(data_loader):
        batch_data = to_cuda(batch_data)
        _, _, output_dict, _ = model_fn(batch_data,
                                        model,
                                        epoch,
                                        False)
        
        with torch.no_grad():          
            assert 'x_src1' in output_dict and 'f_src' in output_dict
            
            # collect data
            src_scene_name = batch_data['src_scan'][0]
            ref_scene_name = batch_data['ref_scan'][0]
            ref_instances = batch_data['ref_graph']['idx2name']
            src_instances = batch_data['src_graph']['idx2name']
            ref_labels = batch_data['ref_graph']['labels']
            src_labels = batch_data['src_graph']['labels']
            x_src = output_dict['x_src1']
            x_ref = output_dict['x_ref1']
            f_src = output_dict['f_src']
            f_ref = output_dict['f_ref']
            
            assert isinstance(src_labels, list) and isinstance(src_labels[0],str)
            
            data2save_src = {'instances':src_instances,
                             'labels':src_labels,
                             'x':x_src,
                             'f':f_src}
            data2save_ref = {'instances':ref_instances,
                             'labels':ref_labels,
                             'x':x_ref,
                             'f':f_ref}
 
            msg = '{}-{}'.format(src_scene_name,ref_scene_name)
            
            # save data
            src_out_dir = os.path.join(save_dir,src_scene_name)
            ref_out_dir = os.path.join(save_dir,ref_scene_name)
            os.makedirs(src_out_dir,exist_ok=True)
            os.makedirs(ref_out_dir,exist_ok=True)
            
            torch.save(data2save_src,os.path.join(src_out_dir,'features.pth'))
            torch.save(data2save_ref,os.path.join(ref_out_dir,'features.pth'))
            print(msg)
            
    
    print('*********** finished RAG ***********')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN for scene graph matching')
    parser.add_argument('--cfg_file', type=str, default='config/scannet.yaml', 
                        help='path to config file')   
    parser.add_argument('--checkpoint', type=str, 
                        help='folder to save/load checkpoint')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--epoch',type=int,help='Load checkpoint at assigned epoch',default=-1)
    parser.add_argument('--output', type=str, default='', help='output folder')
    args = parser.parse_args()
    print('SGREG FLAG')
    
    # Paramters 
    assert os.path.exists(args.cfg_file), 'config file {} not found'.format(args.cfg_file)
    conf = OmegaConf.load(args.cfg_file)
    conf = create_cfg(conf)
    
    # Model 
    model = SGNet(conf=conf)
    model = model.cuda()
    epoch, f = checkpoint_restore(model,
                                  args.checkpoint,
                                  'sgnet',
                                  args.epoch)

    for name, params in model.named_parameters():
        if name.split('.')[0] in conf.model.fix_modules:
            params.requires_grad = False    
    model.eval()
    model_fn = SGNetDecorator(conf=conf)

    # Dataset
    val_loader, _ = val_data_loader(conf)
    print('Validation dataset: {} batches'.format(len(val_loader)))
    
    # Saving directory
    if args.output=='':
        pred_output = os.path.join(conf.dataset.dataroot,
                                'output',
                                os.path.basename(args.checkpoint))
    else:
        pred_output = args.output
    os.makedirs(pred_output,exist_ok=True)
    OmegaConf.save(conf, os.path.join(pred_output,'config.yaml'))

    # Val    
    loss_metrics, eval_metrics =val_epoch(
        val_loader, model, model_fn, epoch -1,save_dir=pred_output)   
    
    