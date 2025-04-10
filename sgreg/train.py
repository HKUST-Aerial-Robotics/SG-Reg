import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sgreg.dataset.dataset_factory import train_data_loader, val_data_loader
from sgreg.utils.config import create_cfg
from sgreg.utils.torch import to_cuda, checkpoint_restore, checkpoint_save
from sgreg.utils.utils import RecallMetrics, load_submodule_state_dict, summary_dict
from sgreg.loss.eval import compute_node_matching, compute_registration
from sgreg.sg_reg import SGNet, SGNetDecorator

def create_model(conf,
                 train=True):
    # 1. Initialize model
    model = SGNet(conf=conf)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=conf.train.lr)
    start_epoch = 0

    # 2. Load pre-trained model
    if 'gtransformer_weight' in conf.model:
        submodule_state_dict = load_submodule_state_dict(conf.model.gtransformer_weight,
                                                         ['backbone','optimal_transport'])
        print('load kpconv backbone weight from {}'.format(conf.model.gtransformer_weight))    
        model.backbone.load_state_dict(submodule_state_dict['backbone'])    
        model.optimal_transport.load_state_dict(submodule_state_dict['optimal_transport'])
    elif 'pretrain_weight' in conf.model:
        pretrain_state_dict = load_submodule_state_dict(conf.model.pretrain_weight,
                                                        ['backbone','shape_backbone','optimal_transport'])
        model.backbone.load_state_dict(pretrain_state_dict['backbone'])
        model.shape_backbone.load_state_dict(pretrain_state_dict['shape_backbone'])
        model.optimal_transport.load_state_dict(pretrain_state_dict['optimal_transport'])
        # pretrain_state_dict = load_pretrain_weight(conf.model.pretrain_weight)
        # model.load_state_dict(pretrain_state_dict)
        print('load pre-train modules from {}'.format(conf.model.pretrain_weight))

    # 3. Fix modules
    for name, params in model.named_parameters():
        if name.split('.')[0] in conf.model.fix_modules:
            params.requires_grad = False
    print('fixed modules: {}'.format(conf.model.fix_modules))
    return model, optimizer, start_epoch

def convert_from_batch_pred(data_dict, pred_nodes, pred_scores=None):
    batch_pred_list = []
    batch_pred_score_list = []
    
    if pred_nodes.sum()<1: 
        return batch_pred_list, batch_pred_score_list
    # Instances
    
    for scan_id in torch.arange(data_dict['batch_size']):
        scan_mask = pred_nodes[:,0] == scan_id
        scan_pred = pred_nodes.clone()[scan_mask,1:] # (m,2)
        scan_pred[:,0] = data_dict['src_graph']['idx2name'][
            data_dict['src_graph']['scene_mask']==scan_id][scan_pred[:,0]]
        scan_pred[:,1] = data_dict['ref_graph']['idx2name'][
            data_dict['ref_graph']['scene_mask']==scan_id][scan_pred[:,1]]
        batch_pred_list.append(scan_pred.detach().cpu().numpy())
        if pred_scores is not None:
            batch_pred_score_list.append(pred_scores[scan_mask].detach().cpu().numpy())
    if pred_scores is None:
        return batch_pred_list
    else:
        return batch_pred_list, batch_pred_score_list
    

def train_epoch(data_loader, model, model_fn, optimizer, scheduler,epoch):
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    print('epoch {} start'.format(epoch))
    instance_recall_summary = RecallMetrics()
    registration_recall_summary = RecallMetrics()
    sum_loss_dict = {}
    sum_metric_dict = {}
    registration_rmse = 0.0
    count_scenes=0
    PIR = []
    loss_metrics = {}
    
    for i, data_dict in enumerate(data_loader):
        data_dict = to_cuda(data_dict)
        loss, loss_dict, _, metric_dict = model_fn(data_dict,
                                                    model,
                                                    epoch,
                                                    True)
        if loss is None:
            continue
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            summary_dict(loss_dict, sum_loss_dict)
            summary_dict(metric_dict, sum_metric_dict)
            
            msg = '- iter:{}, loss:{:.3f} '.format(i, loss_dict['loss'])
            msg += 'tp:{}, fp:{}, gt:{}, '.format(metric_dict['nodes_tp'],
                                                  metric_dict['nodes_fp'],
                                                  metric_dict['nodes_gt'])
            if 'precision' in metric_dict.keys():
                msg += 'PIR:{:.3f}, rmse:{:.3f}, '.format(metric_dict['precision']
                                                        /(metric_dict['scenes']+1e-6),
                                                        metric_dict['rmse'] 
                                                        / (metric_dict['scenes']+1e-6))
            
            if 'gt_precision' in metric_dict.keys():
                msg += 'gt PIR: {:.3f}, '.format(metric_dict['gt_precision']
                                             /(data_dict['batch_size']+1e-6))
            print(msg)
            
            count_scenes += data_dict['batch_size']           

    # 
    scheduler.step()
    # lrs = scheduler.get_last_lr()
    for k,v in sum_loss_dict.items():
        sum_loss_dict[k] = sum_loss_dict[k] / (i + 1)
    
    # 
    eval_dict = {}
    eval_dict['node_recall'], eval_dict['node_precision'] =\
        compute_node_matching(sum_metric_dict)
    eval_dict['inlier_ratio'] =\
        100*sum_metric_dict['precision']/(count_scenes+1e-6)
    eval_dict['lr'] = scheduler.get_last_lr()[0]

    print('*** Sumary of eval metrics ***')
    print(' node recall: {:.1f}%, node precision: {:.1f}%'.format(
        eval_dict['node_recall'],eval_dict['node_precision']))
    print(' inlier ratio: {:.1f}%'.format(eval_dict['inlier_ratio']))
    if 'gt_precision' in sum_metric_dict.keys():
        eval_dict['gt_inlier_ratio'] = 100*sum_metric_dict['gt_precision']/(count_scenes+1e-6)
        print(' gt inlier ratio: {:.1f}%'.format(eval_dict['gt_inlier_ratio']))
    return sum_loss_dict, eval_dict

    # PIR = torch.cat(PIR,dim=0).detach().mean().item()
    
    # if registration_recall_summary.num_gt>0:
    #     registration_rmse = registration_rmse / (i + 1)


        
    # print('train epoch:{}, loss: {:.3f}, R(nodes): {:.3f}, P(nodes): {:.3f}, FIR(points): {:.3f}, RR:{:.3f}'.format(
    #     epoch, loss_metrics['loss'], 
    #     instance_recall_summary.get_metrics()['R'], instance_recall_summary.get_metrics()['P'],
    #     PIR, registration_recall_summary.get_metrics()['R']))
    return loss_metrics, {'instances':instance_recall_summary.get_metrics(),
                          'registration':registration_recall_summary.get_metrics(True,True),
                          'PIR':PIR}

def val_epoch(data_loader, model, model_fn, epoch, save_dir=''):
    print('validation start. Save result to {}'.format(save_dir))
    
    scene_pairs = []
    scene_size = []
    sum_metric_dict = {}
    sum_loss_dict = {}
    sum_time_matrix = []
    sum_memory_matrix = []
    eval_dict = {}
    eval_msgs = ''
    count_scenes = 0
    
    for i, batch_data in enumerate(data_loader):
        batch_data = to_cuda(batch_data)
        _, loss_dict, output_dict, metric_dict = model_fn(batch_data,model,0, False)
                                                          

        with torch.no_grad():
            summary_dict(loss_dict, sum_loss_dict)
            summary_dict(metric_dict, sum_metric_dict)
            
            src_scene_name = batch_data['src_scan'][0]
            ref_scene_name = batch_data['ref_scan'][0]
            src_subname = batch_data['src_scan'][0][-1]
            ref_subname = batch_data['ref_scan'][0][-1]
            pair_name = '{}-{}'.format(src_scene_name,ref_scene_name)
            if save_dir != '' and batch_data['batch_size']==1:
                output_folder = os.path.join(save_dir,pair_name)
                if os.path.exists(output_folder) == False:
                    os.makedirs(output_folder)
                from sgreg.utils.io import write_pred_nodes, write_registration_results
                
                # Correspondences
                from sgreg.utils.utils import mask_valid_labels
                ref_instances = batch_data['ref_graph']['idx2name']
                src_instances = batch_data['src_graph']['idx2name']
                ref_labels = batch_data['ref_graph']['labels']
                src_labels = batch_data['src_graph']['labels']
                ref_centroids = batch_data['ref_graph']['centroids']
                src_centroids = batch_data['src_graph']['centroids']
                pred_nodes =output_dict['instance_matches']['pred_nodes'][:,1:].long() # (M,2)
                corr_src_centroids = src_centroids[pred_nodes[:,0]].detach().cpu().numpy() # (M,3)
                corr_ref_centroids = ref_centroids[pred_nodes[:,1]].detach().cpu().numpy() # (M,3)
                pred_src_instances = src_instances[pred_nodes[:,0]].detach().cpu().numpy()
                pred_ref_instances = ref_instances[pred_nodes[:,1]].detach().cpu().numpy()
                pred_gt_masks = output_dict['nodes_masks']
                
                # Remove invalid instances
                pred_nodes = pred_nodes.detach().cpu().numpy().astype(np.int32)
                pred_src_labels = [src_labels[i] for i in pred_nodes[:,0]]
                pred_ref_labels = [ref_labels[i] for i in pred_nodes[:,1]]
                pred_src_masks = mask_valid_labels(pred_src_labels,'floor')
                pred_ref_masks = mask_valid_labels(pred_ref_labels,'floor')
                pred_valid_masks = (pred_src_masks * pred_ref_masks)
                corr_src_centroids = corr_src_centroids[pred_valid_masks]
                corr_ref_centroids = corr_ref_centroids[pred_valid_masks]
                pred_src_instances = pred_src_instances[pred_valid_masks]
                pred_ref_instances = pred_ref_instances[pred_valid_masks]
                pred_gt_masks = pred_gt_masks[pred_valid_masks]

                write_pred_nodes(os.path.join(output_folder,'node_matches.txt'),
                                    pred_src_instances,
                                    pred_ref_instances,
                                    corr_src_centroids,
                                    corr_ref_centroids,
                                    pred_gt_masks)
                
                if 'registration' in output_dict:
                    write_registration_results(output_dict['registration'], 
                                               output_folder)
                
                gt_transform = batch_data['transform'][0].detach().cpu().numpy()
                np.savetxt(os.path.join(output_folder,'gt_transform.txt'),
                           gt_transform,
                           fmt='%.6f')
                sum_time_matrix.append(np.array(output_dict['time_list']))
            sum_memory_matrix.append(np.array(output_dict['memory_list']))
                                               
            # dl_duration.append(1000*batch_data['loader_duration'])
            msg = 'tp:{}, fp:{}, gt:{}, '.format(metric_dict['nodes_tp'],
                                                  metric_dict['nodes_fp'],
                                                  metric_dict['nodes_gt'])
            msg += 'PIR:{:.3f}, rmse:{:.3f}, '.format(metric_dict['precision']
                                                     /(metric_dict['scenes']+1e-6),
                                               metric_dict['rmse'] 
                                               / (metric_dict['scenes']+1e-6))
            if 'gt_precision' in metric_dict.keys():
                msg += 'gt PIR: {:.3f}, '.format(metric_dict['gt_precision']
                                             /(batch_data['batch_size']+1e-6))
                        
            msg += 'Recall: {}'.format(metric_dict['recall'])
            eval_msgs += '['+pair_name+'] '+msg+'\n'
            
            print('iter: {}, loss: {:.3f} '.format(i, loss_dict['loss']) + msg)
            count_scenes+=batch_data['batch_size']

        torch.cuda.empty_cache()

    for k,v in sum_loss_dict.items():
        sum_loss_dict[k] = sum_loss_dict[k] / (i + 1)
            
    eval_dict['node_recall'], eval_dict['node_precision'] =\
        compute_node_matching(sum_metric_dict)
    eval_dict['inlier_ratio'] = 100*sum_metric_dict['precision']/(sum_metric_dict['scenes']+1e-6)
   
    print('*** Sumary of eval metrics ***')
    print(' node recall: {:.1f}%, node precision: {:.1f}%'.format(
        eval_dict['node_recall'],eval_dict['node_precision']))
    print(' inlier ratio: {:.1f}%'.format(eval_dict['inlier_ratio']))
    
    if 'recall' in sum_metric_dict.keys() and sum_metric_dict['scenes']>0:
        eval_dict['rmse'], eval_dict['recall'] = \
            compute_registration(sum_metric_dict)     
        print(' rmse: {:.3f}m, recall: {:.1f}%'.format(
            eval_dict['rmse'],eval_dict['recall']))
    if len(sum_time_matrix)>0:
    # if False:
        header = ' point, shape, init , gnn , fuse, m_node, m_pnts, regt; Sum'
        sum_time_matrix = np.vstack(sum_time_matrix) # (-1, 6)
        sum_time_matrix = sum_time_matrix[1:,:] # remove the first row
        np.savetxt(os.path.join(save_dir,'time_matrix.txt'),
                   sum_time_matrix,
                   header=header,
                   fmt='%.3f')
        mean_times = sum_time_matrix.mean(axis=0) # (6,)
        time_msg = [' {:.2f}'.format(t) for t in mean_times]
        time_msg += '; {:.2f}'.format(mean_times.sum().item())
        print('T(ms):'+ header + '\n      ' 
              + ' '.join(time_msg))
    
        sum_memory_matrix = np.hstack(sum_memory_matrix).reshape(-1,3) # (B,3)
        mean_memories = sum_memory_matrix.mean(axis=0) # (3,)
        print('allocated: {:.1f}MB, reserved: {:.1f}MB, max_reserved: {:.1f}MB'.format(
            mean_memories[0],mean_memories[1],mean_memories[2]))
        np.savetxt(os.path.join(save_dir,'memory_usage.txt'),
                     sum_memory_matrix,
                     header='allocated, reserved, max_reserved',
                     fmt='%.1f')
    
    if 'gt_precision' in sum_metric_dict.keys():
        eval_dict['gt_inlier_ratio'] = 100*sum_metric_dict['gt_precision']/(count_scenes+1e-6)
        print(' gt inlier ratio: {:.1f}%'.format(eval_dict['gt_inlier_ratio']))
    if save_dir is not None:
        with open(os.path.join(save_dir,'eval_metrics.txt'),'w') as f:
            if 'recall' in eval_dict.keys():
                f.write('rmse: {:.3f}m, recall: {:.1f}%\n'.format(eval_dict['rmse'],
                                                                 eval_dict['recall']))
            f.write(eval_msgs)
        
    # memory_msg = '# pairs, volumes, allocated memory(MB), reserved memory(MB), max reserved memory(MB)\n'
    # for i in range(memory_array.shape[0]):
    #     memory_msg += '{}, {:.1f}, {:.1f}, {:.1f}, {:.1f}\n'.format(scene_pairs[i],
    #                                                             scene_size[i],
    #                                                             memory_array[i,0],
    #                                                             memory_array[i,1],
    #                                                             memory_array[i,2])
    # with open(os.path.join(save_dir,'memory_usage.txt'),'w') as f:
    #     f.write(memory_msg)
    
    return sum_loss_dict, eval_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN for scene graph matching')
    parser.add_argument('--cfg_file', type=str, default='config/scannet.yaml', 
                        help='path to config file')
    parser.add_argument('--dataroot', type=str, default='/data2/ScanNetGraph',
                        help='path to dataset')    
    parser.add_argument('--checkpoint', type=str, 
                        help='folder to save/load checkpoint')

    args = parser.parse_args()
    os.makedirs(args.checkpoint, exist_ok=True)
    
    # Paramters 
    assert os.path.exists(args.cfg_file), 'config file {} not found'.format(args.cfg_file)
    conf = OmegaConf.load(args.cfg_file)
    conf = create_cfg(conf)
    OmegaConf.save(conf,os.path.join(args.checkpoint,'config.yaml'))

    # Model 
    model, optimizer, start_epoch = create_model(conf,train=True)
    start_epoch, f = checkpoint_restore(model=model,
                       exp_path=args.checkpoint,
                       exp_name='sgnet',
                       epoch=start_epoch,
                       optimizer=optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=20, 
                                        gamma=0.5, 
                                        last_epoch=start_epoch-1, 
                                        verbose=True)
    model_fn = SGNetDecorator(conf=conf)
    
    # Log
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint,'log'))
    output_folder = os.path.join(args.dataroot,
                               'output',
                               os.path.basename(args.checkpoint))
    if os.path.exists(output_folder) == False:
        os.makedirs(output_folder)

    # Dataset
    train_loader, neighbor_limits = train_data_loader(conf)
    print('Trainning dataset: {} batches'.format(len(train_loader)))
    
    if conf.train.val_interval>0:
        val_loader, _ = val_data_loader(conf)
        print('Validation dataset: {} batches'.format(len(val_loader)))
      
    # Trainning
    cur_epoch = 0
    for epoch in range(start_epoch,conf.train.epochs+1):
        model.train()
        train_loss_dict, eval_train_dict = train_epoch(train_loader, 
                                                         model, 
                                                         model_fn, 
                                                         optimizer, 
                                                         scheduler, 
                                                         epoch)

        assert 'loss' in train_loss_dict.keys(), 'loss not found in train_loss_dict'
        for k,v in train_loss_dict.items():
            writer.add_scalar('train/{}'.format(k), v, epoch)
        
        for k,v in eval_train_dict.items():
            writer.add_scalar('train/{}'.format(k), v, epoch)
        
        if conf.train.val_interval>0 and epoch % conf.train.val_interval == 0:
            model.eval()
            val_loss_dict, eval_val_dict = val_epoch(val_loader, 
                                                     model, 
                                                     model_fn, 
                                                     epoch)

            for k,v in val_loss_dict.items():
                writer.add_scalar('val/{}'.format(k), v, epoch)
            for k,v in eval_val_dict.items():
                writer.add_scalar('val/{}'.format(k), v, epoch)

        checkpoint_save(model=model,
                        optimizer=optimizer,
                        exp_path=args.checkpoint,
                        exp_name='sgnet',
                        epoch=epoch,
                        save_freq=conf.train.save_interval,
                        ignore_keys=['bert'])
        cur_epoch = epoch

    print('train finished')

    # Save
    if conf.train.val_interval>0:
        val_epoch(val_loader, model, model_fn, cur_epoch, save_dir=output_folder)    
    # print('save to {}'.format(cfg.run_dir))
    # save_ckpt(model=model,
    #           optimizer=optimizer,
    #           scheduler=scheduler,epoch=epoch)
