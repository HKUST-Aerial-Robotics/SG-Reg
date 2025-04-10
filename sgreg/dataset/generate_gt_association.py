import os, sys
import argparse
import open3d as o3d
import numpy as np
import torch
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
from sgreg.ops import apply_transform
import csv

from sgreg.dataset.scene_graph import load_raw_scene_graph
from sgreg.utils.utils import read_scans, read_scan_pairs

def generate_edges(name:str,
                   graph:dict):
    n = len(graph['nodes'])
    searched_inst = []
    NEIGHBOR_RATIO = 2.0
    floors = []
    ceilings = []
    floors_names = ['floor','carpet']
    ceiling_names = ['ceiling']
    
    # between objects
    for i,inst_i in graph['nodes'].items():
        searched_inst.append(i)
        radius_a = LA.norm(inst_i.box.extent)/2
        if inst_i.label=='wall': radius_a = 0.1
        elif inst_i.label in floors_names: 
            floors.append(i)
            continue
        elif inst_i.label in ceiling_names: 
            ceilings.append(i)
            continue
        for j, inst_j in graph['nodes'].items():
            if j in searched_inst: continue
            radius_b = LA.norm(inst_j.box.extent)/2
            if inst_j.label=='wall': radius_b = 0.1
            elif inst_j.label in floors_names: continue
            elif inst_j.label in ceiling_names: continue
            radius = max(radius_a,radius_b)
            dist = LA.norm(inst_i.box.center - inst_j.box.center)
            if dist<radius*NEIGHBOR_RATIO:
                graph['edges'].append((i,j))
            
            # pass

    # object->one floor
    for i, inst_i in graph['nodes'].items():
        if i in floors or i in ceilings: continue
        closet_floor = {'id':None,'dist':100.0}
        closet_ceiling = {'id':None,'dist':100.0}
        
        for j in floors: # find the closet floor
            dist = LA.norm(inst_i.box.center - graph['nodes'][j].box.center)
            if dist<closet_floor['dist']: 
                closet_floor['dist'] = dist
                closet_floor['id'] = j
        for j in ceilings: # find the closet ceiling
            dist = LA.norm(inst_i.box.center - graph['nodes'][j].box.center)
            if dist<closet_ceiling['dist']:
                closet_ceiling['dist'] = dist
                closet_ceiling['id'] = j

        # add edge
        if closet_floor['id'] is not None:
            graph['edges'].append((i,closet_floor['id']))
        if closet_ceiling['id'] is not None:
            graph['edges'].append((i,closet_ceiling['id']))

    print('Extract {} edges for {} graph'.format(len(graph['edges']),name))
    return graph

def generate_global_edges(name:str,graph:dict):
    graph['global_edges'] = []
    
    for idx, inst in graph['nodes'].items():
        for jdx, inst_j in graph['nodes'].items():
            # if (idx,jdx) in graph['edges'] or (jdx,idx) in graph['edges']: continue
            
            if idx==jdx: continue
            elif jdx in graph['descriptive_instances']:
                dist = LA.norm(inst.cloud.get_center()-inst_j.cloud.get_center())
                graph['global_edges'].append((idx,jdx,dist))
                
    return graph
     
def compute_cloud_overlap(cloud_a:o3d.geometry.PointCloud,cloud_b:o3d.geometry.PointCloud,search_radius=0.2):
    # compute point cloud overlap 
    Na = len(cloud_a.points)
    Nb = len(cloud_b.points)
    correspondences = []
    cloud_a_occupied = np.zeros((Na,1))
    pcd_tree_b = o3d.geometry.KDTreeFlann(cloud_b)
    
    for i in range(Na):
        [k,idx,_] = pcd_tree_b.search_radius_vector_3d(cloud_a.points[i],search_radius)
        if k>1:
            cloud_a_occupied[i] = 1
            correspondences.append([i,idx[0]])
    assert len(correspondences)==len(np.argwhere(cloud_a_occupied==1))
    iou = len(correspondences)/(Na+Nb-len(correspondences))
    return iou, correspondences

def semantic_sampling(graph:dict, max_count:int=5, min_size:int=200):
    
    semantics = [] # list of labels
    semanitcs_histogram = []
    instances_samples = []
    AMBIGUOUS_TYPES = ['floor','carpet','wall','ceiling','chair','swivel chair'] #todo: remove wall
    DESCRIPTIVE_TYPES = ['pillar','door','elevator','stair','fridge'] # todo: extract decriptive instances
    
    # Sample by semantic histogram
    for idx, inst in graph['nodes'].items():
        if inst.label in AMBIGUOUS_TYPES: continue
        
        if inst.label not in semantics:
            semantics.append(inst.label)
            semanitcs_histogram.append(1)
        else:
            semanitcs_histogram[semantics.index(inst.label)] += 1
    
    descriptive_semantics_indices = [index for index,count in enumerate(semanitcs_histogram) if count<max_count]
    descriptive_semantics = [semantics[index] for index in descriptive_semantics_indices]
    
    #
    semantic_hist_msg = ''
    for index, seamntic_category in enumerate(semantics):
        semantic_hist_msg += '{}:{} '.format(seamntic_category,semanitcs_histogram[index])
    print(semantic_hist_msg)
    
    #
    for idx, inst in graph['nodes'].items():
        if inst.label in descriptive_semantics and len(inst.cloud.points)>min_size:
            instances_samples.append(idx)
    
    print('descriptive semantics:', descriptive_semantics)
    print('{} descriptive instances'.format(len(instances_samples)))

    return instances_samples

def find_association(src_graph:dict,
                     tar_graph:dict,
                     min_iou=0.5,
                     search_radius=0.1):
    ''' find gt association 
    Return:
        - matche_results: [(src_idx, tar_idx, iou)]
        - correspondences: np.array, (Mp,4), [src_idx, tar_idx, u, v]
        - global_uvs: np.array, (Mp,2), [u, v]
    '''
    import numpy.linalg as LA
    # find association
    Nsrc = len(src_graph['nodes'])
    Ntar = len(tar_graph['nodes'])
    assert 'global_cloud' in src_graph and 'global_cloud' in tar_graph, 'global cloud not exist'
    if Nsrc==0 or Ntar==0: 
        print('{} src instances and {} tar instances!'.format(Nsrc,Ntar))
        return [],[],[],[]
    
    iou = np.zeros((Nsrc,Ntar))
    correspondences = [] # [src_idx, tar_idx, u, v]
    n_match_pts = 0
    # SEARCH_RADIUS = 0.05
    assignment = np.zeros((Nsrc,Ntar),dtype=np.int32)

    src_graph_list = [src_idx for src_idx,_ in src_graph['nodes'].items()]
    tar_graph_list = [tar_idx for tar_idx,_ in tar_graph['nodes'].items()]
    xyzi_src = src_graph['xyzi']
    xyzi_tar = tar_graph['xyzi']
    
    # calculate iou
    for row_,src_idx in enumerate(src_graph_list):
        src_inst = src_graph['nodes'][src_idx]
        # src_centroid = src_inst.cloud.get_center()
        src_point_index = np.where(xyzi_src[:,-1]==src_idx)[0][0]
        
        assert (xyzi_src[:,-1]==src_idx).sum() == len(src_inst.cloud.points)
        for col_, tar_idx in enumerate(tar_graph_list):
            # xyz_tar = xyzi_tar[xyzi_tar[:,-1]==tar_idx,:3]
            tar_point_index = np.where(xyzi_tar[:,-1]==tar_idx)[0][0]
            
            tar_inst = tar_graph['nodes'][tar_idx]
            iou[row_,col_], uvs = compute_cloud_overlap(src_inst.cloud,tar_inst.cloud,search_radius=search_radius)
            if len(uvs)>0:
                nodes_corres = [[src_idx,tar_idx,uv[0]+src_point_index,uv[1]+tar_point_index] for uv in uvs]
                correspondences.append(np.array(nodes_corres))
                # correspondences.extend([[src_idx,tar_idx,uv[0],uv[1]] for uv in uvs])
                n_match_pts += len(uvs)
    if len(correspondences)>0:
        correspondences = np.vstack(correspondences)
        assert correspondences.shape[1]==4
    else:
        correspondences = np.zeros((0,4))
    
    #
    pcd_src = o3d.geometry.PointCloud()
    pcd_tar = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_graph['xyzi'][:,:3])
    pcd_tar.points = o3d.utility.Vector3dVector(tar_graph['xyzi'][:,:3])
    global_iou, _ = compute_cloud_overlap(pcd_src,pcd_tar,search_radius=search_radius)
    
    # find match 
    row_maximum = np.zeros((Nsrc,Ntar),dtype=np.int32)
    col_maximum = np.zeros((Nsrc,Ntar),dtype=np.int32)
    row_maximum[np.arange(Nsrc),np.argmax(iou,1)] = 1 # maximum match for each row
    col_maximum[np.argmax(iou,0),np.arange(Ntar)] = 1 # maximu match for each column
    assignment = row_maximum*col_maximum # maximum match for each row and column
    
    # filter
    valid_assignment = iou>min_iou
    assignment = assignment*valid_assignment
    
    #
    matches = np.argwhere(assignment==1)
    matche_results = [[src_graph_list[match[0]],tar_graph_list[match[1]],iou[match[0],match[1]]] for match in matches]
    
    return matche_results, correspondences, global_iou, \
        {'src_names':src_graph_list,
         'tar_names':tar_graph_list,
         'iou':torch.from_numpy(iou)}

def transform_scene_graph(graph:dict,transormation=np.eye(4)):
    translation = transormation[:3,3]
    rotate = transormation[:3,:3]
    for idx, instance in graph['nodes'].items():
        instance.cloud.transform(transormation)
        instance.box.rotate(rotate)
        instance.box.translate(translation)
        
    #
    xyz = torch.tensor(graph['xyzi'][:,:3]).float()
    xyz = apply_transform(xyz,torch.tensor(transormation).float())
    graph['xyzi'][:,:3] = xyz.numpy()
    

def get_geometries(graph:dict,translation=np.array([0,0,0]),rotate=np.eye(3),include_boxes=False,edge_type=None):
    geometries = []
    STURCTURE_LABELS = ['floor','carpet']
    transform = np.eye(4)
    transform[:3,:3] = rotate
    transform[:3,3] = translation
    # print('transform: \n',transform)
    
    # nodes
    for idx, instance in graph['nodes'].items():
        instance.cloud.transform(transform)
        instance.box.rotate(rotate)
        instance.box.translate(translation)

        geometries.append(o3d.geometry.PointCloud(instance.cloud))
        if include_boxes:
            geometries.append(instance.box)
    
    # edges
    DIST = 2.0
    if edge_type !=None:
        for edge in graph[edge_type]:
            # if edge[2]<DIST: continue
            src_idx = edge[0]
            tar_idx = edge[1]
            src_inst = graph['nodes'][src_idx]
            tar_inst = graph['nodes'][tar_idx]
            src_center = src_inst.cloud.get_center() #src_inst.box.get_center()
            tar_center = tar_inst.cloud.get_center() #tar_inst.box.get_center()
            # print(src_center.transpose(), tar_center.transpose())
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.vstack((src_center,tar_center))),
                lines=o3d.utility.Vector2iVector(np.array([[0,1]])))
            line_set.paint_uniform_color((0.9,0.9,0.9))
            # if src_inst.label in STURCTURE_LABELS or tar_inst.label in STURCTURE_LABELS:
            #     line_set.paint_uniform_color((0,0,1))
            
            geometries.append(line_set)    
    
    return geometries

def get_match_lines(src_graph:dict,tar_graph:dict,matches,translation=np.array([0,0,0]),masks=None):
    lines = []
    for i, match in enumerate(matches):
        src_idx = match[0]
        tar_idx = match[1]
        if src_idx not in src_graph or tar_idx not in tar_graph: continue
        assert src_idx in src_graph, '{} not in src graph'.format(src_idx)
        assert tar_idx in tar_graph, '{} not in tar graph'.format(tar_idx)
        src_inst = src_graph[src_idx]
        tar_inst = tar_graph[tar_idx]
        src_center = src_inst.cloud.get_center() #src_inst.box.get_center()
        tar_center = tar_inst.cloud.get_center()
        # print(src_center.transpose(), tar_center.transpose())
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.vstack((src_center,tar_center))),
            lines=o3d.utility.Vector2iVector(np.array([[0,1]])))
        if masks is not None:
            if masks[i]: # true
                line_set.paint_uniform_color((0,1,0))
            else:
                line_set.paint_uniform_color((1,0,0))
        lines.append(line_set)
        
    return lines

def get_correspondences_lines(src_graph:dict,tar_graph:dict,matches,correspondences,translation=np.array([0,0,0])):
    # lines = []
    sample_id = np.random.choice(len(matches))
    src_idx = matches[sample_id][0]
    tar_idx = matches[sample_id][1]
    mask = np.logical_and(correspondences[:,0]==src_idx,correspondences[:,1]==tar_idx)
    point_correspondences = correspondences[mask,2:].astype(np.int32) # (Mp,2)
    Mp = point_correspondences.shape[0]
    print('draw src:{}, tar:{}, correspondences:{}/{}'.format(
        src_idx,tar_idx,point_correspondences.shape[0],correspondences.shape[0]))
    src_pts = np.asarray(src_graph[src_idx].cloud.points)[point_correspondences[:,0]]
    tar_pts = np.asarray(tar_graph[tar_idx].cloud.points)[point_correspondences[:,1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack((src_pts,tar_pts))),
        lines=o3d.utility.Vector2iVector(np.arange(Mp*2).reshape(2,Mp).transpose(1,0)))
    line_set.paint_uniform_color((0,0,1))
    
    return line_set

def save_nodes_edges(graph:dict,output_dir:str):
    with open(os.path.join(output_dir,'nodes.csv'),'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=['node_id','label','score','center','quaternion','extent','cloud_dir','cloud_size'])
        writer.writeheader()
        for idx,inst in graph['nodes'].items():
            quat = R.from_matrix(inst.box.R.copy()).as_quat()
            centroid = inst.cloud.get_center()
            label_str = inst.label
            if ' ' in label_str:
                label_str = label_str.replace(' ','_')
            writer.writerow({'node_id':idx,'label':label_str,'score':inst.score,
                            'center':np.array2string(centroid,precision=6,separator=',')[1:-1],
                            'quaternion':np.array2string(quat,precision=6,separator=',')[1:-1],
                            'extent':np.array2string(inst.box.extent,precision=6,separator=',')[1:-1],
                            'cloud_dir':inst.cloud_dir,
                            'cloud_size':len(inst.cloud.points)})
        csvfile.close()
    
    edges_t = torch.tensor(graph['edges']).long()
    global_edges_t = torch.tensor(graph['global_edges']).float()
    torch.save({'edges':edges_t,
                'global_edges':global_edges_t},
               os.path.join(output_dir,'edges.pth'))
        
    xyzi = torch.tensor(graph['xyzi'])
    torch.save(xyzi,os.path.join(output_dir,'xyzi.pth'))

def process_scene_pair(
                    src_scene_dir:str,
                    ref_scene_dir:str,
                    match_f_dir:str,
                    gt_transform_dir:str='',
                    min_iou:float=0.2, 
                    save_nodes:bool=True,
                    compute_matches:bool=True):
    
    assert os.path.exists(os.path.join(src_scene_dir,'instance_box.txt'))
    assert os.path.exists(os.path.join(ref_scene_dir,'instance_box.txt'))

    src_scene_name = os.path.basename(src_scene_dir)
    ref_scene_name = os.path.basename(ref_scene_dir)
    print('--------------------{}-{}---------------------'.format(src_scene_name,
                                                                  ref_scene_name))

    # load
    src_graph = load_raw_scene_graph(src_scene_dir)
    ref_graph = load_raw_scene_graph(ref_scene_dir)
    src_graph = generate_edges(src_scene_name,src_graph)
    ref_graph = generate_edges(ref_scene_name,ref_graph)
    
    #
    src_graph['descriptive_instances'] = semantic_sampling(src_graph,
                                                           max_count=50,
                                                           min_size=1000)
    ref_graph['descriptive_instances'] = semantic_sampling(ref_graph,
                                                           max_count=50,
                                                           min_size=1000)
    src_graph = generate_global_edges(src_scene_name,src_graph)
    ref_graph = generate_global_edges(ref_scene_name,ref_graph)
    
    # Save Graph: nodes.csv and edges.csv
    if save_nodes:
        save_nodes_edges(src_graph,src_scene_dir)
        save_nodes_edges(ref_graph,ref_scene_dir)  
          
    # find association
    if gt_transform_dir!='':
        T_ref_src = np.loadtxt(gt_transform_dir)
        transform_scene_graph(src_graph,T_ref_src)
    
    if compute_matches:
        matches, correspondences, global_map_iou, ious_map = \
            find_association(src_graph,ref_graph,min_iou) # [(src_idx, tar_idx)], 
        assert len(matches)>0, 'no matches found'
    else:
        matches = []
        correspondences = []
        global_map_iou = 0.0
        ious_map = {'src_names':[],'tar_names':[],'iou':torch.zeros(0)}
    
    # Save GT: matches_ab.pth
    if match_f_dir != '':
        if len(matches)<1 or ious_map is None:
            print('No matches found')
            assert False
            return None
        if len(src_graph['nodes'])<1 or len(ref_graph['nodes'])<1:
            print('No nodes found')
            return None
        if len(src_graph['edges'])<1 or len(ref_graph['edges'])<1:
            print('No edges found')
            return None
        
        matches_t = torch.tensor(matches) # (M,3)
        # correspondences_t = torch.tensor(correspondences) # (P,4)
        global_map_iou_t = torch.tensor(global_map_iou)
        assert matches_t.min()>=0, 'negative index in matches'
        
        torch.save((matches_t,ious_map,global_map_iou_t),
                   match_f_dir)
        print('Saved {} node matches for {}-{}'.format(len(matches), 
                                                    src_scene_name,
                                                    ref_scene_name))
        
    return {'src':src_graph,
            'ref':ref_graph,
            'matches':matches, 
            'correspondences':correspondences}

def process_scene_thread(args):
    src_scene_dir, ref_scene_dir, match_folder, gt_transform_folder, min_iou, save_nodes, compute_matches = args
    src_scene_name = os.path.basename(src_scene_dir)
    ref_scene_name = os.path.basename(ref_scene_dir)
    match_f = os.path.join(match_folder,
                           '{}-{}.pth'.format(src_scene_name,ref_scene_name))
    if gt_transform_folder=='':
        gt_transform_f = ''
    else:
        gt_transform_f = os.path.join(gt_transform_folder,
                                  '{}-{}.txt'.format(src_scene_name,ref_scene_name))
    
    process_scene_pair(src_scene_dir,
                        ref_scene_dir,
                        match_f,
                        gt_transform_f,
                        min_iou,
                        save_nodes,
                        compute_matches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate GT association and visualize(optional) for ScanNet')
    parser.add_argument('--graphroot', type=str, default='/data2/ScanNetGraph')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--split_file', type=str, default='val')
    parser.add_argument('--min_iou', type=float, default=0.3,help='minimum iou to be considered as valid')
    parser.add_argument('--transform', action='store_true',help='transform src graph to tar graph')
    parser.add_argument('--viz_edges',type=str,help='default:None, edges, global_edges or None')
    parser.add_argument('--viz_match', action='store_true',help='visualize gt matches')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--WEBRTC_IP', type=str)
    parser.add_argument('--save_nodes', action='store_true',help='save nodes and edges')
    parser.add_argument('--compute_matches', action='store_true',help='compute matches')

    args = parser.parse_args()
    if args.transform:
        gt_transform_folder = os.path.join(args.graphroot,'gt')
        assert os.path.exists(gt_transform_folder), 'gt transform folder not exist'
    else:
        gt_transform_folder = ''
        
    if 'RioGraph' in args.graphroot:
        assert args.transform, 'RioGraph dataset requires gt transform'
    
    match_folder= os.path.join(args.graphroot,'matches')
    scan_pairs = read_scan_pairs(os.path.join(args.graphroot, 'splits', args.split_file + '.txt'))
    # scan_pairs = [['scene0064_00c','scene0064_00d']]
    print('Generate GT and graph for {} pairs of scans'.format(len(scan_pairs)))

    # for pair in scan_pairs:
    #     process_scene_thread((os.path.join(args.graphroot,args.split,pair[0]),
    #                         os.path.join(args.graphroot,args.split,pair[1]),
    #                         os.path.join(args.graphroot,'matches'),
    #                         gt_transform_folder,
    #                         args.min_iou,
    #                         args.save_nodes, 
    #                         args.compute_matches))
    #     break
    # exit(0)

    import multiprocessing as mp
    p = mp.Pool(processes=16)
    p.map(process_scene_thread,
            [(os.path.join(args.graphroot,args.split, pair[0]),
            os.path.join(args.graphroot,args.split, pair[1]),
            os.path.join(args.graphroot,'matches'),
            gt_transform_folder,
            args.min_iou,
            args.save_nodes,
            args.compute_matches) for pair in scan_pairs])    
    p.close()
    p.join()
    print('finished {} scans'.format(len(scan_pairs)))
    
