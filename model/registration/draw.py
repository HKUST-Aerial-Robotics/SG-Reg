import os, argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def read_error_file(dir):
    with open(dir,'r') as f:
        scans =[]
        errors = []
        for line in f.readlines():
            if '#' in line:continue
            scan, rot_e, t_e = line.split(',')
            scans.append(scan)
            errors.append(np.array([rot_e,t_e]).astype(np.float32))
        errors = np.concatenate(errors,axis=0).reshape(-1,2)
        return errors
    
def print_errors(errors, method='instance register'):
    print('----------- {} ----------'.format(method))
    print('         rotation   translation')
    print('mean:     {:.3f}    {:.3f}'.format(errors[:,0].mean(),errors[:,1].mean()))
    print('max :     {:.3f}    {:.3f}'.format(errors[:,0].max(),errors[:,1].max()))

def draw_registration(errors:list, method:list):
    '''
    Input:
        - errors: (N,2) array, first column is rotation error, second column is translation error
    '''
    rotation_threshold = 2.0
    translation_threshold = 0.2
    inliners_rot = []
    inliners_t = []
    assert len(errors)==len(method)
    
    for error in errors:
        n = error.shape[0]
        tp = (error[:,0]<rotation_threshold) & (error[:,1]<translation_threshold)
        
        inliners_rot.append(tp.sum()/n)
        inliners_t.append(tp.sum()/n)
         
    # fig, (ax0,ax1) = plt.subplots(1,2)
    fig, ax0 = plt.subplots()
    ax0.bar(method,inliners_rot,color=['green','blue'],label=method)
    for i, v in enumerate(inliners_rot):
        ax0.text(i-0.1, v+0.01, str(round(v,2)))
    ax0.set_title('Rotation (<{} deg), Translation (<{}m)'.format(rotation_threshold,translation_threshold))
    ax0.set_ylabel('Recall')
    ax0.grid(True)
    
    # ax1.bar(method,inliners_t,color=['green','blue'],label=method)
    # for i, v in enumerate(inliners_t):
    #     ax1.text(i-0.1, v+0.01, str(round(v,2)))
    # ax1.set_title('Translation (<{}m)'.format(translation_threshold))
    # ax1.set_ylabel('Recall')
    # ax1.grid(True)
    
    plt.savefig('imgs/register2.png')
    
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Registration test')
    parser.add_argument('--graphroot', type=str, default='/data2/ScanNetGraph')
    # parser.add_argument('--prediction', type=str, default='lap', help='Prediction folder under graph root')
    PREDICTIONS = [] #["cross_neg_nll2","gat_lap"]
    args = parser.parse_args()
    
    geotrans_global_errors = read_error_file(os.path.join(args.graphroot,'pred','geotransform_global_errors.txt'))
    teaser_global_errors = read_error_file(os.path.join(args.graphroot,'pred','teaser_global_errors.txt'))
    # print_errors(teaser_global_errors,'global register')
    
    error_list = []
    for pred in PREDICTIONS:
        error = read_error_file(os.path.join(args.graphroot,'pred',pred,'errors_instance_register.txt'))
        print_errors(error)
        error_list.append(error)
    
    draw_registration(error_list+[geotrans_global_errors,teaser_global_errors], PREDICTIONS+['GeoTrans_G','Teaser_G'])
    
    exit(0)
    eval_file = os.path.join(args.graphroot,'pred',args.prediction,'eval.json')
    with open(eval_file,'r') as f:
        data = json.load(f)
        print(data)

        tp_residual = []
        fp_residual = []
        for inst in data:
            if inst['tp']==1:
                tp_residual.append(inst['residual'])
            else:
                fp_residual.append(inst['residual'])
        print('tp {}, fp {}'.format(len(tp_residual),len(fp_residual)))
        tp_residual = np.array(tp_residual)
        fp_residual = np.array(fp_residual)
        
        # Draw
        np.random.seed(19680801)

        n_bins = 20

        fig, (ax0) = plt.subplots()

        colors = ['green','red'] #, 'tan', 'lime']
        ax0.hist(tp_residual,n_bins, histtype='bar', color=['green'], label=['tp'], rwidth=0.5)
        ax0.hist(fp_residual,n_bins, histtype='bar', color=['red'], label=['fp'], rwidth=0.2)
        
        ax0.legend(prop={'size': 10})
        ax0.set_xlabel('residual')
        ax0.set_ylabel('Count')
        ax0.grid(True)
        ax0.set_title('Registration residual for each instance')
        
        # plt.savefig('imgs/res.png')
