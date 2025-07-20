import os
from os.path import join as osp
from sgreg.utils.utils import read_scan_pairs
import shutil

if __name__=='__main__':
    print('Select X scenes to publish these sequences online.')
    ########################## ARGS ##########################
    INPUT_DATAROOT = '/data2/ScanNetGraph'
    OUTPUT_DATAROOT = '/data5/cliuci/dataset/ScanNetGraphMini'
    SPLIT='public_validate.txt'
    MOVE_GT = False
    #########################################################
    
    scan_pairs = read_scan_pairs(osp(INPUT_DATAROOT,'splits',SPLIT))
    print('Load {} scan pairs'.format(len(scan_pairs)))
    
    for pair in scan_pairs:
        print('------- Processing {}-{} -------'.format(pair[0],pair[1]))
        output_src_folder = osp(OUTPUT_DATAROOT,'val',pair[0])
        output_ref_folder = osp(OUTPUT_DATAROOT,'val',pair[1])
        if os.path.exists(output_src_folder)==False:
            shutil.copytree(osp(INPUT_DATAROOT,'val',pair[0]),
                            output_src_folder)
            print('Copy {}'.format(output_src_folder))
            
        if os.path.exists(output_ref_folder)==False:
            shutil.copytree(osp(INPUT_DATAROOT,'val',pair[1]),
                            output_ref_folder)
            print('Copy {}'.format(output_ref_folder))
        
        # GT transform
        if MOVE_GT:
            input_gt_file = osp(INPUT_DATAROOT,'gt', '{}-{}.txt'.format(pair[0],pair[1]))
            output_gt_file = osp(OUTPUT_DATAROOT,'gt', '{}-{}.txt'.format(pair[0],pair[1]))
            if os.path.exists(output_gt_file)==False:
                shutil.copyfile(input_gt_file,
                                output_gt_file)
                print('Copy the gt file {}'.format(output_gt_file))
        
        # GT matches
        input_match_file = osp(INPUT_DATAROOT,'matches', '{}-{}.pth'.format(pair[0],pair[1]))
        output_match_file = osp(OUTPUT_DATAROOT,'matches', '{}-{}.pth'.format(pair[0],pair[1]))
        if os.path.exists(output_match_file)==False:
            shutil.copyfile(input_match_file,
                            output_match_file)
            print('Copy the gt matches {}'.format(output_match_file))

    print('------- Finish -------')
    