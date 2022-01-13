import os
import argparse
import numpy as np
from trainer.trainer_1 import Trainer as trainer_1
from trainer.trainer_2 import Trainer as trainer_2
from trainer.trainer_baseline import Trainer as trainer_baseline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--baseline', action='store_true', default=False)
    parser.add_argument('--method', default='A-ESZSL', type=str, choices=['A-ESZSL', 'A-LAGO']) 
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--stage', default=1, type=int, choices=[1, 2])
    parser.add_argument('--data_type', default='CUB', type=str, choices=['CUB', 'alpha-CLEVR'])      
    parser.add_argument('--diag_upto', default=1, type=int)
    parser.add_argument('--is_normalized', action='store_true', default=False)
    parser.add_argument('--is_abs', action='store_true', default=False)
    parser.add_argument('--channels', default=2048, type=int)
    parser.add_argument('--use_location_info', action='store_true', default=False)
    parser.add_argument('--cpt_weight', default=0.2, type=float)
    parser.add_argument('--umc_start', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)    
    # currently, our model is using resnet101 pretrain on ImageNet and fix it
    #parser.add_argument('--encoder_type', default='resnet101', type=str)
    #parser.add_argument('--pretrained', action='store_true', default=False)
    #parser.add_argument('--fix_weight', action='store_true', default=False)  
    parser.add_argument('--output_PATH', default='/eva_data/hdd4/yu_hsuan_li/logic_kernel/output/test', type=str)
    parser.add_argument('--attributes_PATH', default='/eva_data/hdd4/yu_hsuan_li/logic_kernel/output/cvpr/' + 'cub' + '/attributes/attributes_v1_umc_0.2/' + 'classifier_10000.pth', type=str)
    parser.add_argument('--gate_PATH', default='/eva_data/hdd4/yu_hsuan_li/logic_kernel/output/clean_code_check_stage2/model_200000.pth', type=str)
    parser.add_argument('--baseline_PATH', default='/eva_data/hdd4/yu_hsuan_li/logic_kernel/output/A_LAGO/model_5000.pth', type=str)
    parser.add_argument('--info_PATH', default=None, type=str)
    args = parser.parse_args()
    
    if args.baseline:
        trainer = trainer_baseline(args)
    else:
        if args.stage == 1:
            trainer = trainer_1(args)
        elif args.stage == 2:
            trainer = trainer_2(args)
        else:
            raise ValueError('Please set the correct method.')
    
    trainer.exe()