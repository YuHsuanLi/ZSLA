import numpy as np
import torch 
import torch.nn as nn
from networks import model_baseline
import os
import matplotlib
import time
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
from torch import optim
import json
import time
import pickle
from tools import utils, training_tools, loss, test
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import argparse

SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.is_test = args.test
        self.is_train = not self.is_test
        self.output_PATH = args.output_PATH
        self.info_PATH = args.info_PATH
        self.data_type = args.data_type
        self.diag_upto = args.diag_upto
        self.channels = args.channels
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu") 
        self.method = args.method
        self.set_seen_attribute_idx()
        self.use_location_info = args.use_location_info
        self.batch_size = args.batch_size
        self.model = model_baseline.ATT_zeroshot(
            channels = self.channels,
            fix_weight = True, 
            encoder_type = 'resnet101', 
            pretrained = True,
            use_attribute_idx = self.use_attribute_idx,
            device = self.device,
            use_location_info = self.use_location_info, 
            data_type = self.data_type, 
            method = self.method, 
        ).to(self.device)
        self.set_data()
        self.optimizer = optim.Adam([{'params': self.model.base_attribue_classifier, 'lr':1e-4, 'weight_decay':1e-4, 'betas':(0.9, 0.999)}])     
        self.attribute_vectors = self.model.att_batt_matrix.to(self.device)
        self.baseline_PATH = args.baseline_PATH
    def set_seen_attribute_idx(self):
        self.use_attribute_idx = utils.get_use_attribute_idx(self.data_type, diag_upto=self.diag_upto, use_all = False)
        if self.data_type == 'CUB':
            self.all_attribute_idx = utils.get_use_attribute_idx(self.data_type, diag_upto=14, use_all = False)
            self.not_use_attribute_idx = np.array(list(set(self.all_attribute_idx).difference(set(self.use_attribute_idx))))
        elif self.data_type == 'alpha-CLEVR':
            self.all_attribute_idx = utils.get_use_attribute_idx(self.data_type, diag_upto=-1, use_all = True)
            self.not_use_attribute_idx = np.array(list(set(self.all_attribute_idx).difference(set(self.use_attribute_idx))))
            
    def set_data(self):
        '''
        dataloader
        '''
        if self.data_type == 'CUB':
            trX_collect, trY_collect, trP_collect, trA_collect, valX_collect, valY_collect, valP_collect, valA_collect, class_description_train, class_description_val, attribute_name, attribute_part_label_tensor = training_tools.get_CUB_dataset()
            attribute_part_label_tensor = torch.from_numpy(attribute_part_label_tensor).float()
        elif self.data_type == 'alpha-CLEVR':
            self.att_num = 24
            trX_collect, trY_collect, trP_collect, trA_collect, test_seenX_collect, test_seenY_collect, test_seenP_collect, test_seenA_collect, test_unseenX_collect, test_unseenY_collect, test_unseenP_collect, test_unseenA_collect = training_tools.get_alpha_CLEVR_dataset(self.info_PATH)
            #valX_collect = np.concatenate([test_seenX_collect, test_unseenX_collect], 0)
            #valY_collect = np.concatenate([test_seenY_collect, test_unseenY_collect], 0)
            #valP_collect = np.concatenate([test_seenP_collect, test_unseenP_collect], 0)
            #valA_collect = np.concatenate([test_seenA_collect, test_unseenA_collect], 0)
            attribute_part_label_tensor = None
            clean_trX_collect, clean_trY_collect, clean_trP_collect, clean_trA_collect, clean_test_seenX_collect, clean_test_seenY_collect, clean_test_seenP_collect, clean_test_seenA_collect, clean_test_unseenX_collect, clean_test_unseenY_collect, clean_test_unseenP_collect, clean_test_unseenA_collect = training_tools.get_alpha_CLEVR_dataset()
            valX_collect = np.concatenate([clean_test_seenX_collect, clean_test_unseenX_collect], 0)
            valY_collect = np.concatenate([clean_test_seenY_collect, clean_test_unseenY_collect], 0)
            valP_collect = np.concatenate([clean_test_seenP_collect, clean_test_unseenP_collect], 0)
            valA_collect = np.concatenate([clean_test_seenA_collect, clean_test_unseenA_collect], 0)  
                    
        self.train_data = (trX_collect, trY_collect, trP_collect, trA_collect, attribute_part_label_tensor)
        self.test_data = (valX_collect, valY_collect, valP_collect, valA_collect, attribute_part_label_tensor)
        self.train_num = trX_collect.shape[0]
        self.test_num = valX_collect.shape[0]
     
    def exe(self):
        if self.is_train:
            self.train()
        else:
            self.test()
            
    def test(self):
        '''
        testing
        '''
        self.model = torch.load(self.baseline_PATH,  map_location=self.device).to(self.device)
        self.model.device = self.device
        result = {} 
        self.model.eval()
        result = test.baseline_evaluation(self.model, self.test_data, self.use_attribute_idx, self.not_use_attribute_idx)
        print('--------testing---------')
        print('mAUROC_seen: ', result['mAUROC']['seen'])
        print('mAUROC_unseen: ', result['mAUROC']['unseen'])
        print('mAP_seen: ', result['mAP']['seen'])
        print('mAP_unseen: ', result['mAP']['unseen'])
        print('mLA_seen: ', result['mLA']['seen'])
        print('mLA_unseen: ', result['mLA']['unseen'])        
        
    def train(self): 
        '''
        file copy and logger
        '''
        if(os.path.isdir(self.output_PATH)==False):
            os.mkdir(self.output_PATH)
        else: #if self.output_PATH != '/eva_data/hdd4/yu_hsuan_li/logic_kernel/output/' + 'test':
            print('have the folder already')
            exit(0)
        self.logger = utils.Tensorboard(self.output_PATH+'/logdir')
        backup_dir = os.path.join(self.output_PATH, 'backup_files')
        #os.makedirs(backup_dir, exist_ok=True)
        #os.system('cp *.py %s/' % backup_dir)
        #os.system('cp *.ipynb %s/' % backup_dir)
        #os.system('cp -r ./networks %s/' % backup_dir)
        
        '''
        training
        '''
        for epoch in range(5001):
            print('epoch: ', epoch)
            running_loss = 0
            start = time.time()
            for i, indices in enumerate (BatchSampler(SubsetRandomSampler(range(self.train_num)), self.batch_size, drop_last=False)):
                self.model.train()
                #collect = trX_collect, trY_collect, trP_collect, trA_collect
                collect = self.train_data[:4]
                attribute_part_label_tensor = self.train_data[4]
                features, y, attribute_labels, part_masks = utils.get_data(self.data_type, collect, attribute_part_label_tensor, indices, self.device)              
                '''
                get image feature
                ''' 
                if self.use_location_info:
                    base_att_score, att_score, base_attention, att_attention= self.model(features, use_attribute_idx=self.use_attribute_idx, should_be_peak=part_masks) 
                else:
                    base_att_score, att_score, base_attention, att_attention = model(features, use_attribute_idx=use_attribute_idx) 
                attribute_labels = attribute_labels[:, self.use_attribute_idx]

                '''
                loss
                '''
                L_img_att = self.model.get_loss(att_score, attribute_labels)

                loss = L_img_att 
                running_loss += loss

                # updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('--------training---------')
            print('running_loss', running_loss.item()/(i+1))
            self.logger.log_scalar('running_loss',running_loss.item()/(i+1),epoch)

            end = time.time()
            print(end - start)
            # test
            running_loss = 0
            if (epoch) % 50 == 0: 
                result = {} 
                result = test.baseline_evaluation(self.model, self.test_data, self.use_attribute_idx, self.not_use_attribute_idx)
                print('--------testing---------')
                print('mAUROC_seen: ', result['mAUROC']['seen'])
                print('mAUROC_unseen: ', result['mAUROC']['unseen'])
                print('mAP_seen: ', result['mAP']['seen'])
                print('mAP_unseen: ', result['mAP']['unseen'])
                print('mLA_seen: ', result['mLA']['seen'])
                print('mLA_unseen: ', result['mLA']['unseen'])
                self.logger.log_scalar('test mAUROC_seen', result['mAUROC']['seen'],epoch)
                self.logger.log_scalar('test mAUROC_unseen', result['mAUROC']['unseen'],epoch)
                self.logger.log_scalar('test mAP_seen', result['mAP']['seen'],epoch)
                self.logger.log_scalar('test mAP_unseen', result['mAP']['unseen'],epoch)
                self.logger.log_scalar('test mLA_seen', result['mLA']['seen'],epoch)
                self.logger.log_scalar('test mLA_unseen', result['mLA']['unseen'],epoch)
            if (epoch) % 2500 == 0: 
                torch.save(self.model, self.output_PATH+'/model_' + str(epoch) +'.pth')