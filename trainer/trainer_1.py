import numpy as np
import torch 
import torch.nn as nn
from networks import logic_network 
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
from tools import loss as Loss

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
        self.is_normalized = args.is_normalized
        self.is_abs = args.is_abs
        self.channels = args.channels
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")  
        self.use_location_info = args.use_location_info
        self.cpt_weight = args.cpt_weight
        self.umc_start = args.umc_start
        self.batch_size = args.batch_size
        #self.encoder_type = args.encoder_type
        #self.pretrained = args.pretrained
        #self.fix_weight = args.fix_weight   
        self.attributes_PATH = args.attributes_PATH
        self.set_seen_attribute_idx()        
        self.set_data()
        self.classifier = torch.nn.Linear(self.channels, self.att_num, bias=False).to(self.device)
        self.optimizer =  optim.Adam([{'params': self.classifier.parameters(), 'lr':1e-3, 'weight_decay':1e-4, 'betas':(0.5, 0.9)}, 
])
        
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
            self.att_num = 312
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
     
    def get_attributes(self, classifier_weight, is_abs=True, is_normalized = True):
        '''
        classifier_weight: tensor[att_numXCh], (the weight of) attributes
        is_abs: bool, apply element-wise absolute-value operator on attribute detectors or not
        is_normalized: bool, apply L2-normalization on attribute detectors or not
        return: tensor[att_numXCh], attributes (after normalized or absolute-value operation)
        '''
        if is_normalized:
            classifier_weight = F.normalize(classifier_weight, dim=1)
        if is_abs:
            classifier_weight = torch.abs(classifier_weight)
        return classifier_weight
    
    def exe(self):
        if self.is_train:
            self.train()
        else:
            self.test()
            
    def test(self):
        '''
        testing
        '''
        self.classifier = torch.load(self.attributes_PATH, map_location=self.device).to(self.device)     
        result = {} 
        self.classifier.eval()
        attributes = self.get_attributes(self.classifier.weight, is_abs=self.is_abs, is_normalized=self.is_normalized)
        result = test.attribute_evaluation(attributes, self.data_type, self.test_data, self.use_attribute_idx, self.not_use_attribute_idx, self.device)
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
        elif self.output_PATH != '/eva_data/hdd4/yu_hsuan_li/logic_kernel/output/' + 'test':
            print('have the folder already')
            exit(0)
        self.logger = utils.Tensorboard(self.output_PATH+'/logdir')
        backup_dir = os.path.join(self.output_PATH, 'backup_files')
        os.makedirs(backup_dir, exist_ok=True)
        os.system('cp *.py %s/' % backup_dir)
        os.system('cp *.ipynb %s/' % backup_dir)
        os.system('cp -r ./networks %s/' % backup_dir)
        
        '''
        training
        '''
        epoch_num = 10001
        for epoch in range(epoch_num):
            # train
            print('epoch: ', epoch)
            running_loss = 0
            start = time.time()
            self.classifier.train()
            for i, indices in enumerate (BatchSampler(SubsetRandomSampler(range(self.train_num)), self.batch_size, drop_last=False)):
                self.classifier.train() 
                collect = self.train_data[:4]
                attribute_part_label_tensor = self.train_data[4]
                features, y, attribute_labels, part_masks = utils.get_data(self.data_type, collect, attribute_part_label_tensor, indices, self.device)
                '''
                get image feature
                '''
                gap_feature = torch.mean(features,[-1,-2]) #BSX2048
                attributes = self.get_attributes(self.classifier.weight, is_abs=self.is_abs, is_normalized=self.is_normalized)
                if self.use_location_info:
                    image_attr_pred, attention = utils.get_att_score(features, attributes, should_be_peak=part_masks) 
                else:
                    image_attr_pred, attention = utils.get_att_score(features, attributes, should_be_peak=np.array([])) 

                '''
                loss: 
                1. img_att
                2. cpt_loss
                '''
                L_img_att = Loss.CE(image_attr_pred[:, self.use_attribute_idx], attribute_labels[:, self.use_attribute_idx], balanced=False)
                middle_graph = Loss.get_middle_graph(kernel_size = 7, device=self.device)
                graph_norm = torch.norm(middle_graph, dim=(1,2))
                cpt_loss = Loss.CPT_loss(attention[:, self.use_attribute_idx], middle_graph)
                if epoch < self.umc_start:
                    loss = L_img_att
                else:
                    loss = L_img_att + self.cpt_weight*cpt_loss 
                running_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('--------training---------')
            print('running_loss: ', running_loss.item()/(i+1))
            print('img_att_loss: ', L_img_att.item())
            print('cpt_loss: ', cpt_loss.item())

            self.logger.log_scalar('training running_loss', running_loss.item()/(i+1),epoch)
            self.logger.log_scalar('img_att_loss', L_img_att.item(),epoch)
            self.logger.log_scalar('cpt_loss', cpt_loss,epoch)
            end = time.time()
            print(end - start)
            # test
            loc_total = np.zeros(self.att_num) 
            loc_correct = np.zeros(self.att_num)
            has_gt = []
            has_pred = []
            if epoch % 200 == 0:
                self.classifier.eval()
                attributes = self.get_attributes(self.classifier.weight, is_abs=self.is_abs, is_normalized=self.is_normalized)
                result = test.attribute_evaluation(attributes, self.data_type, self.test_data, self.use_attribute_idx, self.not_use_attribute_idx, self.device)
                print('--------testing---------')
                print('mAUROC_seen: ', result['mAUROC']['seen'])
                print('mAUROC_unseen: ', result['mAUROC']['unseen'])
                print('mAP_seen: ', result['mAP']['seen'])
                print('mAP_unseen: ', result['mAP']['unseen'])
                print('mLA_seen: ', result['mLA']['seen'])
                print('mLA_unseen: ', result['mLA']['unseen'])    
                
            if epoch % 5000 == 0:
                torch.save(self.classifier, self.output_PATH+'/classifier_' + str(epoch) +'.pth')