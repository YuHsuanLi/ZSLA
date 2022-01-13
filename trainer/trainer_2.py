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
        self.get_base_att = 'and' 
        self.get_new_att = 'avg'    
        self.is_normalized = args.is_normalized
        self.is_abs = args.is_abs
        self.channels = args.channels
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")  
        self.attributes_PATH = args.attributes_PATH
        self.set_seen_attribute_idx()
        self.model = logic_network.LN(
            channels = self.channels, 
            get_base_att = self.get_base_att, 
            get_new_att = self.get_new_att, 
            device = self.device, 
            fix_weight=True, 
            use_attribute_idx = self.use_attribute_idx, 
            is_normalized = self.is_normalized,
            is_abs = self.is_abs,
            data_type = self.data_type)
        self.set_data()
        self.model.set_attributes(attributes_path=self.attributes_PATH)
        self.attributes = self.model.get_attributes() #att_numXCh
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4, weight_decay = 1e-4, betas = (0.5, 0.9))
        self.gate_PATH = args.gate_PATH
        
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
            att_num = 24
            trX_collect, trY_collect, trP_collect, trA_collect, test_seenX_collect, test_seenY_collect, test_seenP_collect, test_seenA_collect, test_unseenX_collect, test_unseenY_collect, test_unseenP_collect, test_unseenA_collect = training_tools.get_alpha_CLEVR_dataset() # we use clean to test
            valX_collect = np.concatenate([test_seenX_collect, test_unseenX_collect], 0)
            valY_collect = np.concatenate([test_seenY_collect, test_unseenY_collect], 0)
            valP_collect = np.concatenate([test_seenP_collect, test_unseenP_collect], 0)
            valA_collect = np.concatenate([test_seenA_collect, test_unseenA_collect], 0)
            attribute_part_label_tensor = None
                    
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
        self.model = torch.load(self.gate_PATH,  map_location=self.device).to(self.device)
        self.model.device = self.device
        self.model.encoder.device = self.device
        self.model.Logic_Gate.device = self.device
        self.model = self.model.to(self.device)
        result = {} 
        self.model.eval()
        result = test.gate_evaluation(self.model, self.test_data, self.use_attribute_idx, self.not_use_attribute_idx)
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
        epoch = 0
        total_epoch = 100001 if self.get_base_att == 'linear' else 200001
        while(epoch <= total_epoch):
            self.model.train()
            if epoch%50 == 0:
                print('Epoch ',epoch,' started.')

            s_tick = time.time()   
            self.model.train()      

            #get base attributes
            base_attributes = self.model.get_base_attributes(self.attributes)
                        
            #get attributes            
            new_attributes = self.model.get_new_attributes(base_attributes)
            
            #loss: attribute_gen_loss
            attribute_gen_loss = loss.distance(self.attributes[self.use_attribute_idx], new_attributes[self.use_attribute_idx], mode = 'l2').mean() 
            total_loss = attribute_gen_loss 
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if (epoch%50==0):         
                print('total_loss: ' + str(total_loss))
                print('-------------------------')
                self.logger.log_scalar('total_loss',total_loss.item(),epoch)
                       
            if epoch==5000 or epoch%10000 == 0:
                print('--------testing--------')          
                result = {} 
                self.model.eval()
                result = test.gate_evaluation(self.model, self.test_data, self.use_attribute_idx, self.not_use_attribute_idx)
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

            if epoch%100000 == 0:
                torch.save(self.model, self.output_PATH+'/model_' + str(epoch) +'.pth')
            #exit(0)    
            epoch += 1