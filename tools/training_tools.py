import numpy as np
import torch 
import torch.nn as nn
from tools import Part_loader_correct as cub_dataloader
#import data_loader as fonts_dataloader
#from networks import lago_network
import os
#from utils import Tensorboard
from tools import utils
import time
import pandas as pd
import random
import torch.nn.functional as F
from torch import optim
import pickle
from networks import encoder
import scipy.io


def get_CUB_dataset():
    ########################################################
    # Get Data
    print('Loading Data ...')
    
    train_data = cub_dataloader.Class_Dataset('train_val', renumber=True, dataset_root_path='/eva_data/hdd4/yu_hsuan_li/logic_kernel/dataset/CUB')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle=True, num_workers=8)

    val_data = cub_dataloader.Class_Dataset('test_unseen', renumber=True, dataset_root_path='/eva_data/hdd4/yu_hsuan_li/logic_kernel/dataset/CUB')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = 64, shuffle=False, num_workers=8)
    
    #prepare for training class camat
    class_description_train = train_data.get_class_attribute(True)
    
    #prepare for testing class camat
    class_description_val = val_data.get_class_attribute(True)
    
    attribute_name = train_data.attribute_name
    
    attribute_part_label = train_data.attribute_part_label

    #pre-extract for feature
    try:
        print('restoring features...')
        with open('./data/CUB/Res101_patch_features.pickle','rb') as f:
            db = pickle.load(f)
        trX_collect = db['trainval_X']
        trY_collect = db['trainval_Y']
        trP_collect = db['trainval_P']
        trA_collect = db['trainval_A']
        valX_collect = db['test_X']
        valY_collect = db['test_Y'] 
        valP_collect = db['test_P'] 
        valA_collect = db['test_A'] 
    
    except:
        extractor = encoder.encoder(fix_weight=True).eval().cuda()    
        print('extracting training features...')
        trX_collect = []
        trA_collect = []
        trY_collect = []
        trP_collect = []
        for count, tr_data in enumerate(train_loader):
            print(count,'/',len(train_loader))
            tr_x = tr_data[0].cuda() 
            tr_a = tr_data[6]
            tr_y = tr_data[5]
            tr_p = tr_data[1]
            with torch.no_grad():
                trX_collect += [extractor(tr_x).cpu().detach().numpy()]
                trY_collect += [tr_y.cpu().detach().numpy()]
                trP_collect += [tr_p.cpu().detach().numpy()]
                trA_collect += [tr_a.cpu().detach().numpy()]
        trX_collect = np.concatenate(trX_collect,0)
        trY_collect = np.concatenate(trY_collect,0)
        trP_collect = np.concatenate(trP_collect,0)
        trA_collect = np.concatenate(trA_collect,0)
        
        print('extracting testing features...')
        valX_collect = []
        valY_collect = []
        valP_collect = []
        valA_collect = []
        
        for count, val_data in enumerate(val_loader):
            print(count,'/',len(val_loader))
            val_x = val_data[0].cuda() 
            val_a = val_data[6]
            val_y = val_data[5]
            val_p = val_data[1]
        
            with torch.no_grad():
                valX_collect += [extractor(val_x).cpu().detach().numpy()]
                valY_collect += [val_y.cpu().detach().numpy()]
                valP_collect += [val_p.cpu().detach().numpy()]
                valA_collect += [val_a.cpu().detach().numpy()]
        
        valX_collect = np.concatenate(valX_collect,0)
        valY_collect = np.concatenate(valY_collect,0)
        valP_collect = np.concatenate(valP_collect,0)
        valA_collect = np.concatenate(valA_collect,0)
        
        with open('Res101_patch_features.pickle','wb') as f:
            db =  {'trainval_X':trX_collect,
                   'trainval_Y':trY_collect,
                   'trainval_P':trP_collect,
                   'trainval_A':trA_collect,
                   'test_X':valX_collect,
                   'test_Y':valY_collect,
                   'test_P':valP_collect,
                   'test_A':valA_collect,}
            pickle.dump(db, f)
    
    return trX_collect, \
            trY_collect, \
            trP_collect, \
            trA_collect, \
            valX_collect, \
            valY_collect, \
            valP_collect, \
            valA_collect, \
            class_description_train, \
            class_description_val, \
            attribute_name, \
            attribute_part_label


def get_alpha_CLEVR_dataset(info_path = None): #依照GZSL分類
    #try:
    split_mat = scipy.io.loadmat('./data/alpha-CLEVR/att_splits.mat')
    print('restoring features...')
    with open('./data/alpha-CLEVR/clevr_features.pickle','rb') as f:
        db = pickle.load(f)
    X_collect = db['features']
    
    train_idx = split_mat['trainval_loc'][:936]-1 #936
    test_seen_idx = split_mat['test_seen_loc']-1 #480
    test_unseen_idx = split_mat['test_unseen_loc']-1 #2400
    train_idx = train_idx.squeeze(-1)
    test_seen_idx = test_seen_idx.squeeze(-1)
    test_unseen_idx = test_unseen_idx.squeeze(-1)
    trX_collect = X_collect[train_idx]
    test_seenX_collect = X_collect[test_seen_idx]
    test_unseenX_collect = X_collect[test_unseen_idx]
    
    
    if info_path == None:
        with open('./data/alpha-CLEVR/clevr_info.pickle','rb') as f:
            db = pickle.load(f)

        Y_collect = db['labels']
        trY_collect = Y_collect[train_idx]
        test_seenY_collect = Y_collect[test_seen_idx]
        test_unseenY_collect = Y_collect[test_unseen_idx]
        valY_collect = np.concatenate([test_seenY_collect, test_unseenY_collect])
        
        P_collect = db['locations']
        trP_collect = P_collect[train_idx]
        test_seenP_collect = P_collect[test_seen_idx]
        test_unseenP_collect = P_collect[test_unseen_idx]
        

        camat = db['camat']
        trA_collect = camat[trY_collect].squeeze()
        test_seenA_collect = camat[test_seenY_collect].squeeze()
        test_unseenA_collect = camat[test_unseenY_collect].squeeze()
    else:
        with open(info_path,'rb') as f:
            db = pickle.load(f)
        Y_collect = db['labels']
        trY_collect = Y_collect[train_idx]
        test_seenY_collect = Y_collect[test_seen_idx]
        test_unseenY_collect = Y_collect[test_unseen_idx]
        valY_collect = np.concatenate([test_seenY_collect, test_unseenY_collect])
        
        P_collect = db['locations']
        trP_collect = P_collect[train_idx]
        test_seenP_collect = P_collect[test_seen_idx]
        test_unseenP_collect = P_collect[test_unseen_idx]
        
        A_collect = db['attribute_labels']
        trA_collect = A_collect[train_idx]
        test_seenA_collect = A_collect[test_seen_idx]
        test_unseenA_collect = A_collect[test_unseen_idx]
        
    return trX_collect, \
            trY_collect, \
            trP_collect, \
            trA_collect, \
            test_seenX_collect, \
            test_seenY_collect, \
            test_seenP_collect, \
            test_seenA_collect, \
            test_unseenX_collect, \
            test_unseenY_collect, \
            test_unseenP_collect, \
            test_unseenA_collect, \


def L1(pred,y):
    return torch.mean(torch.sum(torch.abs(pred-y),-1))

def CE(pred,y, balanced=False):
    if balanced:
         return torch.sum(-y*torch.log(torch.clamp(pred,1e-6,1e+6)))/(torch.sum(y)+1e-6) +torch.sum(-(1-y)*torch.log(torch.clamp(1-pred,1e-6,1e+6)))/(torch.sum(1-pred)+1e-6)

    return torch.mean(torch.sum(-y*torch.log(torch.clamp(pred,1e-6))-(1-y)*torch.log(torch.clamp(1-pred,1e-6)),-1))

def turn_parts_to_picknum(parts):
    '''
    parts: 32X15X3
    return: 32X15
    '''
    part = parts.clone()
    part[:, :, :2] = part[:, :, :2]/32
    part[:, :, 0] = (part[:, :, 0]).to(int)*7 + (part[:, :, 1]).to(int)
    part[:, :, 1] = part[:, :, 2]
    part[:, :, 0] = torch.where(part[:, :, 1] != 0, part[:, :, 0], -1+0*part[:, :, 0])
    #part[no_att_idx, 0] = -1
    return part[:, :, 0]

