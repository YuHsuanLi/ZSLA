import numpy as np
import numpy
import torch 
import torch as T
import torch.nn as nn
import time
from torchvision.utils import save_image
import matplotlib
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import random 
from tools import utils, training_tools, loss
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from sklearn import metrics
from tools.evaluation_tools import acc_part_evaluator_feature, retrievalAP_feature
SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)

def get_cub_attribute_part_label_tensor(test_data):
    return torch.from_numpy(test_data.attribute_part_label).float()

def mAUROC(y_true, y_pred, use_attribute_idx, not_use_attribute_idx):
    '''
    y_true: numpy[image_numXatt_num]
    y_pred: numpy[image_numXatt_num]
    use_attribute_idx: numpy[seen_att_num]
    not_use_attribute_idx: numpy[unseen_att_num]
    '''
    ret = []
    for L in range(y_true.shape[-1]):
        fpr, tpr, thresholds = metrics.roc_curve(y_true[...,L], y_pred[...,L])
        roc_auc = metrics.auc(fpr, tpr)
        ret += [roc_auc]
    ret = np.array(ret)
    seen = ret[use_attribute_idx].mean()
    unseen = ret[not_use_attribute_idx].mean()
    return seen, unseen

def mAP(y_true, y_pred, use_attribute_idx, not_use_attribute_idx):
    '''
    y_true: numpy[image_numXatt_num]
    y_pred: numpy[image_numXatt_num]
    use_attribute_idx: numpy[seen_att_num]
    not_use_attribute_idx: numpy[unseen_att_num]
    '''
    ret = []
    for L in range(y_true.shape[-1]):
        ap = retrievalAP_feature.compute_ap(y_true[...,L], y_pred[...,L])
        ret += [ap]
    ret = np.array(ret)
    seen = ret[use_attribute_idx].mean()
    unseen = ret[not_use_attribute_idx].mean()
    return seen, unseen

def mLA(loc_correct, loc_total, use_attribute_idx, not_use_attribute_idx):
    '''
    loc_correct: numpy[image_numXatt_num]
    loc_total: numpy[image_numXatt_num]
    use_attribute_idx: numpy[seen_att_num]
    not_use_attribute_idx: numpy[unseen_att_num]
    '''
    seen = np.mean(loc_correct[use_attribute_idx]/loc_total[use_attribute_idx])
    unseen = np.mean(loc_correct[not_use_attribute_idx]/loc_total[not_use_attribute_idx])
    return seen, unseen

def get_attribute_pred(data_type, attributes, data, device):
    total_att_num = attributes.shape[0]
    loc_total = np.zeros(total_att_num) 
    loc_correct = np.zeros(total_att_num)
    has_gt = []
    has_pred = []
    batch_size = 64
    valX_collect, valY_collect, valP_collect, valA_collect, attribute_part_label_tensor = data[0], data[1], data[2], data[3], data[4]
    test_num = valX_collect.shape[0]    
    for i, indices in enumerate (BatchSampler(SequentialSampler(range(test_num)), batch_size, drop_last=False)):
        with torch.no_grad():
            collect = valX_collect, valY_collect, valP_collect, valA_collect 
            features, y, attribute_labels, part_masks = utils.get_data(data_type, collect, attribute_part_label_tensor, indices, device)            
            image_attr_pred, att_attention = utils.get_att_score(features, attributes)               
            has_pred.append(image_attr_pred.cpu().detach().numpy())
            has_gt.append(attribute_labels.cpu().detach().numpy())
            loc_correct_, loc_total_ = get_attribute_loc_correct(part_masks, attribute_labels, att_attention)
            loc_correct += loc_correct_
            loc_total += loc_total_
    has_pred = np.concatenate(has_pred, axis=0)    
    has_gt = np.concatenate(has_gt, axis=0)  
    return has_gt, has_pred, loc_correct, loc_total 

def get_baseline_pred(model, data, device):
    total_att_num = model.att_num
    loc_total = np.zeros(total_att_num) 
    loc_correct = np.zeros(total_att_num)
    has_gt = []
    has_pred = []
    batch_size = 64
    use_attribute_idx = utils.get_use_attribute_idx(model.data_type, diag_upto=-1, use_all = True)
    valX_collect, valY_collect, valP_collect, valA_collect, attribute_part_label_tensor = data[0], data[1], data[2], data[3], data[4]
    test_num = valX_collect.shape[0]    
    for i, indices in enumerate (BatchSampler(SubsetRandomSampler(range(test_num)), batch_size, drop_last=False)):
        with torch.no_grad():
            collect = valX_collect, valY_collect, valP_collect, valA_collect 
            features, y, attribute_labels, part_masks = utils.get_data(model.data_type, collect, attribute_part_label_tensor, indices, model.device)            
            base_att_score, att_score, base_attention, att_attention = model(features, use_attribute_idx=use_attribute_idx) 
            '''
            if model.method == 'ESZSL_like_CE':
                # no sigmoid, which deos not affect the result of mAUROC, mAP, mLA but cause nummerical problem
                image_attr_pred = att_score
            elif model.method == 'LAGO_singleton':
                # no sigmoid, which deos not affect the result of mAUROC, mAP, mLA but cause nummerical problem
                image_attr_pred = att_score
            '''
            image_attr_pred = att_score
            att_attention =  torch.sigmoid(att_attention)

            has_pred.append(image_attr_pred.cpu().detach().numpy())
            has_gt.append(attribute_labels.cpu().detach().numpy())            
            loc_correct_, loc_total_ = get_attribute_loc_correct(part_masks, attribute_labels, att_attention)
            loc_correct += loc_correct_
            loc_total += loc_total_
    has_pred = np.concatenate(has_pred, axis=0)    
    has_gt = np.concatenate(has_gt, axis=0)  

    return has_gt, has_pred, loc_correct, loc_total 


def get_attribute_loc_correct(part_masks, attribute_labels, att_attention):
    total_att_num = attribute_labels.shape[1]
    loc_total = np.zeros(total_att_num) 
    loc_correct = np.zeros(total_att_num)
    has_attribute_idx = [np.where(attribute_label.cpu().detach().numpy()==1)[0] for attribute_label in attribute_labels] 
    attentions = att_attention.reshape(-1, total_att_num, 49)
    attentions = attentions.cpu().detach().numpy()
    pred_loc = [np.argmax(attention, -1)[idx] for attention, idx in zip(attentions, has_attribute_idx)] 

    for pred, part_mask, att_num in zip(pred_loc, part_masks, has_attribute_idx):
        att_part_mask = part_mask[att_num]
        for p, gt_mask, a in zip(pred, att_part_mask, att_num):
            g = torch.where(gt_mask==1)[0]
            p = p.item()
            if g.shape[0]!=0:
                if (p in g) or (p-1 in g) or (p+1 in g) or (p+7 in g) or (p-7 in g):                 
                    loc_correct[a] +=1   
                loc_total[a] +=1   
    return loc_correct, loc_total


def attribute_evaluation(attributes, data_type, data, use_attribute_idx, not_use_attribute_idx, device):
    att_true, att_pred, loc_correct, loc_total = get_attribute_pred(data_type, attributes, data, device)    
    mAUROC_seen, mAUROC_unseen = mAUROC(att_true, att_pred, use_attribute_idx, not_use_attribute_idx)
    mAP_seen, mAP_unseen = mAP(att_true, att_pred, use_attribute_idx, not_use_attribute_idx)
    mLA_seen, mLA_unseen = mLA(loc_correct, loc_total, use_attribute_idx, not_use_attribute_idx)
    result = {}
    result['mAUROC'] = {'seen':mAUROC_seen, 
                      'unseen':mAUROC_unseen}
    result['mAP'] = {'seen':mAP_seen, 
                      'unseen':mAP_unseen}
    result['mLA'] = {'seen':mLA_seen, 
                      'unseen':mLA_unseen}
    return result

def gate_evaluation(model, data, use_attribute_idx, not_use_attribute_idx):
    model.eval()
    attributes = model.get_attributes()   
    base_attributes = model.get_base_attributes(attributes, train=False)
    new_attributes = model.get_new_attributes(base_attributes, train=False)
    result = attribute_evaluation(new_attributes, model.data_type, data, use_attribute_idx, not_use_attribute_idx, model.device)
    return result

def baseline_evaluation(model, data, use_attribute_idx, not_use_attribute_idx):
    model.eval()
    att_true, att_pred, loc_correct, loc_total = get_baseline_pred(model, data, model.device)    
    mAUROC_seen, mAUROC_unseen = mAUROC(att_true, att_pred, use_attribute_idx, not_use_attribute_idx)
    mAP_seen, mAP_unseen = mAP(att_true, att_pred, use_attribute_idx, not_use_attribute_idx)
    mLA_seen, mLA_unseen = mLA(loc_correct, loc_total, use_attribute_idx, not_use_attribute_idx)
    result = {}
    result['mAUROC'] = {'seen':mAUROC_seen, 
                      'unseen':mAUROC_unseen}
    result['mAP'] = {'seen':mAP_seen, 
                      'unseen':mAP_unseen}
    result['mLA'] = {'seen':mLA_seen, 
                      'unseen':mLA_unseen}
    return result

"""
def get_has_pred_loc_acc(model, data, attributes_):    
    model.eval()
    attributes = model.get_attributes()
    loc_total = np.zeros(model.total_att_num) 
    loc_correct = np.zeros(model.total_att_num)
    has_gt = []
    has_pred = []
    use_attribute_idx = utils.get_use_attribute_idx(model.data_type, diag_upto=-1, use_all = True) 
    batch_size = 64
    if model.data_type == 'CUB':
        valX_collect, valY_collect, valP_collect, valA_collect, attribute_part_label_tensor = data[0], data[1], data[2], data[3], data[4]
        test_num = valX_collect.shape[0] 
    elif model.data_type == 'alpha-CLEVR':
        valX_collect, valY_collect, valP_collect, valA_collect = data[0], data[1], data[2], data[3]
        test_num = valX_collect.shape[0]    
    base_attributes = model.get_base_attributes(attributes, train=False)
    new_attributes = model.get_new_attributes(base_attributes, train=False)
    for i, indices in enumerate (BatchSampler(SequentialSampler(range(test_num)), batch_size, drop_last=False)):
        with torch.no_grad():
            collect = valX_collect, valY_collect, valP_collect, valA_collect 
            features, y, attribute_labels, part_masks = utils.get_data(model.data_type, collect, attribute_part_label_tensor, indices, model.device)
            
            #base_attributes = model.get_base_attributes(attributes, train=False)
            #new_attributes = model.get_new_attributes(base_attributes, train=False)
            image_attr_pred, att_attention = utils.get_att_score(features, new_attributes)   
            
            has_pred.append(image_attr_pred.cpu().detach().numpy())
            has_gt.append(attribute_labels.cpu().detach().numpy())
            
            '''
            att loc acc
            '''
            has_attribute_idx = [np.where(attribute_label.cpu().detach().numpy()==1)[0] for attribute_label in attribute_labels] #BSX
            attentions = att_attention.reshape(-1, model.total_att_num, 49)
            attentions = attentions.cpu().detach().numpy()
            pred_loc = [np.argmax(attention, -1)[idx] for attention, idx in zip(attentions, has_attribute_idx)] #BSX

            for pred, part_mask, att_num in zip(pred_loc, part_masks, has_attribute_idx):
                att_part_mask = part_mask[att_num]
                for p, gt_mask, a in zip(pred, att_part_mask, att_num):
                    g = torch.where(gt_mask==1)[0]
                    p = p.item()
                    if g.shape[0]!=0:
                        if (p in g) or (p-1 in g) or (p+1 in g) or (p+7 in g) or (p-7 in g):                 
                            loc_correct[a] +=1   
                        loc_total[a] +=1   
    has_pred = np.concatenate(has_pred, axis=0)    
    has_gt = np.concatenate(has_gt, axis=0)  

    return has_gt, has_pred, loc_total, loc_correct #7057X312, 7057X312, 312, 312

def get_has_loc_acc(has_gt, has_pred, loc_total, loc_correct, use_attribute_idx):
    '''
    this function will return 
    1. has micro positive 
    2. has micro negative
    3. has macro positive 
    4. has macro negative
    5. loc micro 
    6. loc macro
    '''
    possitive_correct = np.sum(np.round(has_pred*has_gt), 0) 
    negative_correct = np.sum(np.round((1-has_pred)*(1-has_gt)), 0)
    possitive_total = np.sum(has_gt, 0)
    negative_total = np.sum((1-has_gt), 0)
    has_micro_positive = sum(possitive_correct[use_attribute_idx])/sum(possitive_total[use_attribute_idx])*100
    has_micro_negative = sum(negative_correct[use_attribute_idx])/sum(negative_total[use_attribute_idx])*100
    has_macro_positive = np.mean(possitive_correct[use_attribute_idx]/possitive_total[use_attribute_idx])*100
    has_macro_negative = np.mean(negative_correct[use_attribute_idx]/negative_total[use_attribute_idx])*100
   
    loc_micro = sum(loc_correct[use_attribute_idx])/sum(loc_total[use_attribute_idx])*100
    loc_macro = np.mean(loc_correct[use_attribute_idx]/loc_total[use_attribute_idx])*100
    
    return has_micro_positive, has_micro_negative, has_macro_positive, has_macro_negative, loc_micro, loc_macro
"""