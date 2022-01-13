import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
import numpy as np
from networks import transformer_vit as logic_gate
import copy
from networks import encoder
from scipy.linalg import qr
import random
from torch import autograd
from tools import utils

SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

class LN(nn.Module):
    def __init__(self, channels=2048, get_base_att = 'and', get_new_att = 'avg', device = 'cpu', fix_weight=True, use_attribute_idx=None, is_normalized = False, encoder_type='resnet101', pretrained=True, use_location_info=False, is_abs=True, data_type = 'CUB'):
        '''
        channels: int, Ch
        get_base_att: str, the way we get our base attributes from attributes
        get_new_att: str, the way we get our attributes from base attributes
        device: str, the device name
        fix_weight: bool, decide the encoder is trainable or not
        use_attribute_idx: numpy[att_num]
        is_normalized: bool, apply L2-normalization on attribute detectors or not
        encoder_type: str, the architecture of encoder
        pretrained: bool, decide if load weight that pretrained on ImageNet
        is_abs: bool, apply element-wise absolute-value operator on attribute/base-attribute detectors or not
        data_type: str, the datest we use (CUB or ɑ-CLEVR)
        '''
        # by input
        super().__init__()    
        self.channels = channels
        self.get_base_att = get_base_att
        self.get_new_att = get_new_att
        self.device = device
        self.encoder = encoder.encoder(fix_weight=fix_weight, encoder_type=encoder_type, pretrained=pretrained).to(device)
        self.use_attribute_idx = use_attribute_idx  
        self.is_normalized = is_normalized
        self.muti_class_scale_value = 25
        self.single_class_scale_value = 25 
        self.is_abs = is_abs
        self.data_type = data_type         
        self.att_batt_matrix = utils.get_attribute_vectors(self.data_type) #根據不同的dataset會有不同的att_batt_matrix和base_attribute_name #312X88 or 100X20
        self.base_attribute_name = utils.get_base_attribute_name()
        self.base_att_num = self.att_batt_matrix.shape[1]
        self.total_att_num = self.att_batt_matrix.shape[0] #312
        
        # setting different model according to different get_base_att and get_base_att
        if self.get_base_att == 'and' or self.get_new_att == 'or':
            self.Logic_Gate = logic_gate.logic_gate(self.channels, 1, self.device, 'first', is_abs=self.is_abs)
        if self.get_base_att == 'linear':
            self.transform_fn = torch.nn.Linear(self.use_attribute_idx.shape[0], self.base_att_num,bias=False).to(device) #20X20 (#seen att X#base att)
        
    
    '''
    Functions__out
    '''
    def get_attributes(self):
        '''
        if want to get seen attributes trained by stage 1, call this function
        this function will return all attributes, but if the attribute is unseen, it will be a zero vector (to make sure that we do not use the unseen attributes)
        '''
        att = torch.zeros(self.total_att_num, self.channels).to(self.device)
        att[self.use_attribute_idx] = self.attributes.to(self.device)
        if self.is_normalized:
            att = F.normalize(att, dim=1)
        if self.is_abs:
            return (torch.abs(att))
        return att
    
    def get_gt_attributes(self):
        '''
        if want to get all (seen + unseen) attributes trained by stage 1, call this function
        this function will return all attributes
        '''
        
        att = self.gt_attributes.to(self.device)
        if self.is_normalized:
            att = F.normalize(att, dim=1)
        if self.is_abs:
            return (torch.abs(att))
        return att

    def set_attributes(self, attributes_path):
        '''
        set gt attributes and set attributes (the different part of 'gt attribute' and 'attribute' is that unseen attributes in 'attributes' are zero vectors )
        attributes_path: str, the path that we store the attributes we trained in stage 1
        '''   
        self.gt_attributes = torch.load(attributes_path, map_location=self.device).to(self.device).weight.clone()
        self.attributes = self.gt_attributes[self.use_attribute_idx].detach() # #32X2048

    def get_base_attributes(self, attributes, train=True):   
        '''
        attribute: tensor[att_numXCh]
        train: bool, is in training stage or not
        '''
        if self.get_base_att == 'and':         
            if self.is_normalized:
                attributes = F.normalize(attributes, dim=1)
                return F.normalize(self.get_base_attributes_by_gate(attributes, self.use_attribute_idx, self.att_batt_matrix, select_att_way = 'random', train=train), dim=1)
            else:
                return self.get_base_attributes_by_gate(attributes, self.use_attribute_idx, self.att_batt_matrix, select_att_way = 'random', train=train)
        
        if self.get_base_att == 'linear':
            if self.is_normalized:
                attributes = F.normalize(attributes, dim=1)
                return  F.normalize(torch.abs(self.transform_fn.weight @ attributes[self.use_attribute_idx]), dim=1)
            else:
                return torch.abs(self.transform_fn.weight @ attributes[self.use_attribute_idx]) #88X2048 
           
    def get_new_attributes(self, base_attributes, train=True):
        if self.get_new_att == 'avg':
            base_attributes = F.normalize(base_attributes, dim=1)
            if self.is_normalized:
                return  F.normalize(torch.from_numpy(self.att_batt_matrix.astype(np.float32)).to(self.device)/sum(self.att_batt_matrix[0]) @ base_attributes, dim=1)
            else:
                return torch.from_numpy(self.att_batt_matrix.astype(np.float32)).to(self.device)/sum(self.att_batt_matrix[0]) @ base_attributes                   
    """
    def att_cls_fast(x, a, should_be_peak=np.array([])):
        '''
        x: tensor[BSXChX7X7]
        a: tensor[att_numXCh]
        should_be_peak: tensor[BSXatt_numX49]
        return: 
            att_score: tensor[BSXatt_num]
            attention: tensor[BSXatt_numX7X7]
        '''
        BS = x.shape[0]
        att_num = a.shape[0] 
        a = a.unsqueeze(-1)
        a = a.unsqueeze(-1)
        x = torch.nn.functional.normalize(x,2,1)
        a = torch.nn.functional.normalize(a,2,1)
        attention = F.conv2d(input=x, weight=a) # BSXatt_numX7X7

        # without location information
        if should_be_peak.shape[0] == 0: 
            att_score = F.max_pool2d(attention, kernel_size=7).view(BS, -1)

        # with location information
        else: 
            flatten_attention = attention.reshape((BS, att_num, 49))
            should_be_peak = should_be_peak.clone().detach() # BSXatt_numX49
            feature_pick_peak_by_label = torch.sum(flatten_attention * should_be_peak, -1) / torch.clamp(torch.sum(should_be_peak, -1), 1e-8)
            feature_average = torch.mean(flatten_attention, -1)  

            # 1 if the parts for the attribute show in an image, else 0
            attribute_part_show_or_not = (torch.sum(should_be_peak, -1) != 0).float() # BSXatt_num             
            # if the attribute is in the image we select the right patch; othersise, use the average of all patches
            att_score = (attribute_part_show_or_not * feature_pick_peak_by_label) + ((1 - attribute_part_show_or_not) * feature_average)

        att_score = 25*(2*att_score-1)
        attention = 25*(2*attention-1)
        att_score = torch.sigmoid(att_score)
        attention = torch.sigmoid(attention)
        return att_score, attention    
    """
    
    '''
    Functions__in
    '''
    def get_base_attributes_by_gate(self, attributes, use_attribute_idx, att_batt_matrix, select_att_way = 'random', train=True):
        '''
        model: model, mainly use its logic gate
        attributes: tensor[312X2048]
        use_attribute_idx: 1d array, the accessable attribute idx
        att_batt_matrix: 2d array, [#att X #base att]
        select_att_way: string, indicating the method to select attributes to create base attributes
        train: bool, is in training stage or not
        return: tensor[20X2048], base attributes
        '''
        # copy attribute and force not use part to be zero
        attributes_copy = attributes.clone()
        not_use_attribute_idx = list(set(range(self.total_att_num)).difference(set(use_attribute_idx)))
        attributes_copy[not_use_attribute_idx] = torch.zeros_like(attributes_copy[0])

        # set what is the element to be used to 'and' , this part should generate a list with len = base_att_num
        att_idx_candidates = utils.base_attribute_could_be_and_from(att_batt_matrix, use_attribute_idx)
        base_att_num = att_batt_matrix.shape[1] #88
        if select_att_way == 'random':
            # shuffle base_idx_candidates
            [random.shuffle(att_idx_candidates[i]) for i in range(base_att_num)]
            att_idx = [att_idx_candidates[i][:2] for i in range(base_att_num)]

            # idx1 stores the first idx to be and, idx2 stores the another idx to be and           
            idx1 = [att_idx[i][0] if len(att_idx[i]) >=1 else 0 for i in range(base_att_num)] 
            idx2 = [att_idx[i][1] if len(att_idx[i]) >=2 else 0 for i in range(base_att_num)] 
            base_cannot_get = [i for i in range(base_att_num) if len(att_idx[i]) < 2]
            attribute_be_and_1 = attributes_copy[idx1].to(self.device)
            attribute_be_and_2 = attributes_copy[idx2].to(self.device)
            base_attributes = self.Logic_Gate.logic_and(attribute_be_and_1, attribute_be_and_2, train=train)
            base_attributes[base_cannot_get] = torch.zeros(len(base_cannot_get), self.channels).to(self.device)

        return base_attributes          

   
            
   


