import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
import numpy as np
import copy
from networks import encoder
from scipy.linalg import qr
import random
from torch import autograd
from tools import utils
from tools import loss as Loss
SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)



class ATT_zeroshot(nn.Module):
    def __init__(self, channels, fix_weight=True, encoder_type='resnet101', pretrained=True, use_attribute_idx=None, device='cpu', use_location_info=False, data_type='CUB', method='A-ESZSL'):
        '''
        channels: int, Ch
        fix_weight: bool, decide the encoder is trainable or not
        encoder_type: str, the architecture of encoder
        pretrained: bool, decide if load weight that pretrained on ImageNet
        use_attribute_idx: numpy[att_num]
        device: str, the device name
        use_attribute_idx: numpy[att_num]
        data_type: str, the datest we use (CUB or ɑ-CLEVR)
        method: str, the baseline method (A-LAGO or A-ESZSL)
        '''
        super(ATT_zeroshot, self).__init__()
        self.encoder = encoder.encoder(fix_weight=fix_weight, encoder_type=encoder_type, pretrained=pretrained).to(device)
        self.channels = channels
        self.kernel_size = 7        
        self.use_location_info = use_location_info
        self.use_attribute_idx = use_attribute_idx
        self.device = device
        self.data_type = data_type
        self.method = method
        self.att_batt_matrix = torch.tensor(utils.get_attribute_vectors(self.data_type)) #312X88, 100X20
        if data_type=='CUB':
            self.att_num = 312
            self.base_att_num = 88 # we actually consider only 31 base attributes, we left the space for 57 (88-31) attributes, but we won't train them 
            self.use_base_attribute_idx = utils.get_cub_base_idx()
        elif data_type == 'alpha-CLEVR':
            self.att_num = 24
            self.base_att_num = 11
            self.use_base_attribute_idx = np.array(list(set(range(11)))) 
        self.use_att_num = self.use_attribute_idx.shape[0]
        self.use_base_att_num = self.use_base_attribute_idx.shape[0]
        self.base_attribue_classifier = nn.Parameter(2e-4 * torch.rand([self.use_base_att_num, channels]), requires_grad=True)
        
    def get_base_attributes_from_buffer(self):        
        '''
        if want to get base attributes, call this function
        this function will return all base attributes but if the attribute is unseen, it will be zero
        '''
        base_att = torch.zeros(self.base_att_num, self.channels).to(self.device)
        base_att[self.use_base_attribute_idx] = self.base_attribue_classifier

        return base_att
        
       
        
    def get_att_attention(self, base_attention, features):
        '''
        base_attention: tensor[BSXbase_att_numX7X7]
        features: tensor[BSXChX7X7]
        '''
        BS = base_attention.shape[0]
        att_batt_matrix = self.att_batt_matrix.to(self.device) #312X88 or 100X20
        batt_att_matrix = self.att_batt_matrix.permute(1, 0).to(self.device) #88X312 or 20X100 
        
        if self.method == 'A-ESZSL':
            base_attention_permute = base_attention.permute(0, 2, 3, 1)
            att_attention = base_attention_permute@batt_att_matrix.float() # cub:BSX7X7X312 
            att_attention = att_attention.permute(0, 3, 1, 2) # cub:BSX312X7X7 
            return att_attention
        elif self.method == 'A-LAGO':
            base_attention = torch.sigmoid(base_attention)
            base_attention_permute = base_attention.permute(0, 2, 3, 1) # cub:BSX7X7X88
            att_batt_expand = torch.einsum("ilmj,kj->ilmkj", (base_attention_permute, att_batt_matrix)) + (1-att_batt_matrix).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand((BS, 7, 7, att_batt_matrix.shape[0], att_batt_matrix.shape[1])) # cub:BSX7X7X312X88
            att_batt_expand = att_batt_expand[:, :, :, :, self.use_base_attribute_idx]
            att_attention = torch.prod(att_batt_expand, dim=-1) # cub:BSX7X7X312
            att_attention = att_attention.permute(0, 3, 1, 2) # cub:BSX312X7X7
            return att_attention
    
    def forward(self, features, use_attribute_idx=np.array([]), should_be_peak=np.array([])):
        '''   
        features: tensor[BSXChX7X7]
        use_attribute_idx: numpy[seen_att_num]
        should_be_peak: tensor[BSXatt_numX49]
        ret: base_att_score, att_score, base_attention, att_attention (without sigmoid)
        '''
        base_att = self.get_base_attributes_from_buffer()
        base_att = base_att.unsqueeze(-1).unsqueeze(-1)
        if use_attribute_idx.shape[0] == 0:
            use_attribute_idx = model.use_attribute_idx
        batch_size = features.size(0)    
        
        base_attention= F.conv2d(input=features, weight=base_att)  # 64Xbase_att_numX7X7
        base_att_score = F.max_pool2d(base_attention, kernel_size=7).view(batch_size, -1) #BSX88       
        att_attention = self.get_att_attention(base_attention, features)
        if should_be_peak.shape[0] == 0: # without location information
            sim = F.max_pool2d(att_attention, kernel_size=self.kernel_size).view(batch_size, -1) #BSX312
        else:
            flatten_attention = att_attention.reshape((batch_size, self.att_num, 49))
            should_be_peak = should_be_peak.clone().detach() # BsXatt_numX49
            feature_pick_peak_by_label = torch.sum(flatten_attention * should_be_peak, -1) / torch.clamp(torch.sum(should_be_peak, -1), 1e-8)
            feature_average = torch.mean(flatten_attention, -1)  
            attribute_part_show_or_not = (torch.sum(should_be_peak, -1) != 0).float() # BsXatt_num, 1 if the parts for the attribute show in an image, else 0.
            sim = (attribute_part_show_or_not * feature_pick_peak_by_label) + ((1 - attribute_part_show_or_not) * feature_average)
        
        att_score = sim        

        return base_att_score[:, self.use_base_attribute_idx], att_score[:, use_attribute_idx], base_attention[:, self.use_base_attribute_idx], att_attention[:, use_attribute_idx]
    
    def get_loss(self, att_score, attribute_labels):
        if self.method == 'A-ESZSL':
            image_attr_pred =  torch.sigmoid(att_score)
            loss = Loss.CE(image_attr_pred, attribute_labels, balanced=False)
            return loss
        elif self.method == 'A-LAGO':
            image_attr_pred =  torch.sigmoid(25*(2*att_score-1)) #25->156 
            loss = Loss.CE(image_attr_pred, attribute_labels, balanced=False)
            return loss
  
    