import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
import numpy as np
import copy
import random
SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def CE(pred,y, balanced=False):
    if balanced:
         return torch.sum(-y*torch.log(torch.clamp(pred,1e-6,1e+6)))/(torch.sum(y)+1e-6) +torch.sum(-(1-y)*torch.log(torch.clamp(1-pred,1e-6,1e+6)))/(torch.sum(1-pred)+1e-6)

    return torch.mean(torch.sum(-y*torch.log(torch.clamp(pred,1e-6))-(1-y)*torch.log(torch.clamp(1-pred,1e-6)),-1))

def similarity(vector1, vector2, sim_scale, sigmoid=True, mode = 'cos_sim'):
    '''
    vector1: tensor[?XCh]
    vector2: tensor[?XCh]
    sim_scale: int, a scalar used when compute the similarity, expand the raw similarity *sim_scale before sigmoid
    sigmoid: bool, use sigmoid or not
    return: tensor[?]
    '''
    if mode == 'cos_sim':
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * sim_scale
        if sigmoid:
            return result.sigmoid()
        return result
    elif mode == 'dot_sim':
        result = vector1@vector2.permute(1,0)
        sim = torch.nn.functional.softmax(result, -1)
    return sim

def distance(vector1, vector2, mode = 'l2'):
    '''
    vector1: tensor[?XCh]
    vector2: tensor[?XCh]
    return: tensor[?]
    '''        
    if mode == 'l2':
        return F.pairwise_distance(vector1, vector2)

def same_base_att_loss(base_attributes1, base_attributes2, temperature = 0.1,  base_temperature = 0.07, const=0):
    #temperature = 0.1
    #base_temperature = 0.07
    batch_size = base_attributes1.shape[0]
    '''
    same_base_att_feature = torch.zeros_like(torch.cat((base_attributes1, base_attributes2), dim=0))
    idx1 = [2*i for i in range(batch_size)]
    idx2 = [2*i+1 for i in range(batch_size)]
    same_base_att_feature[idx1] = base_attributes1
    same_base_att_feature[idx2] = base_attributes2
    '''
    same_base_att_feature = torch.cat((base_attributes1, base_attributes2), dim=0)
    same_base_att_label = torch.arange(batch_size)
    #print(same_base_att_label)

    # normalize
    same_base_att_feature = F.normalize(same_base_att_feature, dim=1)
    same_base_att_label = torch.as_tensor(same_base_att_label)    
    labels = same_base_att_label.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(base_attributes1.device)
    contrast_count = 2
    anchor_feature = same_base_att_feature
    anchor_count = contrast_count
    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, same_base_att_feature.T), temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    #print(logits.shape)
    #exit(0)
    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(base_attributes1.device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss

def same_base_att_loss_l2(base_attributes1, base_attributes2):
    loss_tmp = distance(base_attributes1, base_attributes2, mode = 'l2')
    same_base_att_loss = loss_tmp.mean() 
    return same_base_att_loss
    
def reg_loss(model, samples):    
    and_self = model.Logic_Gate.logic_and(samples, samples)
    or_self = model.Logic_Gate.logic_or(samples, samples)
    L_and_self = torch.mean(distance(samples, and_self))
    L_or_self = torch.mean(distance(samples, or_self))

    random_pair_idx = torch.randperm(samples.shape[0])
    shuffled_samples = torch.chunk(samples[random_pair_idx],2,0)
    
    A, B = shuffled_samples[0], shuffled_samples[1]
    if(samples.shape[0]%2!=0):
        A = A[:-1]
    
    A_or_B = model.Logic_Gate.logic_or(A, B)
    A_and_B = model.Logic_Gate.logic_and(A, B)
    B_or_A = model.Logic_Gate.logic_or(B, A)
    B_and_A = model.Logic_Gate.logic_and(B, A)

    A_or_B_and_A = model.Logic_Gate.logic_and(A_or_B, A)
    A_or_B_or_A = model.Logic_Gate.logic_or(A_or_B, A)
    A_and_B_and_A = model.Logic_Gate.logic_and(A_and_B, A)
    A_and_B_or_A = model.Logic_Gate.logic_or(A_and_B, A)

    L_or_and = torch.mean(distance(A, A_or_B_and_A))
    L_or_or = torch.mean(distance(A_or_B, A_or_B_or_A))
    L_and_and = torch.mean(distance(A_and_B, A_and_B_and_A))
    L_and_or = torch.mean(distance(A, A_and_B_or_A))

    L_or_exchange = torch.mean(distance(A_or_B, B_or_A))
    L_and_exchange = torch.mean(distance(A_and_B, B_and_A))
    return L_and_self, L_or_self, L_or_and, L_or_or, L_and_and, L_and_or, L_or_exchange, L_and_exchange

def reg_loss_plus(model, samples):            
    random_pair_idx = torch.randperm(samples.shape[0])
    shuffled_samples = torch.chunk(samples[random_pair_idx],2,0)
    A, B = shuffled_samples[0], shuffled_samples[1]   
    if(samples.shape[0]%2!=0):
        A = A[:-1]
    A_or_B = model.Logic_Gate.logic_or(A, B)
    L_or_plus = torch.mean(distance(A_or_B, (A+B)/2))
    return  L_or_plus

def get_middle_graph(kernel_size = 7, device='cpu'):
    raw_graph = torch.zeros((2 * kernel_size -1, 2 * kernel_size -1))
    for x in range(- kernel_size + 1, kernel_size):
        for y in range(- kernel_size + 1, kernel_size):
            raw_graph[x + (kernel_size - 1), y + (kernel_size - 1)] = x**2 + y**2
    middle_graph = torch.zeros((kernel_size**2, kernel_size, kernel_size))
    for x in range(kernel_size):
        for y in range(kernel_size):
            middle_graph[x*kernel_size + y, :, :] = \
                raw_graph[kernel_size - 1 - x: 2 * kernel_size - 1 -x, kernel_size - 1 - y: 2 * kernel_size - 1 -y]
    middle_graph = middle_graph.to(device)
    return middle_graph

def CPT_loss(attention, middle_graph, part_mask=None, sigmoid = False):
    '''
    attention: BSX16X7X7
    middle_graph: 49X7X7
    '''
    if part_mask==None:
        batch_size = attention.shape[0] #32
        att_size = attention.shape[1] #15
        peak_id = torch.argmax(attention.reshape(batch_size * att_size, -1), dim=1) #BSX37 49
        peak_mask = middle_graph[peak_id, :, :].view(batch_size, att_size, 7, 7)
        if sigmoid:
            cpt_loss = torch.mean(torch.sum(nn.Sigmoid()(attention * peak_mask), (1, 2, 3)))
        else:
            cpt_loss = torch.mean(torch.sum(attention * peak_mask, (1, 2, 3)))
        return cpt_loss
    else:
        batch_size = attention.shape[0] #32
        att_size = attention.shape[1] #15
        peak_mask = 1 - part_mask
        if sigmoid:
            cpt_loss = torch.mean(torch.sum(nn.Sigmoid()(attention * peak_mask), (1, 2, 3)))
        else:
            cpt_loss = torch.mean(torch.sum(attention * peak_mask, (1, 2, 3)))
        return cpt_loss
