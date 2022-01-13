import numpy as np
import numpy
import torch
import torch.nn.functional as F
import random
from sklearn import metrics
from tqdm import tqdm
SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def att_acc(x, a, should_be_peak, attribute_labels, muti_class=False, loc=True):
        '''
        x: BSX7X7X2048
        a: 312X2048
        attribute_labels: BSX312
        should_be_peak: list: len(BSX312)
        '''
        att_total = np.zeros(312)
        att_correct = np.zeros(312)
        possitive_correct = np.zeros(312)
        negative_correct = np.zeros(312)
        possitive_total = np.zeros(312)
        negative_total = np.zeros(312)
        counter = 0
        correct = 0
        total = 0
        N_attr = a.shape[0]
        Bs = x.shape[0]
        a = a.unsqueeze(-1)
        a = a.unsqueeze(-1)
        x_norm = torch.nn.functional.normalize(x,2,1)
        a_norm = torch.nn.functional.normalize(a,2,1)
        attention = F.conv2d(input=x_norm, weight=a_norm) #BSX312X7X7
        if should_be_peak==None:
            sim = F.max_pool2d(attention, kernel_size=7).view(Bs, -1) #BSX312   
            attribute_score = sim
            possitive_correct += torch.sum(torch.round(attribute_score*attribute_labels), 0).cpu().detach().numpy()            
            negative_correct += torch.sum(torch.round((1-attribute_score)*(1-attribute_labels)), 0).cpu().detach().numpy()
            possitive_total += torch.sum(attribute_labels, 0).cpu().detach().numpy()
            negative_total += torch.sum((1-attribute_labels), 0).cpu().detach().numpy()
        else:
            if not loc:
                flatten_attention = attention.reshape((Bs, 312, 49))
                should_be_peak = np.array(should_be_peak)
                sim = [] # BsX312
                for flat_att, peak in zip(flatten_attention, should_be_peak):
                    peak = [p[p!=-1] for p in peak] 
                    peak = [list(range(49)) if len(p) == 0 else p for p in peak] #如果都沒有，取平均
                    peak = [np.unique(p) for p in peak] #移除重複項
                    tmp = [torch.mean(flat_att[i][peak[i]]) for i in range(312)]
                    tmp = torch.stack(tmp)
                    sim.append(tmp)
                sim = torch.stack(sim)
                attribute_score = sim
                possitive_correct += torch.sum(torch.round(attribute_score*attribute_labels), 0).cpu().detach().numpy()            
                negative_correct += torch.sum(torch.round((1-attribute_score)*(1-attribute_labels)), 0).cpu().detach().numpy()
                possitive_total += torch.sum(attribute_labels, 0).cpu().detach().numpy()
                negative_total += torch.sum((1-attribute_labels), 0).cpu().detach().numpy()
            else:
                flatten_attention = attention.reshape((Bs, 312, 49))
                should_be_peak = np.array(should_be_peak)
                sim = [] # BsX312
                for flat_att, peak, attribute_label in zip(flatten_attention, should_be_peak, attribute_labels):
                    peak = [p[p!=-1] for p in peak] 
                    peak = [list(range(49)) if len(p) == 0 else p for p in peak] #如果都沒有，取平均
                    detect_peak = torch.argmax(flat_att, -1)
                    #print(detect_peak[0])
                    #print(len(peak))
                    #print(len(detect_peak))
                    #print(peak[0])
                    idx = 0
                    for p, d_p, att in zip(peak, detect_peak, attribute_label):
                        #if att ==1:
                        counter +=1
                        if len(p) != 49 and att==1:
                        #if len(p) != 49:
                        #if att==1:     
                            d_p = d_p.item()
                            if (d_p in p) or (d_p+1 in p) or (d_p-1 in p) or (d_p+7 in p) or (d_p-7 in p):
                                correct +=1
                                att_correct[idx] +=1
                            total+=1
                            att_total[idx] +=1
                        idx +=1
        if should_be_peak==None or loc==False:
            return possitive_correct, negative_correct, possitive_total, negative_total
        return correct, total, counter, att_correct, att_total

def att_acc_best_th(x, a, should_be_peak, attribute_labels, muti_class=False, loc=True):
        '''
        x: BSX7X7X2048
        a: 312X2048
        attribute_labels: BSX312
        should_be_peak: list: len(BSX312)
        '''
        att_total = np.zeros(312)
        att_correct = np.zeros(312)
        possitive_correct = np.zeros(312)
        negative_correct = np.zeros(312)
        possitive_total = np.zeros(312)
        negative_total = np.zeros(312)
        counter = 0
        correct = 0
        total = 0
        N_attr = a.shape[0]
        Bs = x.shape[0]
        a = a.unsqueeze(-1)
        a = a.unsqueeze(-1)
        x_norm = torch.nn.functional.normalize(x,2,1)
        a_norm = torch.nn.functional.normalize(a,2,1)
        attention = F.conv2d(input=x_norm, weight=a_norm) #BSX312X7X7
        if should_be_peak==None:
            sim = F.max_pool2d(attention, kernel_size=7).view(Bs, -1) #BSX312   
            attribute_score = sim
            possitive_correct += torch.sum(torch.round(attribute_score*attribute_labels), 0).cpu().detach().numpy()            
            negative_correct += torch.sum(torch.round((1-attribute_score)*(1-attribute_labels)), 0).cpu().detach().numpy()
            possitive_total += torch.sum(attribute_labels, 0).cpu().detach().numpy()
            negative_total += torch.sum((1-attribute_labels), 0).cpu().detach().numpy()
        else:
            if not loc:
                flatten_attention = attention.reshape((Bs, 312, 49))
                should_be_peak = np.array(should_be_peak)
                sim = [] # BsX312
                for flat_att, peak in zip(flatten_attention, should_be_peak):
                    peak = [p[p!=-1] for p in peak] 
                    peak = [list(range(49)) if len(p) == 0 else p for p in peak] #如果都沒有，取平均
                    peak = [np.unique(p) for p in peak] #移除重複項
                    tmp = [torch.mean(flat_att[i][peak[i]]) for i in range(312)]
                    tmp = torch.stack(tmp)
                    sim.append(tmp)
                sim = torch.stack(sim)
                attribute_score = sim
                possitive_correct += torch.sum(torch.round(attribute_score*attribute_labels), 0).cpu().detach().numpy()            
                negative_correct += torch.sum(torch.round((1-attribute_score)*(1-attribute_labels)), 0).cpu().detach().numpy()
                possitive_total += torch.sum(attribute_labels, 0).cpu().detach().numpy()
                negative_total += torch.sum((1-attribute_labels), 0).cpu().detach().numpy()
            else:
                flatten_attention = attention.reshape((Bs, 312, 49))
                should_be_peak = np.array(should_be_peak)
                sim = [] # BsX312
                for flat_att, peak, attribute_label in zip(flatten_attention, should_be_peak, attribute_labels):
                    peak = [p[p!=-1] for p in peak] 
                    peak = [list(range(49)) if len(p) == 0 else p for p in peak] #如果都沒有，取平均
                    detect_peak = torch.argmax(flat_att, -1)
                    #print(detect_peak[0])
                    #print(len(peak))
                    #print(len(detect_peak))
                    #print(peak[0])
                    idx = 0
                    for p, d_p, att in zip(peak, detect_peak, attribute_label):
                        #if att ==1:
                        counter +=1
                        if len(p) != 49 and att==1:
                        #if len(p) != 49:
                        #if att==1:     
                            d_p = d_p.item()
                            if (d_p in p) or (d_p+1 in p) or (d_p-1 in p) or (d_p+7 in p) or (d_p-7 in p):
                                correct +=1
                                att_correct[idx] +=1
                            total+=1
                            att_total[idx] +=1
                        idx +=1
        if should_be_peak==None or loc==False:
            return possitive_correct, negative_correct, possitive_total, negative_total
        return correct, total, counter, att_correct, att_total


def turn_parts_to_picknum(parts):
        '''
        parts: 32X15X3
        return: 32X15X2
        '''
        part = parts.clone()
        part[:, :, :2] = part[:, :, :2]/32
        part[:, :, 0] = (part[:, :, 0]).to(int)*7 + (part[:, :, 1]).to(int)
        part[:, :, 1] = part[:, :, 2]
        part[:, :, 0] = torch.where(part[:, :, 1] != 0, part[:, :, 0], -1+0*part[:, :, 0])
        #part[no_att_idx, 0] = -1
        return part[:, :, 0]


def evaluate_location_acc(model, att, loaders, use_attribute_idx, not_use_attribute_idx):
    
    """
    (attribute part location acc)
    """
    device = att.device
    return_data = {'seen_loc_acc':[],
                   'unseen_loc_acc':[],
                   'loc_correct':[],
                   'loc_total':[],}
    name = ['Training set',
            'Testing set']


    for key, test_loader in enumerate(loaders):
        with torch.no_grad():
            #att = model.attributes()
            correct = 0
            total = 0
            counter = 0
            att_count = np.zeros(312)
            att_correct = np.zeros(312)
            att_total = np.zeros(312)
            
            tcav_bar = tqdm(test_loader)
            tcav_bar.set_description('Calculating location acc for %s' % name[key])
            
            for data in tcav_bar:
                model.eval()           
                x, y, parts, attribute_labels = data[0], data[1], data[2], data[3]
                #print(y[0])
                att_count += sum(attribute_labels).cpu().detach().numpy()
                picknum = turn_parts_to_picknum(parts)
                x, y, attribute_labels= x.to(device), y.to(device), attribute_labels.to(device)
                corresponding_idx = [] #len=312 
                for i in range(312):
                    corresponding_idx.append(np.where(test_loader.dataset.attribute_part_label[i]!=0)[0])
                pick_attr = [] #BSX312
                for p_n in picknum:
                    tmp = [p_n[corresponding_idx[i]].cpu().detach().numpy() for i in range(312)]
                    pick_attr.append(tmp)
                features = x
                correct_, total_, counter_, att_correct_, att_total_ = att_acc(features, att, pick_attr, attribute_labels)
                correct += correct_
                total += total_
                counter += counter_
                att_correct += att_correct_
                att_total += att_total_
            return_data['seen_loc_acc'] += [sum(att_correct[use_attribute_idx])/sum(att_total[use_attribute_idx])]
            return_data['unseen_loc_acc'] += [sum(att_correct[not_use_attribute_idx])/sum(att_total[not_use_attribute_idx])]
            return_data['loc_correct'] += [att_correct]
            return_data['loc_total'] += [att_total]
    return return_data


def evaluate_attribute_acc(model, att, loaders, use_attribute_idx, not_use_attribute_idx):
    return_data = {'seen_acc':[],
                   'seen_pos_acc':[],
                   'seen_neg_acc':[],
                   'unseen_acc':[],
                   'unseen_pos_acc':[],
                   'unseen_neg_acc':[],}
    device = att.device
    name = ['Training set',
            'Testing set']
    for key, test_loader in enumerate(loaders):
        with torch.no_grad():
            possitive_correct = np.zeros(312)
            negative_correct = np.zeros(312)
            possitive_total = np.zeros(312)
            negative_total = np.zeros(312)
            
            tcav_bar = tqdm(test_loader)
            tcav_bar.set_description('Calculating attribute level probability for %s' % name[key])
            for data in tcav_bar:  
                model.eval()           
                x, y, attribute_labels = data[0], data[1], data[3]
                x, y, attribute_labels= x.to(device), y.to(device), attribute_labels.to(device)
                features = x
                possitive_correct_, negative_correct_, possitive_total_, negative_total_ = att_acc(features, att, None, attribute_labels)
                possitive_correct += possitive_correct_
                negative_correct += negative_correct_
                possitive_total += possitive_total_
                negative_total += negative_total_
                
        return_data['seen_pos_acc'] += [sum(possitive_correct[use_attribute_idx])/sum(possitive_total[use_attribute_idx])]
        return_data['seen_neg_acc'] += [sum(negative_correct[use_attribute_idx])/sum(negative_total[use_attribute_idx])]
        return_data['seen_acc'] += [(return_data['seen_pos_acc'][-1]+return_data['seen_neg_acc'][-1])/2]
        return_data['unseen_pos_acc'] += [sum(possitive_correct[not_use_attribute_idx])/sum(possitive_total[not_use_attribute_idx])]
        return_data['unseen_neg_acc'] += [sum(negative_correct[not_use_attribute_idx])/sum(negative_total[not_use_attribute_idx])]
        return_data['unseen_acc'] += [(return_data['unseen_pos_acc'][-1]+return_data['unseen_neg_acc'][-1])/2]
    
    return return_data


      
