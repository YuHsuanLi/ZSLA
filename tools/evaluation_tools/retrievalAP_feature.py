import torch
import numpy as np
from sklearn import metrics
from tqdm import tqdm

def roc_test(y_true, y_pred):
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    
    best_th = thresholds[np.argmax(tpr-fpr)]
    y_pred_under_best_th = np.where(y_pred>=best_th, 1, 0)

    CORRECT_or_not = np.where(y_pred_under_best_th==y_true,1,0)

    ACC = np.mean(CORRECT_or_not)
    TPR = np.mean(CORRECT_or_not[np.where(y_true==1)])
    TNR = np.mean(CORRECT_or_not[np.where(y_true==0)])

    return roc_auc, best_th, TPR, TNR, ACC


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def compute_ap(y_true, y_pred, at_k=50):
    '''rank_list: image idx ranked by the probability
       pos_list: label list
       at_k: only consider the performance of top K highest score images'''

    idx = np.argsort(-y_pred)
    
    r = y_true[idx]
    ap_at_k = average_precision(r[:at_k])
    
    return ap_at_k


def eval_APatK_and_RocAuc(model, loaders, use_attribute_idx, not_use_attribute_idx, at_k, att=None):
    """
    for attribute (mAP)
    """
    
    return_data = {}
    
    return_data['ap_list'] = []
    return_data['seen_attribute_mAP'] = []
    return_data['unseen_attribute_mAP'] = []
        
    return_data['roc_auc_list'] = []
    return_data['seen_attribute_roc_auc'] = []
    return_data['unseen_attribute_roc_auc'] = []
    
    name = ['Train',
            'Test']
    
    for key, test_loader in enumerate(loaders):
        with torch.no_grad():
            pred_prob = []
            att_label = []
            tcav_bar = tqdm(test_loader)
            tcav_bar.set_description('Calculating attribute level probability for %s' % name[key])
            for data in tcav_bar:
                model.eval()           
                x, y, attribute_labels = data[0], data[1], data[3]
                x, y, attribute_labels= x.to(model.device), y.to(model.device), attribute_labels.to(model.device)

                features = x
                
                try: # case for gate
                    prob, _ = model.att_cls_fast(features, att, np.concatenate([use_attribute_idx, not_use_attribute_idx],0))
                except: # case for baseline
                    prob, _ = model(features, np.concatenate([use_attribute_idx, not_use_attribute_idx],0))
                    
                pred_prob += [prob.detach().cpu().numpy()]
                att_label += [attribute_labels.detach().cpu().numpy()]

        pred_prob = np.concatenate(pred_prob,0)
        att_label = np.concatenate(att_label,0)
        
        ap_list = []
        roc_auc_list = []
        
        tcav_bar = tqdm(range(pred_prob.shape[1]))
        tcav_bar.set_description('Calculating AP score')        
        
        for attr_counter in tcav_bar:
            
            y_pred, y_true = pred_prob[:,attr_counter],att_label[:,attr_counter]
            
            ap_list += [compute_ap(y_true, y_pred, at_k)]
            
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
            roc_auc_list += [metrics.auc(fpr, tpr)]
        
        ap_list = np.array(ap_list)
        roc_auc_list = np.array(roc_auc_list)
        
        return_data['ap_list'] += [ap_list]
        return_data['seen_attribute_mAP'] += [np.mean(ap_list[:len(use_attribute_idx)])]
        return_data['unseen_attribute_mAP'] += [np.mean(ap_list[len(use_attribute_idx):])]
        
        return_data['roc_auc_list'] += [roc_auc_list]
        return_data['seen_attribute_roc_auc'] += [np.mean(roc_auc_list[:len(use_attribute_idx)])]
        return_data['unseen_attribute_roc_auc'] += [np.mean(roc_auc_list[len(use_attribute_idx):])]
        
    return return_data




