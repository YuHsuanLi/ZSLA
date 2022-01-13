# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import torchvision.models as models
import torch
import numpy as np
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from skimage import transform as skTr
import scipy.io
from itertools import chain
identity = lambda x:x
import random
SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


class Class_Dataset():
    def __init__(self, split, transform=transforms.ToTensor(), size=224, renumber=False, attribute_batchsize=1, attribute_shuffle=False, dataset_root_path='/home/user/tzuyin/ra_project/dataset', n_episode=600):

        self.img_label = pd.read_table(dataset_root_path+'/CUB_200_2011/image_class_labels.txt',sep=' ',header=None).values[...,1]-1        
        self.corresponding_path = pd.read_table(dataset_root_path+'/CUB_200_2011/images.txt',sep=' ',header=None).values[...,1:]
        
        self.class_name = pd.read_table(dataset_root_path+'/CUB_200_2011/classes.txt',sep=' ',header=None).values[...,1:][...,0]
        
        self.image_path = np.stack([dataset_root_path+'/CUB_200_2011/images/'+item for item in self.corresponding_path],axis=0)[...,0]
        self.attribute = pd.read_table(dataset_root_path+'/CUB_200_2011/attributes/class_attribute_labels_continuous.txt',sep=' ',header=None).values/100.
        self.split_mat = scipy.io.loadmat(dataset_root_path+'/att_splits.mat')  
        self.attribute_name = pd.read_table(dataset_root_path+'/CUB_200_2011/attributes.txt',sep=' ',header=None).values[...,1]
        self.object_xywh = pd.read_table(dataset_root_path+'/CUB_200_2011/bounding_boxes.txt',sep=' ',header=None).values[...,1:]
        self.image_attribute_labels = pd.read_csv(dataset_root_path+'/CUB_200_2011/attributes/processed_image_attribute_labels.csv',header=None).values

        corresponding_path = pd.read_table(dataset_root_path+'/CUB_200_2011/images.txt',sep=' ',header=None).values[...,1:]  
        self.mask_path = np.expand_dims(np.stack([dataset_root_path+'/CUB_200_2011/segmentations/'+item[0][:-3]+'png' for item in corresponding_path],axis=0),axis=-1)[...,0]
        self.attribute_batchsize = attribute_batchsize
        
        self.size = size
        if(split=='train'):
            self.img_id_list = self.split_mat['train_loc']-1
        elif(split=='train_val'):
            self.img_id_list = self.split_mat['trainval_loc']-1
        elif(split=='val'):
            self.img_id_list = self.split_mat['val_loc']-1
        elif(split=='test_seen'):
            self.img_id_list = self.split_mat['test_seen_loc']-1
        elif(split=='test_unseen'):
            self.img_id_list = self.split_mat['test_unseen_loc']-1
        elif(split=='all'):
            self.img_id_list = np.array([[i] for i in range(self.img_label.shape[0])])
        else:
            raise "The selection must in the set of {'train','train_val','val','test_seen','test_unseen'}"
        
        self.img_id_list=self.img_id_list[:,0]
        
        self.n_episode=n_episode
        self.attr_list = [i for i in range(312)]

        self.cl_list = np.unique(self.img_label[self.img_id_list]).tolist()
        
        attribute_name = self.attribute_name = pd.read_table(dataset_root_path+'/CUB_200_2011/attributes.txt',sep=' ',header=None).values[...,1]
        self.renumbered_label = np.sort(np.unique(self.img_label[self.img_id_list]))
        self.class_idx = np.unique(self.img_label[self.img_id_list])
        self.label_transform_list = (np.ones([200])*-1).astype(int)
        for key, value in enumerate(self.renumbered_label):
            self.label_transform_list[value] = key
        self.renumber = renumber
        
        
        part_name = {'back':[0],
                    'beak':[1],
                    'belly':[2],
                    'breast':[3],
                    'crown':[4],
                    'forehead':[5],
                    'eye':[6,10],
                    'head':[4,5,6,10],
                    'leg':[7,11],
                    'wing':[8,12],
                    'nape':[9],
                    'tail':[13],
                    'throat':[14],
                    'upperparts':[0,3,4,5,6,10,14],
                    'underparts':[2,8,12,13],
                    'primary':[0,2,3,4,5,6,8,9,10,12,13,14]}
          
        attribute_part_label = np.zeros([312,15])
    
        for key_attr, attr_n in enumerate(attribute_name):
            for key_part, part_n in enumerate(part_name):

                if('head' in attr_n and 'forehead' not in attr_n):
                    for counter in part_name['head']:
                        attribute_part_label[key_attr, counter]=1
                
                elif('forehead' in attr_n):
                    for counter in part_name['forehead']:
                        attribute_part_label[key_attr, counter]=1
                        
                elif(part_n in attr_n):
                    for counter in part_name[part_n]:
                        attribute_part_label[key_attr, counter]=1

        self.attribute_part_label = attribute_part_label       
        
        
                
        def read_parts(filename):
            id_to_parts = dict()
            with open(filename, 'r') as fin:
                for line in fin.readlines():
                    line_split = line.strip().split(' ')
                    img_id, part_id, x, y, visible = int(line_split[0]), int(line_split[1]), float(line_split[2]), float(line_split[3]), int(line_split[4])
                    if part_id == 1:
                        id_to_parts[img_id] = [[x, y, visible], ]
                    else:
                        id_to_parts[img_id].append([x, y, visible])
            return id_to_parts
        
        id_to_parts = read_parts(dataset_root_path+'/CUB_200_2011/parts/part_locs.txt')
        self.id_to_parts = np.array([id_to_parts[N] for N in id_to_parts])

        self.class_sub_meta = {}
        for clsidx in self.cl_list:
            self.class_sub_meta[clsidx] = []

        for x,l,A,z,M,B in zip(self.image_path[self.img_id_list],
                               self.img_label[self.img_id_list],
                               self.image_attribute_labels[self.img_id_list], 
                               self.id_to_parts[self.img_id_list], 
                               self.mask_path[self.img_id_list], 
                               self.object_xywh[self.img_id_list]):

            A_ = np.where(A==1)[0]
            for y in A_:
                idx = np.where(attribute_part_label[y]==1)[0][0]
                apl = np.sum(attribute_part_label[idx],0)*np.array(z)[idx][2]
      
            self.class_sub_meta[l].append({'path':x, 'part': z, 'mask': apl, 'class':l, 'mask_path':M, 'box':B, 'att_label': A})

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = 1,
                                  shuffle = False,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)  
        # 分class
        '''
        for clsidx in self.cl_list:
            sub_dataset = SubAttributeDataset(self.class_sub_meta[clsidx], clsidx, size, is_train = False)
            self.sub_dataloader += [torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)]
        '''
        # 不分class
        flatten_attr_sub_meta = []        
        for clsidx in self.cl_list:
            #print(i, len(self.attr_sub_meta[i]))
            flatten_attr_sub_meta += self.class_sub_meta[clsidx]
        #print(len(flatten_attr_sub_meta))
        sub_dataset = SubAttributeDataset(flatten_attr_sub_meta, clsidx, size, is_train = False)
        self.sub_dataset = sub_dataset
        self.sub_meta = flatten_attr_sub_meta
        self.image_size = size
        self.sub_dataloader += [torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)]
            
            
        self.get_cls_n = 0
        
    def get_class(self, n):
        #n is corresponding to the current set
        self.get_cls_n
        
    def renumber_index(self, label):
        if self.renumber==True:
            return self.label_transform_list[label]
        else:
            return label    
        
    def __getitem__(self,idx):
        image_file = os.path.join(self.sub_meta[idx]['path'])
        mask_file = self.sub_meta[idx]['mask_path']
               
        data_numpy = plt.imread(image_file)/255.
        data_mask = plt.imread(mask_file)
        if(len(data_mask.shape)!=2):
            data_mask = data_mask[...,0]
        
        if(len(data_numpy.shape)!=3):
            data_numpy = np.tile(data_numpy[...,None],(1,1,3))

        origin_len = data_numpy.shape[:-1]
        data_numpy = skTr.resize(data_numpy, (self.image_size, self.image_size))
        
        data_mask = data_mask[None,None]
        
        try:
            assert np.sum(data_mask) != 0
            data_mask = torch.round(torch.nn.functional.interpolate(torch.from_numpy(data_mask), (self.image_size, self.image_size))).numpy()[0,0]
        except:
            raise print(mask_file)

        part_mask = self.sub_meta[idx]['mask']
        
        joints_vis = self.sub_meta[idx]['part']
        
        joints_vis_ = np.array(joints_vis)        
        joints_vis = np.copy(joints_vis_)
        
        joints_vis[...,0] = joints_vis_[...,1]*(self.image_size/origin_len[0])
        joints_vis[...,1] = joints_vis_[...,0]*(self.image_size/origin_len[1])
         
                    
        obj_yxwh = np.array(self.sub_meta[idx]['box'])
        obj_xyhw = np.zeros_like(obj_yxwh)
        
        obj_xyhw[...,0] = obj_yxwh[...,1]*(self.image_size/origin_len[0])
        obj_xyhw[...,1] = obj_yxwh[...,0]*(self.image_size/origin_len[1])
        obj_xyhw[...,2] = obj_yxwh[...,3]*(self.image_size/origin_len[0])
        obj_xyhw[...,3] = obj_yxwh[...,2]*(self.image_size/origin_len[1])
        
        class_label = self.sub_meta[idx]['class']
        img_attribute_label = self.sub_meta[idx]['att_label']
        class_label = np.array(class_label)
        class_label = self.renumber_index(class_label).astype(np.int64)
        #_1, _2, _3, _4, _5, _6, _7 = next(iter(self.sub_dataloader[self.get_cls_n]))
        #return _1, _2, _3, _4, _5, _6, _7, self.class_name[self.cl_list[self.get_cls_n]], self.cl_list[self.get_cls_n]
        return data_numpy.astype(np.float32).transpose([2,0,1]), joints_vis.astype(np.float32), part_mask.astype(np.float32), data_mask[None].astype(np.float32), obj_xyhw.astype(np.float32), class_label, img_attribute_label.astype(np.float32), self.class_name[self.cl_list[self.get_cls_n]], self.cl_list[self.get_cls_n]
        
    def __len__(self):
        return len(self.img_id_list)
    
    def get_class_attribute(self, return_tensor=False):
            if(self.renumber == True):
                #print(self.renumbered_label[10:15])
                if(return_tensor):
                    return torch.from_numpy(self.attribute[self.renumbered_label].astype(np.float32))
                else:
                    return self.attribute[self.renumbered_label].astype(np.float32)
            else:
                if(return_tensor):
                    return torch.from_numpy(self.attribute.astype(np.float32))
                else:
                    return self.attribute.astype(np.float32)
'''
class Attribute_Dataset():
    def __init__(self, split, transform=transforms.ToTensor(), size=224, renumber=False, attribute_batchsize=1, attribute_shuffle=True, dataset_root_path='/eva_data_1/yu_hsuan_li/logic_kernel/dataset', n_episode=600):

        self.img_label = pd.read_table(dataset_root_path+'/CUB_200_2011/image_class_labels.txt',sep=' ',header=None).values[...,1]-1        
        self.corresponding_path = pd.read_table(dataset_root_path+'/CUB_200_2011/images.txt',sep=' ',header=None).values[...,1:]
        
        self.image_path = np.stack([dataset_root_path+'/CUB_200_2011/images/'+item for item in self.corresponding_path],axis=0)[...,0]
        self.attribute = pd.read_table(dataset_root_path+'/CUB_200_2011/attributes/class_attribute_labels_continuous.txt',sep=' ',header=None).values/100.
        self.split_mat = scipy.io.loadmat(dataset_root_path+'/att_splits.mat')  
        self.attribute_name = pd.read_table(dataset_root_path+'/CUB_200_2011/attributes.txt',sep=' ',header=None).values[...,1]
        self.object_xywh = pd.read_table(dataset_root_path+'/CUB_200_2011/bounding_boxes.txt',sep=' ',header=None).values[...,1:]
        #self.split_mat = scipy.io.loadmat(dataset_root_path+'/CUB_200_2011/att_splits.mat')  
        self.image_attribute_labels = pd.read_csv(dataset_root_path+'/CUB_200_2011/attributes/processed_image_attribute_labels.csv',header=None).values
        
        corresponding_path = pd.read_table('/eva_data_1/yu_hsuan_li/logic_kernel/dataset/CUB/CUB_200_2011/images.txt',sep=' ',header=None).values[...,1:]        
        self.mask_path = np.expand_dims(np.stack(['/eva_data_1/yu_hsuan_li/logic_kernel/dataset/CUB/CUB_200_2011/segmentations/'+item[0][:-3]+'png' for item in corresponding_path],axis=0),axis=-1)[...,0]
        self.attribute_batchsize = attribute_batchsize
        
        self.size = size
        if(split=='train'):
            self.img_id_list = self.split_mat['train_loc']-1
        elif(split=='train_val'):
            self.img_id_list = self.split_mat['trainval_loc']-1
        elif(split=='val'):
            self.img_id_list = self.split_mat['val_loc']-1
        elif(split=='test_seen'):
            self.img_id_list = self.split_mat['test_seen_loc']-1
        elif(split=='test_unseen'):
            self.img_id_list = self.split_mat['test_unseen_loc']-1
        elif(split=='all'):
            self.img_id_list = np.array([[i] for i in range(self.img_label.shape[0])])
        else:
            raise "The selection must in the set of {'train','train_val','val','test_seen','test_unseen'}"
        
        self.img_id_list=self.img_id_list[:,0]
        
        self.n_episode=n_episode
        self.attr_list = [i for i in range(312)]

        self.cl_list = np.unique(self.img_label[self.img_id_list]).tolist()
        
        attribute_name = self.attribute_name = pd.read_table('/eva_data_1/yu_hsuan_li/logic_kernel/dataset/CUB/CUB_200_2011/attributes.txt',sep=' ',header=None).values[...,1]
        
        part_name = {'back':[0],
                    'beak':[1],
                    'belly':[2],
                    'breast':[3],
                    'crown':[4],
                    'forehead':[5],
                    'eye':[6,10],
                    'head':[4,5,6,10],
                    'leg':[7,11],
                    'wing':[8,12],
                    'nape':[9],
                    'tail':[13],
                    'throat':[14],
                    'upperparts':[0,3,4,5,6,10,14],
                    'underparts':[2,8,12,13],
                    'primary':[0,2,3,4,5,6,8,9,10,12,13,14]}
          
        attribute_part_label = np.zeros([312,15])
    
        for key_attr, attr_n in enumerate(attribute_name):
            for key_part, part_n in enumerate(part_name):

                if('head' in attr_n and 'forehead' not in attr_n):
                    for counter in part_name['head']:
                        attribute_part_label[key_attr, counter]=1
                
                elif('forehead' in attr_n):
                    for counter in part_name['forehead']:
                        attribute_part_label[key_attr, counter]=1
                        
                elif(part_n in attr_n):
                    for counter in part_name[part_n]:
                        attribute_part_label[key_attr, counter]=1
            
        def read_parts(filename):
            id_to_parts = dict()
            with open(filename, 'r') as fin:
                for line in fin.readlines():
                    line_split = line.strip().split(' ')
                    img_id, part_id, x, y, visible = int(line_split[0]), int(line_split[1]), float(line_split[2]), float(line_split[3]), int(line_split[4])
                    if part_id == 1:
                        id_to_parts[img_id] = [[x, y, visible], ]
                    else:
                        id_to_parts[img_id].append([x, y, visible])
            return id_to_parts
        
        id_to_parts = read_parts('/eva_data_1/yu_hsuan_li/logic_kernel/dataset/CUB/CUB_200_2011/parts/part_locs.txt')
        self.id_to_parts = np.array([id_to_parts[N] for N in id_to_parts])



        self.attr_sub_meta = {}
        for attr in self.attr_list:
            self.attr_sub_meta[attr] = []

        for x,l,A,z,M,B in zip(self.image_path[self.img_id_list],
                               self.img_label[self.img_id_list],
                               self.image_attribute_labels[self.img_id_list], 
                               self.id_to_parts[self.img_id_list], 
                               self.mask_path[self.img_id_list], 
                               self.object_xywh[self.img_id_list]):
            
            A = np.where(A==1)[0]
            for y in A:
                idx = np.where(attribute_part_label[y]==1)[0][0]
                if(np.array(z)[idx][2]!=0):
                    self.attr_sub_meta[y].append({'path':x, 'part': z, 'mask': attribute_part_label[y], 'class':l, 'mask_path':M, 'box':B})
              
        self.sub_attr_dataloader = [] 
        sub_data_loader_params = dict(batch_size = 1,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)  
        # 分attribute
        
        for attr in self.attr_list:
            sub_dataset = SubAttributeDataset(self.attr_sub_meta[attr], attr, size, is_train = False)
            self.sub_attr_dataloader += [torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)]
        
        self.get_cls_n = 0
        
    def get_attribute(self, n):
        #n is corresponding to the current set
        self.get_cls_n = n

    def __getitem__(self,i):
        _1, _2, _3, _4, _5, _6  = next(iter(self.sub_attr_dataloader[self.get_cls_n]))
        return _1, _2, _3, _4, _5, _6, self.attribute_name[self.attr_list[self.get_cls_n]], self.attr_list[self.get_cls_n]
        
    def __len__(self):
        #return self.n_episode
        return len(self.img_id_list)
'''
class SubAttributeDataset(Dataset):

    def __init__(self, sub_meta, attr, image_size, transform=transforms.ToTensor(), target_transform=identity, is_train=True):
        self.num_joints = 15

        self.is_train = is_train
        self.sub_meta = sub_meta
        self.attr = attr 
        self.transform = transform
        self.target_transform = target_transform

        self.flip = is_train

        self.image_size = image_size

        self.ban_class_idx = None


    def ban_class(self,cls_num):
        self.ban_class_idx=cls_num
    
    def unban_class(self):
        self.ban_class_idx=None

    def __len__(self,):
        return len(self.sub_meta)

    def __getitem__(self, idx):
        #print(idx)
        
        if(isinstance(self.ban_class, type(None))):
            while(self.sub_meta[idx]['class'] in self.ban_class_idx):
                idx = (idx + 1)%len(self.sub_meta)
        
        #print(idx)
        image_file = os.path.join(self.sub_meta[idx]['path'])
        mask_file = self.sub_meta[idx]['mask_path']

               
        data_numpy = plt.imread(image_file)/255.
        data_mask = plt.imread(mask_file)
        if(len(data_mask.shape)!=2):
            data_mask = data_mask[...,0]
        
        if(len(data_numpy.shape)!=3):
            data_numpy = np.tile(data_numpy[...,None],(1,1,3))

        origin_len = data_numpy.shape[:-1]
        data_numpy = skTr.resize(data_numpy, (self.image_size, self.image_size))
        
        data_mask = data_mask[None,None]
        
        try:
            assert np.sum(data_mask) != 0
            data_mask = torch.round(torch.nn.functional.interpolate(torch.from_numpy(data_mask), (self.image_size, self.image_size))).numpy()[0,0]
        except:
            raise print(mask_file)

        part_mask = self.sub_meta[idx]['mask']
        
        joints_vis = self.sub_meta[idx]['part']
        
        joints_vis_ = np.array(joints_vis)        
        joints_vis = np.copy(joints_vis_)
        
        joints_vis[...,0] = joints_vis_[...,1]*(self.image_size/origin_len[0])
        joints_vis[...,1] = joints_vis_[...,0]*(self.image_size/origin_len[1])
        
        obj_yxwh = np.array(self.sub_meta[idx]['box'])
        obj_xyhw = np.zeros_like(obj_yxwh)
        
        obj_xyhw[...,0] = obj_yxwh[...,1]*(self.image_size/origin_len[0])
        obj_xyhw[...,1] = obj_yxwh[...,0]*(self.image_size/origin_len[1])
        obj_xyhw[...,2] = obj_yxwh[...,3]*(self.image_size/origin_len[0])
        obj_xyhw[...,3] = obj_yxwh[...,2]*(self.image_size/origin_len[1])
        
        class_label = self.sub_meta[idx]['class']
        img_attribute_label = self.sub_meta[idx]['att_label']
        class_label = np.array(class_label)
        '''
        #return: 
        #1.image [3,image_size,image_size]
        #2.part coordinate [15 part X (x,y,show_or_not[1:show 0:not show])]
        #3.attribute corrisponding part [15 part [1:selected 0:not selected]], ex: for the attribute "has_blue_eye", the part representing "eyes" will be 1 while others are 0.
        #4.data mask [1,image_size,image_size] sementic segmentation mask
        #5.object bounding box [4] corresponding to x,y,h,w
        '''

        return data_numpy.astype(np.float32).transpose([2,0,1]), joints_vis.astype(np.float32), part_mask.astype(np.float32), data_mask[None].astype(np.float32), obj_xyhw.astype(np.float32), class_label, img_attribute_label.astype(np.float32)


def get_region_by_part(name, joints, sementic_mask, obj_xyhw, size_care=1/6):
    '''
    name: a list of string of parts you would like to have; the name should be either in "part name" or "primary" (all bird).
    joints: an array of 15 part X [x,y,show_or_not]
    sementic_mask: sementic segmentation mask
    obj_xyhw: the bounding box of the bird
    size_care: we would like to crop size_careX(hight, width of the whole) rigion to represent the part.(defaule to be 1/6)
    '''
    
    
    part_name = {'back':[0],
                    'beak':[1],
                    'belly':[2],
                    'breast':[3],
                    'crown':[4],
                    'forehead':[5],
                    'eye':[6,10],
                    'head':[4,5,6,10],
                    'leg':[7,11],
                    'wing':[8,12],
                    'nape':[9],
                    'tail':[13],
                    'throat':[14],
                    'upperparts':[0,3,4,5,6,10,14],
                    'underparts':[2,8,12,13]}
    
    if name == 'primary':
        return sementic_mask
    else:
        mask = np.zeros([1,joints.shape[1]])
        #mask = torch.Tensor(mask).to(joints.device)
        for attr_n in name:
            print(attr_n)
            print(part_name)
            print(part_name[attr_n])
            print(mask.shape)
            mask[...,part_name[attr_n]] = 1
        
        applied_mask = mask*joints[...,2]
        batch_num = sementic_mask.shape[0]
        joints_num = joints.shape[1]
        ret = np.zeros_like(sementic_mask)
        img_len = sementic_mask.shape[-2]
        size_care = np.round((obj_xyhw[:,2:]*size_care)/2).astype(int)
        joints = joints.astype(int)
    
        for i in range(batch_num):
            for j in range(joints_num):
                if joints[i, j, 2] == 1 and joints[i, j, 0] >= 0 and joints[i, j, 1] >= 0 and joints[i, j, 0] < img_len and joints[i, j, 1] < img_len:
                    if applied_mask[i, j] == 1:
                        x_ = joints[i, j, 0]
                        y_ = joints[i, j, 1]
                        ret[i,
                           np.maximum(x_-size_care[i,0],0):np.minimum(x_+size_care[i,0],img_len-1),
                           np.maximum(y_-size_care[i,1],0):np.minimum(y_+size_care[i,1],img_len-1),
                           :] = sementic_mask[i,
                                             np.maximum(x_-size_care[i,0],0):np.minimum(x_+size_care[i,0],img_len-1),
                                             np.maximum(y_-size_care[i,1],0):np.minimum(y_+size_care[i,1],img_len-1),
                                             :]    
    
        return ret
'''
if __name__ == '__main__':
    ######test function######
    test = Class_Dataset('train')
    test.get_class(0)
    
    ret = test[0]
    image = ret[0].numpy().transpose([0,2,3,1])
    joints = ret[1].numpy()
    attribute_corr_part = ret[2].numpy()
    sementic_mask = ret[3].numpy().transpose([0,2,3,1])
    obj_xyhw = ret[4].numpy()
    name = ret[5]
    
    get_back_mask = get_region_by_part(['head','leg'], joints, sementic_mask, obj_xyhw, size_care=1/6)
    
    for I in get_back_mask*image:
        plt.figure()
        plt.title(name)
        plt.imshow(I)
        
        
    ######test function######
    test = Attribute_Dataset('train')
    test.get_attribute(0)
    
    ret = test[0]
    image = ret[0].numpy().transpose([0,2,3,1])
    joints = ret[1].numpy()
    attribute_corr_part = ret[2].numpy()
    sementic_mask = ret[3].numpy().transpose([0,2,3,1])
    obj_xyhw = ret[4].numpy()
    name = ret[5]
    
    get_back_mask = get_region_by_part(['head','leg'], joints, sementic_mask, obj_xyhw, size_care=1/6)
    
    for I in get_back_mask*image:
        plt.figure()
        plt.title(name)
        plt.imshow(I)
        
'''    


