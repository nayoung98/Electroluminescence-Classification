import os

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import random

from util import *

# cell Dataloader
class ELcellDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_mode, phase):
        self.data_dir = data_dir
        self.data_mode = data_mode
        self.phase = phase
        
        self.info = pd.read_csv(os.path.join(self.data_dir, 'classification', self.phase, f're_data_df_{self.data_mode}.csv')) # re-labling
    
    def __len__(self):
        return len(self.info['label'])

    def __getitem__(self, index):        
        filename, label, label_re = self.info.iloc[index] # re-labeling
        label_str = 'fault' if label == 1 else 'non_fault'
        image_path = os.path.join(self.data_dir, self.data_mode, label_str, filename)

        img = Image.open(image_path)
        
        # edge crop
        img = edge_crop(img)
        
        # augmentation
        if self.phase == 'train':
            img = self.augment(img) 
        
        # cell crop
        cells = split_module_to_cells(img) # (3, w, h)
        
        img_tensors = []
        for cell in cells:
            img = F.resize(cell, (300, 600))
            img_tensor = self.img2tensor(img)
            img_tensor = torch.transpose(img_tensor, 1, 2)
            img_tensors.append(img_tensor)
        
        img_tensors = torch.stack(img_tensors, dim=0) # (cells, 3, 600, 300) 

        # re-label
        input_dict = {'filename' : filename, 
                      'img_tensor' : img_tensors, # cells
                      'label' : label_re,
                      'img_path': image_path
                      }

        return input_dict
    
    def augment(self, img):
        aug_list = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        img = aug_list(img)
        
        return img
    
    # PIL image -> torch.Tensor type
    def img2tensor(self, img):
        trans_list = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.58, 0.22)
        ])
        img = trans_list(img)
        
        return img
    
# cell Dataloader -> contrastive learning (cell의 개수가 같지 않은 경우를 맞춰줘야 함)
class ELcellContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_mode, phase):
        self.data_dir = data_dir
        self.data_mode = data_mode
        self.phase = phase
        
        self.info = pd.read_csv(os.path.join(self.data_dir, 'classification', self.phase, f're_data_df_{self.data_mode}.csv')) # re-labling
    
    def __len__(self):
        return len(self.info['label'])

    def __getitem__(self, index):        
        filename, label, label_re = self.info.iloc[index] # re-labeling
        label_str = 'fault' if label == 1 else 'non_fault'
        image_path = os.path.join(self.data_dir, self.data_mode, label_str, filename)

        img = Image.open(image_path)
        
        # edge crop
        img = edge_crop(img)
        
        # cell crop -> tensor
        cells = split_module_to_cells(img) # (3, w, h)
        
        img_tensors = []
        aug_tensors = []
        
        for cell in cells:
            img = F.resize(cell, (300, 600))
            img_tensor = self.img2tensor(img)
            img_tensor = torch.transpose(img_tensor, 1, 2)
            img_tensors.append(img_tensor)
            
            # augmentation
            if self.phase == 'train':
                img_aug = self.augment(img) 
                aug_tensor = self.img2tensor(img_aug)
                aug_tensor = torch.transpose(img_tensor, 1, 2)
                aug_tensors.append(aug_tensor)
                
        img_tensors = torch.stack(img_tensors, dim=0) # (cells, 3, 600, 300) 
        aug_tensors = torch.stack(aug_tensors, dim=0) # (cells, 3, 600, 300)
        
        # re-label
        input_dict = {'filename': filename, 
                      'img_tensor': img_tensors, # original_cells
                      'aug_tensor': aug_tensors, # augmented_cells
                      'label': label_re,
                      'img_path': image_path
                      }

        return input_dict
    
    def augment(self, img):
        aug_list = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        img = aug_list(img)
        
        return img
    
    # PIL image -> torch.Tensor type
    def img2tensor(self, img):
        trans_list = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.58, 0.22)
        ])
        img = trans_list(img)
        
        return img
        
class ELmoduleCustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_mode, phase, size_label):
        self.data_dir = data_dir
        self.data_mode = data_mode
        self.phase = phase
        self.info = pd.read_csv(os.path.join(self.data_dir, 'classification', self.phase, f're_data_df_{self.data_mode}.csv')) # re-labling
        # self.info = pd.read_csv(os.path.join(f'/home/pink/nayoung/el/main/code/origin_data.csv')) # total-data
        # self.info = pd.read_csv(os.path.join(self.data_dir, 'classification', self.phase, f'data_df_{self.data_mode}.csv'))
        self.size_label = size_label
            
    def __len__(self):
        return len(self.info['label'])

    def __getitem__(self, index):        
        filename, label, label_re = self.info.iloc[index] # re-labeling
        # filename, label = self.info.iloc[index]
        label_str = 'fault' if label == 1 else 'non_fault'
        image_path = os.path.join(self.data_dir, self.data_mode, label_str, filename)

        img = Image.open(image_path)
        
        # edge crop
        img = edge_crop(img)
        
        if self.size_label == 4:
            img = F.resize(img, (918, 1953)) # /4 
        elif self.size_label == 8:
            img = F.resize(img, (459, 977)) # / 8
        elif self.size_label == 16:
            img = F.resize(img, (230, 489)) # / 16
        
        if self.phase == 'train':
            img = self.augment(img) 
            
        # img -> tensor
        img_tensor = self.img2tensor(img) # (3, h, w)
        img_tensor = torch.transpose(img_tensor, 1, 2) # (3, w, h)

        # re-label
        input_dict = {'img_tensor' : img_tensor,
                      'img_name' : filename,
                      'label' : label_re,
                      'img_path': image_path}
        
        # # old-label
        # input_dict = {'img_tensor' : img_tensor,
        #               'img_name' : filename,
        #               'label' : label,
        #               'img_path': image_path}

        return input_dict

    def img2tensor(self, img):
        trans_list = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.58, 0.22)
        ])
        img = trans_list(img)
        return img
    
    def augment(self, img):
        aug_list = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        img = aug_list(img)
        
        return img
    
    # def collate_fn(self, batch):
    #     # logic to modify a batch of images
    #     imgs, clsses = list(zip(*batch))
        
    #     # transform a batch of images at once
    #     if self.phase == 'train':
    #         imgs = self.augment(imgs) 
        
    #     img_tensor = self.img2tensor(img) # (3, h, w)
    #     img_tensor = torch.transpose(img_tensor, 1, 2) # (3, w, h)
        
    #     # re-label
    #     input_dict = {'img_tensor' : img_tensor,
    #                   'img_name' : filename,
    #                   'label' : label_re,
    #                   'img_path': image_path}
        
    #     return input_dict
    
class ELmoduleEnsembleDataset(torch.utils.data.Dataset):
    # test_ensemble 용도
    def __init__(self, data_dir, data_mode, phase):
        self.data_dir = data_dir
        self.data_mode = data_mode
        self.phase = phase
        self.info = pd.read_csv(os.path.join(self.data_dir, 'classification', self.phase, f're_data_df_{self.data_mode}.csv')) # re-labling
        # self.info = pd.read_csv(os.path.join(self.data_dir, 'classification', self.phase, f'data_df_{self.data_mode}.csv'))
    
    def __len__(self):
        return len(self.info['label'])

    def __getitem__(self, index):        
        # filename, label = self.info.iloc[index]
        filename, label, re = self.info.iloc[index] # re-labeling
        label_str = 'fault' if label == 1 else 'non_fault'
        image_path = os.path.join(self.data_dir, self.data_mode, label_str, filename)

        img = Image.open(image_path)
        
        # edge crop
        # img = edge_crop(img)
        
        img_4 = F.resize(img, (918, 1953)) # /4
        img_8 = F.resize(img, (459, 977)) # /8
        img_16 = F.resize(img, (230, 489)) # /16
            
        # img -> tensor
        img_tensor_4 = self.img2tensor(img_4) # (3, h, w)
        img_tensor_8 = self.img2tensor(img_8) # (3, h, w)
        img_tensor_16 = self.img2tensor(img_16) # (3, h, w)
        
        img_tensor_4 = torch.transpose(img_tensor_4, 1, 2) # (3, w, h)
        img_tensor_8 = torch.transpose(img_tensor_8, 1, 2) # (3, w, h)
        img_tensor_16 = torch.transpose(img_tensor_16, 1, 2) # (3, w, h)

        # input_dict = {'img_tensor_4' : img_tensor_4,
        #               'img_tensor_8' : img_tensor_8,
        #               'img_tensor_16' : img_tensor_16,
        #               'label' : label,
        #               'image_path': image_path}
        
        input_dict = {'img_tensor_4' : img_tensor_4,
                      'img_tensor_8' : img_tensor_8,
                      'img_tensor_16' : img_tensor_16,
                      'label' : re,
                      'image_path': image_path}

        return input_dict

    def img2tensor(self, img):
        trans_list = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.58, 0.22)
        ])
        img = trans_list(img)
        
        return img
    