import os

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import numpy as np
import pandas as pd
from PIL import Image
import random
import ast
import json

from util import *

# Cell 단위 train/validation dataset
class ELcellcropDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cell_size, phase,
                 norm_stat=[0.58, 0.22]):
        self.data_dir = data_dir
        self.cell_size = cell_size # h: 600, w: 300
        self.phase = phase
        
        if self.phase == 'train':
            self.info = pd.read_csv(os.path.join(self.data_dir, 'classification', self.phase, f're_data_df_first.csv'))     
        else:
            self.info = pd.read_csv(os.path.join(self.data_dir, 'classification', self.phase, f're_data_df_first.csv'))
            
        tf_list = [
            transforms.ToTensor(),
            transforms.Normalize(norm_stat[0], norm_stat[1])
        ]
        
        if self.phase == 'train':
            tf_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        
        self.image_transform = transforms.Compose(tf_list)
        
    def __len__(self):
        return len(self.info['label'])

    def __getitem__(self, index):        
        if self.phase == 'train':
            filename, label, label_re, cell_labeling = self.info.iloc[index]
        else:
            filename, label, label_re = self.info.iloc[index]
        label_str = 'fault' if label == 1 else 'non_fault'
        filename = filename.split('.')[0]
        
        image_path = os.path.join(self.data_dir, 'Image', f're4_{label_str}', filename+'.png')
        
        img = Image.open(image_path)
               
        img = self.image_transform(img) # C x H x W
        
        # cell crop
        cells = module2cell(img.unsqueeze(0), self.cell_size).squeeze(0)    # CN x C x CH X CW
        
        input_dict = {'filename' : filename, 
                      'img_tensor' : cells,
                      'label' : label_re,
                      'img_path': image_path
                    }

        return input_dict

# Cell 단위 evaluation dataset
class ELcellcropEvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cell_size, phase,
                 norm_stat=[0.58, 0.22]):
        self.data_dir = data_dir
        self.cell_size = cell_size # h: 600, w: 300
        self.phase = phase

        self.info = pd.read_csv(os.path.join('/home/sliver/SDN/classification/test/final_test_copy2.csv')) 

        tf_list = [
            transforms.ToTensor(),
            transforms.Normalize(norm_stat[0], norm_stat[1])
        ]
        
        self.image_transform = transforms.Compose(tf_list)

    def __len__(self):
        return len(self.info['label'])

    def __getitem__(self, index):        
        filename, label, label_re, cell_labeling = self.info.iloc[index]
        label_str = 'fault' if label == 1 else 'non_fault'
        filename = filename.split('.')[0]
            
        image_path = os.path.join(self.data_dir, 'Image', f're4_{label_str}', filename+'.png')
        img = Image.open(image_path)
       
        img = self.image_transform(img) # C x H x W
        
        # cell crop
        cells = module2cell(img.unsqueeze(0), self.cell_size).squeeze(0) # CN x C x CH X CW

        cell_labeling = ast.literal_eval(cell_labeling)
        cell_labeling = [0 if value == 2 else value for value in cell_labeling] # 불량 의심 -> 정상
        cell_labeling = json.dumps(cell_labeling)
        # print(cells.shape)
    
        # cell-label
        input_dict = {'filename' : filename, 
                      'img_tensor' : cells,
                      'label_cell' : cell_labeling,
                      'label_module': label_re,
                      'img_path': image_path
                      }

        return input_dict

# Module 단위 train/validation/evaluation dataset
class ELmoduleCustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, norm_stat=[0.58, 0.22]):
        self.data_dir = data_dir
        self.phase = phase      
       
        if self.phase == 'train':
            self.info = pd.read_csv(os.path.join('/home/sliver/SDN/classification/train/fine_590.csv'))
            # self.info = pd.read_csv(os.path.join(self.data_dir, 'classification', self.phase, f're_data_df_first.csv'))   
        elif self.phase == 'valid':
            self.info = pd.read_csv(os.path.join(self.data_dir, 'classification', self.phase, f're_data_df_first.csv'))    
        elif self.phase == 'test':
            self.info = pd.read_csv(os.path.join('/home/sliver/SDN/classification/test/final_test_copy2.csv')) # final_test 라벨링 다시 함, 589
        
        tf_list = [
            transforms.ToTensor(),
            transforms.Normalize(norm_stat[0], norm_stat[1])
        ]
        
        if self.phase == 'train':
            tf_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        
        self.image_transform = transforms.Compose(tf_list)
        
    def __len__(self):
        return len(self.info['label'])

    def __getitem__(self, index):     
        if self.phase == 'train':
            filename, label, label_re, cell_labeling = self.info.iloc[index]
        elif self.phase == 'valid':
            filename, label, label_re = self.info.iloc[index]
        elif self.phase == 'test':
            filename, label, label_re, cell_labeling = self.info.iloc[index]
            
        label_str = 'fault' if label == 1 else 'non_fault'
        filename = filename.split('.')[0]
                
        image_path = os.path.join(self.data_dir, 'Image', f'new_{label_str}', filename+'.png') # crop
        
        if not os.path.exists(image_path):
            image_path = os.path.join(self.data_dir, 'new_image_OK', 'new_image_0619', 'OK3', filename+'.jpg')
            if not os.path.exists(image_path):
                image_path = os.path.join(self.data_dir, 'new_image_OK', 'new_image_0619', 'OK4', filename+'.jpg')
            
            img = Image.open(image_path)
            img = edge_crop(img) 
        else:
            img = Image.open(image_path)
        
        img = img.resize((977, 459)) # /8, W x H
        
        img_tensor = self.image_transform(img) # C x H x W

        # re-label
        input_dict = {'img_tensor' : img_tensor,
                      'filename' : filename,
                      'label' : label_re,
                      'img_path': image_path}

        return input_dict
