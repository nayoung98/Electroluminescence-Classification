import argparse
import ast

import os
import cv2
import numpy as np
import pandas as pd
import timm
from tqdm import tqdm
import itertools
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models import *
from dataset import *
from metric import *
from engine import *
# from visualizer import *

# parser
parser = argparse.ArgumentParser(description= 'Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")

parser.add_argument("--data_dir", default='/home/sliver/SDN', type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default='./checkpoints', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='./checkpoints', type=str, dest="log_dir")
parser.add_argument("--save_dir", default='./checkpoints', type=str, dest="save_dir")

parser.add_argument("--model_name", default='ResNet34', type=str, help="The model name")
parser.add_argument("--device", default=0, type=str, help="The GPU number")
parser.add_argument("--multi_gpu", default="off", type=str, help="[on | off]")
parser.add_argument("--data_mode", default="first", type=str, help="[first | second]")
parser.add_argument("--phase", default="test", type=str, help="[train | test]")
parser.add_argument("--load_path", default="./checkpoints", type=str, help="The path of the trained model")
parser.add_argument("--load_best_epoch", default=1, type=int, help="The epoch of the trained model")

args  = parser.parse_args() # parsing

batch_size = args.batch_size

model_name = args.model_name
load_path = args.load_path
load_best_epoch = args.load_best_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = ckpt_dir
save_dir = os.path.join(ckpt_dir, f'gradcam_ver2')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = args.device
multi_gpu = args.multi_gpu
data_mode = args.data_mode
phase = args.phase
load_path = args.load_path

model_config_path = '/home/sliver/SDN/Electroluminescence-Classification/model_config.yaml'
config = load_yaml_config(model_config_path)
model_config = config[model_name]
model = instantiate_from_config(model_config)
torch.cuda.empty_cache()

# gpu
device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
print(device)
device_ids = [5, 6]

# multi-gpu
if multi_gpu == 'on':
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print(device_ids)
        
print('================================================')
print(f"Model : {model_name}")
print("batch size: %d" % batch_size)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("mode: %s" % phase)
print(f'Load path : {load_path}')
print(f'Load epoch : {load_best_epoch}')
print('================================================')

if  phase == 'test':

    ## Dataloader ##
    dataset_test = ELmoduleCustomDataset(data_dir=data_dir, phase='test')
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    ## Models ##
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(load_path))
    
    sigmoid = nn.Sigmoid()

    target_layers = [model.layer4] 
    for param in model.layer4.parameters():
        param.requires_grad = True  
        
    cam = GradCAM(model, target_layers)
    
    for i, data in enumerate(tqdm(loader_test, 0)):
        img_name = data['filename']
        
        input = data['img_tensor']
        input = input.to(device) # (b, c, h, w)
        
        b, c, h, w = input.shape
        input_clone = input.view(b*c, h, w)
         
        # grad cam
        targets = [ClassifierOutputTarget(0)]
        el_cam = cam(input_tensor=input, targets=targets)
        el_cam = el_cam[0, :]

        min_val = np.min(el_cam)
        max_val = np.max(el_cam)
        if min_val != max_val:
            el_cam = (el_cam - min_val) / (max_val - min_val)
        else:
            el_cam = np.zeros_like(el_cam)  # or el_cam.fill(0.5) to set a default mid-value
                
        # Gradient 반전
        el_cam = 1.0 - el_cam
        
        input_clone = input_clone.permute(1,2,0) # (h, w, c)
        input_clone = input_clone.detach().cpu().numpy().astype(np.float32) 
        input_clone = (input_clone - np.min(input_clone)) / (np.max(input_clone) - np.min(input_clone))
        
        visualization = show_cam_on_image(input_clone, el_cam, use_rgb=True)
        cv2.imwrite(os.path.join(save_dir, f"grad_cam_{img_name}.jpg"), visualization)

print('The End!')