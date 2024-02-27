import argparse

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import *
from dataset import *
from metric import *
from engine import *

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# parser
parser = argparse.ArgumentParser(description= 'Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")

parser.add_argument("--data_dir", default='/home/pink/nayoung/el/datasets/el', type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default='./checkpoints', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='./checkpoints', type=str, dest="log_dir")

parser.add_argument("--device", default=0, type=str, help="The GPU number")
parser.add_argument("--multi_gpu", default="off", type=str, help="[on | off]")
parser.add_argument("--data_mode", default="first", type=str, help="[first | second]")
parser.add_argument("--phase", default="train", type=str, help="[train | test]")
parser.add_argument("--size_label", default=4, type=int, help="[4 | 8 | 16]")
parser.add_argument("--load_path", default="./checkpoints", type=str, help="The path of the trained model")
parser.add_argument("--cm", default=1, type=int, help="The number of confusion matrix")


args  = parser.parse_args() # parsing

batch_size = args.batch_size

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir

# Grad-CAM directory
grad_dir = os.path.join(ckpt_dir, 'gradcam')
grad_dir_fp = os.path.join(grad_dir, 'fp')
grad_dir_fn = os.path.join(grad_dir, 'fn')
grad_dir_true = os.path.join(grad_dir, 'true')

if not os.path.exists(grad_dir):
    os.makedirs(grad_dir)
if not os.path.exists(grad_dir_fp):
    os.makedirs(grad_dir_fp)
if not os.path.exists(grad_dir_fn):
    os.makedirs(grad_dir_fn)
if not os.path.exists(grad_dir_true):
    os.makedirs(grad_dir_true)

device = args.device
multi_gpu = args.multi_gpu
data_mode = args.data_mode
phase = args.phase
# size_label = args.size_label
load_path = args.load_path
cm = args.cm

# model = ResNet34()
# model = timm.create_model("resnet34d", pretrained=True, num_classes = 1)
# model = timm.create_model('resnet34.a1_in1k', pretrained=True, num_classes = 1)
model = ResNet34_att()

# gpu
device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

# multi-gpu
if multi_gpu == 'on':
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    
print('================================================')
print(device)

# test options
print("batch size: %d" % batch_size)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("mode: %s" % phase)
# print(f'Size label : {size_label}')
print(f'Load path : {load_path}')
print('================================================')

if  phase == 'test':
    ## Dataloader ##
    # dataset_test = ELcellDataset(data_dir=data_dir, data_mode=data_mode, phase=phase, size_label=size_label)
    dataset_test = ELcellDataset(data_dir=data_dir, data_mode=data_mode, phase='test')
    loader_test = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=8)

    ## Models ##
    # create network
    model.to(device)
    model.load_state_dict(torch.load(load_path))
    sigmoid = nn.Sigmoid()
    
    # Grad-CAM 
    # target_layers = [model.net.layer4[-1]] 
    # for param in model.net.layer4[-1].parameters():
    #     param.requires_grad = True

    # target_layers = [model.layer4[-1]] 
    target_layers = [model.module.model[-1]] 
    for param in model.module.model[-1].parameters():
        param.requires_grad = True
         
    # cam = GradCAM(model, target_layers, use_cuda=False)
    cam = GradCAM(model, target_layers)
    # ---------------------------------- TEST ---------------------------------- #        
    model.eval()
    fp_path, fn_path, true_path = [], [], []
    
    # 배치 안에 있는 grad-cam, 경로 저장 -> fn, fp 따로 저장 되도록 !!
    for i, data in enumerate(tqdm(loader_test, 0)):
        # forward pass
        # input, label = data['img_tensor'].to(device), data['label'].to(device)
        input, label = data['img_tensor'].to(device), data['label'].to(device)
        b, cells, c, h, w = input.shape
        input = input.view(b*cells, c, h, w)
        
        output = model(input)
        output = sigmoid(output)
        
        output = torch.max(output)   
        output = output.unsqueeze(-1)   
            
        # output = output.squeeze(-1)
        label = label.to(torch.float32)
        
        threshold = 0.5
        output = (output > threshold).int()
        
        # grad cam
        targets = [ClassifierOutputTarget(0)]
        el_cam = cam(input_tensor=input, targets=targets)
        
        # clone
        input_clone = input[0].permute(1,2,0)
        # print(input_clone.shape)
        input_clone = input_clone.detach().cpu().numpy().astype(np.float32) 
        # print(input_clone)
        input_clone = (input_clone - np.min(input_clone)) / (np.max(input_clone) - np.min(input_clone))
        # input_clone = input_clone.detach().cpu().numpy().astype(np.float32) / 255.0
        # print(input_clone)  
        
        visualization = show_cam_on_image(input_clone, el_cam[0], use_rgb=True)

        for j in range(len(output)):
            if output[j] != label[j]:
                if output[j] > label[j]:
                    fp_path.append((data['img_name'][j], f"grad_cam_{i}{j}.jpg"))
                    cv2.imwrite(os.path.join(grad_dir_fp, f"grad_cam_{i}{j}.jpg"), visualization)
                else:
                    fn_path.append((data['img_name'][j], f"grad_cam_{i}{j}.jpg"))
                    # fn_path.append(data['img_path'][j])
                    cv2.imwrite(os.path.join(grad_dir_fn, f"grad_cam_{i}{j}.jpg"), visualization)
            else:
                true_path.append((data['img_name'][j], f"grad_cam_{i}{j}.jpg"))
                # true_path.append(data['img_path'][j])
                cv2.imwrite(os.path.join(grad_dir_true, f"grad_cam_{i}{j}.jpg"), visualization)
                
        # Grad-CAM Result save
        # cv2.imwrite(os.path.join(log_dir, f"grad_cam_{i}.jpg"), visualization)
        # cam_path.append(data['img_path'])
        cam_df_fp = pd.DataFrame(fp_path)
        cam_df_fn = pd.DataFrame(fn_path)
        cam_df_true = pd.DataFrame(true_path)
        
        cam_df_fp.to_csv(f'{grad_dir}/cam_fp_path.csv', index=False)
        cam_df_fn.to_csv(f'{grad_dir}/cam_fn_path.csv', index=False)
        cam_df_true.to_csv(f'{grad_dir}/cam_true_path.csv', index=False)
      
        # cam_path.append(data['img_path'][i])
        if i % 100 == 0:
            print(f'Saving the Grad-CAM img_{i//100}')
        # writer_cam.add_image('Grad CAM', img_grid)