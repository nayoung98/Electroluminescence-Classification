import argparse

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import timm
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from models import *
from dataset import *
from metric import *
from engine import *
from util import *
from visualizer import *
import numpy as np

model = timm.create_model('resnet34.a1_in1k', pretrained=True, num_classes = 1)
# model = ResNet34_att()

device = torch.device(f'cuda:3' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=[3, 4, 5])
    
data_dir = '/home/pink/nayoung/el/datasets/el'
ckpt_dir = '/home/pink/nayoung/el/main/checkpoints/240105_resnet34_loss_non'
img_dir = f'{ckpt_dir}/img'

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

data_mode = 'first'
load_path = '/home/pink/nayoung/el/main/checkpoints/240105_resnet34_loss_non/best_38.pth'

dataset_test = ELcellDataset(data_dir=data_dir, data_mode=data_mode, phase='test')
loader_test = DataLoader(dataset_test, batch_size=1,shuffle=False, num_workers=8)

## Models ##
# create network
model.to(device)
model.load_state_dict(torch.load(load_path))
sigmoid = nn.Sigmoid()

with torch.no_grad():   
    model.eval()
    y_pred, y_true = [], []
    fp_path, fn_path = [], []
    true_path = []
    prob_list = []
    
    for i, data in enumerate(tqdm(loader_test, 0)):
        x = 0
        input, label = data['img_tensor'].to(device), data['label'].to(device)
        b, cells, c, h, w = input.shape
        input = input.view(b*cells, c, h, w)
                    
        output = model(input) 
        output = sigmoid(output) # (cells, 1)
        
        cell_size = output.shape[0]
        
        ########################## img visualization ##########################
        img_name = data['filename']
        img_p = data['img_path']
        
        img = cv2.imread(img_p[0])
        cropped = img[:-58, :]
        
        rows = int(cropped.shape[1]/300) # 26 or 24
        cols = int(cropped.shape[0]/600) # 6
        # print(rows, cols)
        # print(len(output))
        mask = np.zeros((cropped.shape[0], cropped.shape[1], 4), dtype=np.uint8)
        
        for col in range(cols):
            for row in range(rows):
                if output[x] >= 0.5:
                    clr = (0, 0, 255, 50)
                else:
                    clr = (255, 0, 0, 50)
                x +=1
                cv2.rectangle(mask, (row*300, col*600), (row*300 + 300, col*600 + 600), clr, -1)
  
        img_output = overlay_transparent(cropped, mask, 0, 0)
        
        cv2.imwrite(f'{img_dir}/{img_name}_{label}.jpg', img_output) 
        print(f'Saving the image{img_name}')
        
        ########################## img check ##########################
        output = torch.max(output)     
        output = output.unsqueeze(-1)   
        
        # prob 확인
        prob_list.append((output, data['filename']))
        label = label.to(torch.float32)
        
        threshold = 0.5
        output = (output > threshold).int()
        
        label = label.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
            
        # false data 확인
        for i in range(len(output)): # batch_size
            if output[i] != label[i]:
                if output[i] > label[i]: # false positive
                    fp_path.append((data['img_path'][i], data['label'][i]))
                else: # false negative
                    fn_path.append((data['img_path'][i], data['label'][i]))
            else:
                true_path.append(data['img_path'][i])
                
        y_true.extend(label)
        y_pred.extend(output)
    
    # csv file로 fp, fn 경로 저장
    df_fp = pd.DataFrame(fp_path)
    df_fn = pd.DataFrame(fn_path)
    df_true = pd.DataFrame(true_path)
    df_prob = pd.DataFrame(prob_list)
            
    df_fp.to_csv(f'{ckpt_dir}/fp_path.csv', index=False) 
    df_fn.to_csv(f'{ckpt_dir}/fn_path.csv', index=False)
    df_true.to_csv(f'{ckpt_dir}/true_path.csv', index=False)
    df_prob.to_csv(f'{ckpt_dir}/prob_list.csv', index=False)
        
    print('Saving the path of the false positive')
    print('Saving the path of the false negative')
    print('Saving the path of the true cases')
        
    acc, precision, recall, f1 = make_metric(y_true, y_pred)
    print(f'Accuracy : {acc}')
    print(f'Precision : {precision}')
    print(f'Recall : {recall}')
    print(f'F1 : {f1}')
    
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Negative', 'Positive']
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(ckpt_dir, f'confusion_matrix.png'))
    