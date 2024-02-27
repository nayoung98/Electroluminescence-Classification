import argparse

import cv2

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
from visualizer import *

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
parser.add_argument("--load_path", default="./checkpoints", type=str, help="The path of the trained model")

args  = parser.parse_args() # parsing

batch_size = args.batch_size

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir

device = args.device
multi_gpu = args.multi_gpu
data_mode = args.data_mode
phase = args.phase

load_path = args.load_path

# model = timm.create_model("resnet34d", pretrained=True, num_classes = 1)
# model = ResNet34()
# model = ResNet50()
# model = EfficientNetB3()
# model = timm.create_model('seresnet50.a1_in1k', pretrained=True, num_classes = 1)
# model = timm.create_model('resnext50d_32x4d.bt_in1k', pretrained=True, num_classes = 1)
# model = timm.create_model('seresnext50_32x4d.racm_in1k', pretrained=True, num_classes = 1)
model = timm.create_model('resnet34.a1_in1k', pretrained=True, num_classes = 1)

# model = ResNet34_att()

# gpu
device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

# multi-gpu
if multi_gpu == 'on':
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
    
print('================================================')
print(device)

# test options
print("batch size: %d" % batch_size)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("mode: %s" % phase)
print(f'Load path : {load_path}')
print('================================================')

# Tensorboard
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
writer_test = SummaryWriter(log_dir=os.path.join(log_dir, 'test'))

if  phase == 'test':
    ## Dataloader ##
    dataset_test = ELcellDataset(data_dir=data_dir, data_mode=data_mode, phase='test')
    loader_test = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=8)

    ## Models ##
    # create network
    model.to(device)
    model.load_state_dict(torch.load(load_path))
    sigmoid = nn.Sigmoid()
    
    # ---------------------------------- TEST ---------------------------------- #        
    with torch.no_grad():   
        model.eval()
        y_pred, y_true = [], []
        fp_path, fn_path = [], []
        true_path = []
        # prob_list = []
        
        for i, data in enumerate(tqdm(loader_test, 0)):
            # forward pass
            input, label = data['img_tensor'].to(device), data['label'].to(device)
            b, cells, c, h, w = input.shape
            input = input.view(b*cells, c, h, w)
                        
            output = model(input) 
            output = sigmoid(output) # (144, 1)
            cell_size = output.shape[0]
            
            # max
            output = torch.max(output)   
            # mean  
            # output = torch.mean(output)   
            
            output = output.unsqueeze(-1)   
            # prob_list.append(output)
            label = label.to(torch.float32)
            
            threshold = 0.5
            output = (output > threshold).int()
            
            label = label.detach().cpu().numpy()
            output = output.detach().cpu().numpy()
            
            # false data 확인
            for i in range(len(output)): # batch_size
                if output[i] != label[i]:
                    if output[i] > label[i]: # false positive
                        fp_path.append((data['filename'][i], data['label'][i]))
                    else: # false negative
                        fn_path.append((data['filename'][i], data['label'][i]))
                else:
                    true_path.append(data['filename'][i])
                
            y_true.extend(label)
            y_pred.extend(output)
    
        # csv file로 fp, fn 경로 저장
        df_fp = pd.DataFrame(fp_path)
        df_fn = pd.DataFrame(fn_path)
        df_true = pd.DataFrame(true_path)
        
        # prob_df = pd.DataFrame(prob_list)
        # prob_df.to_csv('./prob_df.csv', index=False)
        
        df_fp.to_csv(f'{ckpt_dir}/fp_path.csv', index=False) 
        df_fn.to_csv(f'{ckpt_dir}/fn_path.csv', index=False)
        df_true.to_csv(f'{ckpt_dir}/true_path.csv', index=False)
        
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
        plt.close()
        