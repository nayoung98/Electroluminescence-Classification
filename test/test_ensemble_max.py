import argparse

import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import *
from dataset import *
from metric import *

# parser
parser = argparse.ArgumentParser(description= 'Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--batch_size", default=2, type=int, dest="batch_size")

parser.add_argument("--data_dir", default='/home/pink/nayoung/el/datasets/el', type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='./log', type=str, dest="log_dir")

parser.add_argument("--device", default=0, type=str, help="The GPU number")
parser.add_argument("--multi_gpu", default="off", type=str, help="[on | off]")
parser.add_argument("--data_mode", default="first", type=str, help="[first | second]")
parser.add_argument("--phase", default="train", type=str, help="[train | test]")
parser.add_argument("--cm", default="resnet", type=str, help="The number of confusion matrix")


args  = parser.parse_args() # parsing

batch_size = args.batch_size

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = ckpt_dir

device = args.device
multi_gpu = args.multi_gpu
data_mode = args.data_mode
phase = args.phase
cm = args.cm

data_chk_dir = os.path.join(ckpt_dir, 'data_chk')
data_chk_fp = os.path.join(data_chk_dir, 'fp')
data_chk_fn = os.path.join(data_chk_dir, 'fn')
data_chk_true = os.path.join(data_chk_dir, 'true')

if not os.path.exists(data_chk_dir):
    os.makedirs(data_chk_dir)
if not os.path.exists(data_chk_fp):
    os.makedirs(data_chk_fp)
if not os.path.exists(data_chk_fn):
    os.makedirs(data_chk_fn)
if not os.path.exists(data_chk_true):
    os.makedirs(data_chk_true)
    
# Paths of the three models
PATH1 = '/home/pink/nayoung/el/main/checkpoints/231005_resneXt50_8/best_15.pth'
PATH2 = '/home/pink/nayoung/el/main/checkpoints/231004_resnet34D_8/best_17.pth'
PATH3 = '/home/pink/nayoung/el/main/checkpoints/231007_SEresneXt50_8/best_10.pth'
PATH4 = '/home/pink/nayoung/el/main/checkpoints/231004_resnet34_8/best_20.pth'
# PATH5 = '/home/pink/nayoung/el/main/checkpoints/231004_resnet34D_4/best_30.pth'

# Loading the three models
net1 = timm.create_model('resnext50d_32x4d.bt_in1k', pretrained=True, num_classes = 1)
net2 = timm.create_model("resnet34d", pretrained=True, num_classes = 1)
net3 = timm.create_model('seresnext50_32x4d.racm_in1k', pretrained=True, num_classes = 1)
net4 = ResNet34()
# net5 = timm.create_model("resnet34d", pretrained=True, num_classes = 1)

# gpu
device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

if multi_gpu == 'on':
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net1 = nn.DataParallel(net1, device_ids=[2, 3])

print('================================================')
print(device)

# test options
print("batch size: %d" % batch_size)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("mode: %s" % phase)
print('================================================')

# Tensorboard
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
    
# writer_test = SummaryWriter(log_dir=os.path.join(log_dir, 'test'))

if  phase == 'test':
    ## Dataloader ##
    dataset_test = ELmoduleEnsembleDataset(data_dir=data_dir, data_mode=data_mode, phase=phase)
    loader_test = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=8)

    ## Models ##
    # create network
    net1.to(device)
    net2.to(device)
    net3.to(device)
    net4.to(device)
    # net5.to(device)
    
    net1.load_state_dict(torch.load(PATH1))
    net2.load_state_dict(torch.load(PATH2))
    net3.load_state_dict(torch.load(PATH3))
    net4.load_state_dict(torch.load(PATH4))
    # net5.load_state_dict(torch.load(PATH5))
    
    # Sigmoid function
    sigmoid = nn.Sigmoid()

    # ---------------------------------- TEST ---------------------------------- #        
    with torch.no_grad():   
        net1.eval()
        net2.eval()
        net3.eval()
        net4.eval()
        # net5.eval()
                
        y_pred, y_true = [], []
        fp_path, fn_path, true_path = [], [], []
        
        for i, data in enumerate(tqdm(loader_test, 0)):
            # forward pass
            input_4 = data['img_tensor_4'].to(device)
            input_8 = data['img_tensor_8'].to(device)
    
            label = data['label'].to(device)
            
            # probabilities
            output1 = net1(input_8)
            output2 = net2(input_8)
            output3 = net3(input_8)
            output4 = net4(input_8)
            # output5 = net5(input_4)
            
            prob1 = sigmoid(output1)
            prob2 = sigmoid(output2)
            prob3 = sigmoid(output3)
            prob4 = sigmoid(output4)
            # prob5 = sigmoid(output5)
           
            # max
            max_value1, _ = torch.max(torch.stack([prob1[0], prob2[0], prob3[0], prob4[0]]), dim=0)
            max_value2, _ = torch.max(torch.stack([prob1[1], prob2[1], prob3[1], prob4[1]]), dim=0)
            max_value3, _ = torch.max(torch.stack([prob1[2], prob2[2], prob3[2], prob4[2]]), dim=0)
            max_value4, _ = torch.max(torch.stack([prob1[3], prob2[3], prob3[3], prob4[3]]), dim=0)
           
            output = torch.cat([max_value1, max_value2, max_value3, max_value4], dim=0) # (b)
            label = label.to(torch.float32)

            threshold = 0.5
            output = (output > threshold).int()
            
            # false data 확인
            for i in range(len(output)): # batch_size
                if output[i] != label[i]:
                    if output[i] > label[i]: # false positive
                        fp_path.append((data['image_path'][i], data['label'][i]))
                        # fp_path.append(data['img_path'][i])
                    else: # false negative
                        fn_path.append((data['image_path'][i], data['label'][i]))
                        # fn_path.append(data['img_path'][i])
                else:
                    true_path.append(data['image_path'][i])

            label = label.detach().cpu().numpy()
            output = output.detach().cpu().numpy()

            y_true.extend(label)
            y_pred.extend(output)
        
        fp_path_df = pd.DataFrame(fp_path)
        fn_path_df = pd.DataFrame(fn_path)
        true_path_df = pd.DataFrame(true_path)
        
        # prob1_df = pd.DataFrame(prob1_list)
        # prob1_df.to_csv('./prob1_df.csv', index=False)

        fp_path_df.to_csv(f'{data_chk_fp}/fp_path.csv', index=False)
        fn_path_df.to_csv(f'{data_chk_fn}/fn_path.csv', index=False)
        true_path_df.to_csv(f'{data_chk_true}/true_path.csv', index=False)
        
        # evaluation metric
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
        plt.savefig(os.path.join(ckpt_dir, f'confusion_matrix_{cm}.png')) # 에폭에 맞게 수정 
        plt.close()

            
           
