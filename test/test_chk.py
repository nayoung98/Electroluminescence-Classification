import argparse

import os
import numpy as np
from tqdm import tqdm
import pandas as pd
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

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=2, type=int, dest="batch_size")

parser.add_argument("--data_dir", default='/home/pink/nayoung/el/datasets/el', type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='./log', type=str, dest="log_dir")

parser.add_argument("--data_mode", default="first", type=str, help="[first | second]")
parser.add_argument("--phase", default="train", type=str, help="[train | test]")
parser.add_argument("--size_label", default=4, type=int, help="[4 | 8 | 16]")


args  = parser.parse_args() # parsing

lr = args.lr
batch_size = args.batch_size

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = ckpt_dir

data_mode = args.data_mode
phase = args.phase
size_label = args.size_label

PATH = '/home/pink/nayoung/el/main/checkpoints/230825_resnet34_4/best_49.pth'

net = EfficientNetB3()

# gpu
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
       
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     net = nn.DataParallel(net, device_ids=[3, 4])
    
print('================================================')
print(device)

# train options
print("learning rate: {}".format(lr))
print("batch size: %d" % batch_size)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("mode: %s" % phase)
print('================================================')

# Set random seed
np.random.seed(0)
random.seed(0)

# Tensorboard
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer_test = SummaryWriter(log_dir=os.path.join(log_dir, 'test'))

if  phase == 'test':
    ## Dataloader ##
    dataset_test = ELmoduleCustomDataset(data_dir=data_dir, data_mode=data_mode, phase=phase, size_label=size_label)
    loader_test = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=8)

    ## Models ##
    # create network
    net.to(device)
    net.load_state_dict(torch.load(PATH))

    # ---------------------------------- TEST ---------------------------------- #        
    with torch.no_grad():   
        net.eval()
        y_pred, y_true = [], []
        fp_path, fn_path = [], []
        true_path = []
        
        for i, data in enumerate(tqdm(loader_test, 0)):
            # forward pass
            input, label = data['img_tensor'].to(device), data['label'].to(device)
            
            output = net(input)
            output = output.squeeze(-1)
            label = label.to(torch.float32)
            
            threshold = 0.5
            output = (output > threshold).int()
            
            label = label.detach().cpu().numpy()
            output = output.detach().cpu().numpy()
            
            # false data 확인
            for i in range(len(output)): # batch_size
                if output[i] != label[i]:
                    img_grid = torchvision.utils.make_grid(input[i])
                    if output[i] > label[i]: # false positive
                        writer_test.add_image('False positive', img_grid)
                        fp_path.append((data['img_path'][i], data['label'][i]))
                    else: # false negative
                        writer_test.add_image('False negative', img_grid)
                        fn_path.append((data['img_path'][i], data['label'][i]))
                        # fn_path.append(data['img_path'][i])
                else:
                    true_path.append(data['img_path'][i])
                
            y_true.extend(label)
            y_pred.extend(output)
        
        # csv file로 fp, fn 경로 저장
        df_fp = pd.DataFrame(fp_path)
        df_fn = pd.DataFrame(fn_path)
        df_true = pd.DataFrame(true_path)
        
        df_fp.to_csv(f'{ckpt_dir}/fp_path.csv', index=False) 
        df_fn.to_csv(f'{ckpt_dir}/fn_path.csv', index=False)
        df_true.to_csv(f'{ckpt_dir}/true_path.csv', index=False)
        
        print('Saving the path of the false positive')
        print('Saving the path of the false negative')
        print('Saving the path of the true cases')
        
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
        plt.savefig(os.path.join(ckpt_dir, 'confusion_matrix_13.png')) # 에폭에 맞게 수정 
        plt.close()
        

            
           
