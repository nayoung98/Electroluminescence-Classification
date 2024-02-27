import argparse

import os
import numpy as np
import timm
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from models import *
from dataset import *
from util import *
from engine import *
from metric import *

# parser
parser = argparse.ArgumentParser(description= 'Train',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=2, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default='/home/pink/nayoung/el/datasets/el', type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='./log', type=str, dest="log_dir")

parser.add_argument("--device", default=0, type=str, help="The GPU number")
parser.add_argument("--multi_gpu", default="off", type=str, help="[on | off]")
parser.add_argument("--data_mode", default="first", type=str, help="[first | second]")
parser.add_argument("--phase", default="train", type=str, help="[train | valid | test]")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
parser.add_argument("--train_continue_path", default="./checkpoint", type=str, help="The path of the model to load")
parser.add_argument("--size_label", default=4, type=int, help="[4 | 8 | 16]")
parser.add_argument("--load_epoch", default=0, type=int, help="The epoch of the loaded model")


args  = parser.parse_args() # parsing

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

device = args.device
multi_gpu = args.multi_gpu
data_mode = args.data_mode
phase = args.phase
train_continue = args.train_continue
train_continue_path = args.train_continue_path
size_label = args.size_label
load_epoch = args.load_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = ckpt_dir

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

PATH1 = '/home/pink/nayoung/el/main/checkpoints/231005_resneXt50_8/best_15.pth'
PATH2 = '/home/pink/nayoung/el/main/checkpoints/231004_resnet34D_8/best_17.pth'
PATH3 = '/home/pink/nayoung/el/main/checkpoints/231007_SEresneXt50_8/best_10.pth'
PATH4 = '/home/pink/nayoung/el/main/checkpoints/231004_resnet34_8/best_20.pth'
PATH5 = '/home/pink/nayoung/el/main/checkpoints/231004_resnet50_8/best_36.pth'

net = Ensemble_concat2(PATH1, PATH2, PATH3, PATH4, PATH5)

# print(net)

# gpu
device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

if multi_gpu == 'on':
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=[4, 5, 6, 7])
        

print('================================================')
print(device)

# train options
print("learning rate: {}".format(lr))
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("mode: %s" % phase)
print("train continue: %s" % train_continue)
print((f"train continue path : {train_continue_path}"))
print((f"image size : img / {size_label}"))
print('================================================')

# Set random seed
np.random.seed(0)
random.seed(0)

## Dataloader ##
train_dataset = ELmoduleCustomDataset(data_dir=data_dir, data_mode=data_mode, phase='train', size_label=size_label)
valid_dataset = ELmoduleCustomDataset(data_dir=data_dir, data_mode=data_mode, phase='valid', size_label=size_label)
    
loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
loader_val = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

## Models ##
# create network
net.to(device)

# loss function
criterion = nn.BCEWithLogitsLoss().to(device)

# optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)
# schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=5, eta_min=1e-4)

# tensor board
if not os.path.exists(log_dir):
    os.makedirs(os.path.join(log_dir, 'train'))
    os.makedirs(os.path.join(log_dir, 'val'))

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

st_epoch = 0 
best_val_accuracy = 0.0

if train_continue == "on":
    PATH = train_continue_path
    net.load_state_dict(torch.load(PATH))
    st_epoch = load_epoch 
    print(f'Model Load :{load_epoch}')

for epoch in range(st_epoch+1, num_epoch+1):
    # ---------------------------------- Training ---------------------------------- #
        
    train_loss_list = train(net, loader_train, optim, criterion, device)
    train_loss = np.mean(train_loss_list)
            
    print('TRAIN: EPOCH %04d/%04d | LOSS %.4f' % (epoch, num_epoch, train_loss))
    writer_train.add_scalar('loss', train_loss, epoch)

    if epoch % 5 == 0:
        torch.save(net.state_dict(), f'{ckpt_dir}/{epoch}.pth')
        print(f'Saving the model - epoch : {epoch}')

    # ---------------------------------- Validation ---------------------------------- #
        
    with torch.no_grad():   
            
        y_true, y_pred, valid_loss_list = validate(net, loader_val, criterion, device)
        valid_loss = np.mean(valid_loss_list)
        acc, precision, recall, f1 = make_metric(y_true, y_pred)
        
        print(f'Accuracy : {acc}')
        print(f'Precision : {precision}')
        print(f'Recall : {recall}')
        print(f'F1 : {f1}')
        print('VALID: EPOCH %04d/%04d | LOSS %.4f' % (epoch, num_epoch, valid_loss))
            
        writer_val.add_scalar('loss', valid_loss, epoch)
        writer_val.add_scalar('acc', acc, epoch)
        writer_val.add_scalar('f1', f1, epoch)
        
        if acc > best_val_accuracy:
            best_val_accuracy = acc
            torch.save(net.state_dict(), f'{ckpt_dir}/best_{epoch}.pth')
            print(f'Saving the best model - epoch : {epoch}')
        
        # if  train_loss > 3 * valid_loss:
        #     torch.save(net.state_dict(), f'{ckpt_dir}/early_{epoch}.pth')
        #     print(f"Saving the early stopping model for epoch: {epoch}")
        #     break
        
writer_train.close()
writer_val.close()
