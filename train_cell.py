import argparse

import os
import numpy as np
import timm
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torchvision.ops as ops

from models import *
from dataset import *
from util import *
from engine import *
from metric import *
from loss import *

# parser
parser = argparse.ArgumentParser(description= 'Train',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")
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
parser.add_argument("--load_epoch", default=0, type=int, help="The epoch of the loaded model")
parser.add_argument('--patience', type=int, default=10, help='patience epochs for early stopping (default: 30)')

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
load_epoch = args.load_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = ckpt_dir
patience = args.patience

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# net = timm.create_model("resnet34d", pretrained=True, num_classes = 1)
# net = ResNet34()
# net = ResNet50()
# net = EfficientNetB3()
# net = timm.create_model('seresnet50.a1_in1k', pretrained=True, num_classes = 1)
# net = timm.create_model('resnext50d_32x4d.bt_in1k', pretrained=True, num_classes = 1)
# net = timm.create_model('seresnext50_32x4d.racm_in1k', pretrained=True, num_classes = 1)
# model = timm.create_model('resnet34.a1_in1k', pretrained=True, num_classes = 1)
# net = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes = 1)
# print(net)

model = ResNet34_att()

# gpu
device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
# device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

if multi_gpu == 'on':
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
        
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
print('================================================')

# Set random seed
np.random.seed(0)
random.seed(0)

## Dataloader ##
train_dataset = ELcellDataset(data_dir=data_dir, data_mode=data_mode, phase='train')
valid_dataset = ELcellDataset(data_dir=data_dir, data_mode=data_mode, phase='valid')
    
loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
loader_val = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

## Models ##
model.to(device)

# loss function
# criterion_focal = FocalLoss().to(device)
criterion = nn.BCELoss().to(device)
criterion_cont = ContrastiveLoss().to(device)

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=lr)

# early stopping
early_stopping = EarlyStopping(patience=patience, verbose=True, path=ckpt_dir)

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
    model.load_state_dict(torch.load(PATH))
    st_epoch = load_epoch 
    print(f'Model Load :{load_epoch}')

    
for epoch in range(st_epoch+1, num_epoch+1):
    # ---------------------------------- Training ---------------------------------- #
        
    # train_loss_list = train_cell(model, loader_train, optim, criterion, device)
    # train_loss_list = train_cell_cont(model, loader_train, optim, criterion, device)
    # train_loss_list = train_cell_loss(model, loader_train, optim, criterion, criterion, device) # loss 추가
    train_loss_list = train_cell_loss_non(model, loader_train, optim, criterion, criterion, device) # loss 추가 
    train_loss = np.mean(train_loss_list)
    
    print('TRAIN: EPOCH %04d/%04d | LOSS %.4f' % (epoch, num_epoch, train_loss))
    writer_train.add_scalar('loss', train_loss, epoch)
    
    early_stopping(train_loss, model)
    if early_stopping.early_stop:
        print(f"Saving the early stopping model - epoch : {epoch}")
        break
    
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'{ckpt_dir}/{epoch}.pth')
        print(f'Saving the model - epoch : {epoch}')

    if epoch == num_epoch:
        torch.save(model.state_dict(), f'{ckpt_dir}/final_{epoch}.pth')
        print(f'Saving the final model - epoch : {epoch}')
        
    # ---------------------------------- Validation ---------------------------------- #
        
    with torch.no_grad():   
            
        # y_true, y_pred, valid_loss_list = validate_cell(model, loader_val, criterion, device)
        # y_true, y_pred, valid_loss_list = validate_cell_loss(model, loader_val, criterion, criterion, device)
        y_true, y_pred, valid_loss_list = validate_cell_loss_non(model, loader_val, criterion, criterion, device)
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
            torch.save(model.state_dict(), f'{ckpt_dir}/best_{epoch}.pth')
            print(f'Saving the best model - epoch : {epoch}')
            
        # if acc > best_val_accuracy:
        #     torch.save({'model': model.module.state_dict()}, f'{ckpt_dir}/{epoch}.pth')
        #     # model.to('cpu')
        #     # torch.save(model.state_dict(), f'{ckpt_dir}/{epoch}_cpu.pth')
        #     # model.to(device)
        #     print(f'Saving the best model - epoch : {epoch}')
        
writer_train.close()
writer_val.close()
