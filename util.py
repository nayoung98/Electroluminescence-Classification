import os
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch.nn.functional as F
import importlib
import yaml

# 네트워크를 저장
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                   './%s/model_epoch%d.pth' % (ckpt_dir, epoch))

# 네트워크를 로드
def load(path, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_list[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_list[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

# Early Stopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path + 'early_stop_model.pth')
        self.val_loss_min = val_loss

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion, ckpt_dir
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            
            torch.save(model.state_dict(), f'{ckpt_dir}/{epoch}.pth')

def edge_crop(img):
    w, h = img.size
        
    img = img.crop((0, 0, w, h-58))
    
    return img

def split_module_to_cells(img):
        img_array = np.array(img)
        h, w = img_array.shape[0], img_array.shape[1]

        cells =[]
        
        if w > 7800:
            # 셀 크기 계산
            cell_width = w // 26 # 300
            cell_height = h // 6 # 602

            for row in range(6):
                for col in range(26):
                    left = col * cell_width
                    upper = row * cell_height
                    right = (col + 1) * cell_width
                    lower = (row + 1) * cell_height

                    # 각 셀을 잘라내기
                    cell = img.crop((left, upper, right, lower))
                    cell = np.array(cell)
                    
                    cells.append(cell) # 156개 cells
        else: # 7612
            cell_width = w // 24 
            cell_height = h // 6

            for row in range(6):
                for col in range(24):
                    left = col * cell_width
                    upper = row * cell_height
                    right = (col + 1) * cell_width
                    lower = (row + 1) * cell_height

                    # 각 셀을 잘라내기
                    cell = img.crop((left, upper, right, lower))
                    cell = np.array(cell)
            
                    cells.append(cell) # 144개 cells

        # cells = np.array(cells)

        for i in range(len(cells)):
            cells[i] = Image.fromarray(cells[i])

        return cells 

def img_shuffle_col(img):
    w, h = img.size
    img_list = []
        
    dw = w // 4
        
    img1 = img.crop((0, 0, dw, h))
    img2 = img.crop((dw, 0, dw*2, h))
    img3 = img.crop((dw*2, 0, dw*3, h))
    img4 = img.crop((dw*3, 0, w, h))
        
    img_list.append(img1)
    img_list.append(img2)
    img_list.append(img3)
    img_list.append(img4)
    random.shuffle(img_list)
        
    merged_img = Image.new('RGB', (w, h))
    x_offset = 0
        
    for img in img_list:
        merged_img.paste(img, (x_offset, 0))
        x_offset += img.width
            
    return merged_img

def img_shuffle_row(img):
    w, h = img.size
    img_list = []
        
    dh = h // 3
        
    img1 = img.crop((0, 0, w, dh))
    img2 = img.crop((0, dh, w, dh*2))
    img3 = img.crop((0, dh*2, w, h))
        
    img_list.append(img1)
    img_list.append(img2)
    img_list.append(img3)
    random.shuffle(img_list)
        
    merged_img = Image.new('RGB', (w, h))
    y_offset = 0
        
    for img in img_list:
        merged_img.paste(img, (0, y_offset))
        y_offset += img.height
        
    return merged_img

def fix_parameters(model, rg=False):
    for layer in model.children():
        for param in layer.parameters():
            param.requires_grad = rg
        
def mixup_data(inputs, targets, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size, _, _, _ = inputs.shape
    rand_indices = torch.randperm(batch_size)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[rand_indices, :]
    targets_a, targets_b = targets, targets[rand_indices]
    return mixed_inputs, targets_a, targets_b, lam

def create_attention_map(energy):
    m, n = energy.shape

    attention_map = np.ones((m, m))
    block_size = m // 6
    
    for i in range(6): 
        for j in range(block_size):
            for k in range(block_size):
                attention_map[i * block_size + j, i * block_size +k] = energy[i * block_size + j, k]

    attention_map_tensor = torch.from_numpy(attention_map)
    
    return attention_map_tensor

def module2cell(image, cell_size):
    '''
        image: tensor_image
        cell_size: [h, w]
    '''
    b, c, h, w = image.shape
    
    folded_image = F.unfold(image, kernel_size=cell_size, stride=cell_size) # B x C*CH*CW x CN
    
    folded_image = folded_image.permute(0,2,1)                              # B x CN x C*CH*CW
    cells = folded_image.reshape(b, -1, c, cell_size[0], cell_size[1])      # B x CN x C x CH X CW
    
    return cells

def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    for k in config.keys():
        if k not in ['target', 'params']:
            print('warning: Unused key {} while instantiating {}'.format(k, config['target']))
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))

def strlist_to_list(strlist):
    new_list = list(map(int, strlist.strip('][').replace("'", '').split(',')))

    return new_list

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config
