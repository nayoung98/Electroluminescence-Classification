import argparse
import ast

import os
import numpy as np
import pandas as pd
import timm
from tqdm import tqdm
import itertools
import json

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

parser.add_argument("--saved_name", default='full', type=str, help="The saved name of classification result")

args  = parser.parse_args() # parsing

batch_size = args.batch_size
model_name = args.model_name
load_path = args.load_path
load_best_epoch = args.load_best_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = ckpt_dir
save_dir = os.path.join(ckpt_dir, f'epoch{load_best_epoch}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = args.device
multi_gpu = args.multi_gpu
data_mode = args.data_mode
phase = args.phase
load_path = args.load_path
saved_name = args.saved_name

model_config_path = '/home/sliver/SDN/Electroluminescence-Classification/model_config.yaml'
config = load_yaml_config(model_config_path)
model_config = config[model_name]
model = instantiate_from_config(model_config)

# gpu
device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
device_ids = [0, 1, 2, 3, 6]

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

def evaluation(gt, pred, prefix, threshold):
    # print(f'=================== {prefix} Evaluation ===================')
    bal_acc, acc, precision, recall, f1 = balanced_metric(gt, pred) 
    # print(f'Balanced Accuracy : {bal_acc}')
    # print(f'Accuracy : {acc}')
    # print(f'Precision : {precision}')
    # print(f'Recall : {recall}')
    # print(f'F1 : {f1}')
    result = (threshold, bal_acc, acc, precision, recall, f1)
    # confusion matrix
    cm = confusion_matrix(gt, pred)
    classes = ['Negative', 'Positive']
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{prefix}_{threshold}.png'))
    plt.close()
    return result

header = ['Threshold', 'Bal_acc', 'Acc', 'Precision', 'Recall', 'F1']
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# threshold_list = [0.5]
result = []

if  phase == 'test':
    ## Dataloader ##
    # dataset_test = ELmoduleCustomDataset_700(data_dir=data_dir, phase='test')
    dataset_test = ELmoduleCustomDataset(data_dir=data_dir, phase='test')
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    ## Models ##
    # create network
    model.to(device)
    model.load_state_dict(torch.load(load_path))
    sigmoid = nn.Sigmoid()
    
    # ---------------------------------- TEST ---------------------------------- #        
    with torch.no_grad():   
        model.eval()        
        module_prob, module_label = [], []
        fp_path, fn_path = [], []
        tp_path, tn_path = [], []
        chk_list = []
        # header_chk = ['filename', 'prob', 'pred', 'gt']
        header_chk = ['filename', 'prob', 'pred', 'gt', 'img_path']
    
        for i, data in enumerate(tqdm(loader_test, 0)):
            # forward pass
            img_name = data['filename']
            input = data['img_tensor'].to(device)
            label = data['label'].to(device)
        
            # prediction
            output = model(input)
            output = sigmoid(output) 
            output = output.squeeze(-1) # b
            
            prob = output
            pred = (prob > 0.5).float()
            pred = pred.detach().cpu().numpy()
            
            output = output.detach().cpu().numpy()
            
            label = label.to(torch.float32)
            label = label.detach().cpu().numpy()
            
            for i in range(len(output)): # batch_size
                # chk_list.append((img_name[i], prob[i].item(), pred[i].item(), label[i].item()))
                if pred[i] != label[i]:
                    if pred[i] > label[i]: # false positive
                        fp_path.append((img_name[i], prob[i].item(), pred[i].item(), label[i].item(), data['img_path'][i]))
                    else: # false negative
                        fn_path.append((img_name[i], prob[i].item(), pred[i].item(), label[i].item(), data['img_path'][i]))
                else:
                    if label[i].item() == 1.0: 
                        tp_path.append((img_name[i], prob[i].item(), pred[i].item(), label[i].item(), data['img_path'][i]))
                    else:
                        tn_path.append((img_name[i], prob[i].item(), pred[i].item(), label[i].item(), data['img_path'][i]))
                # print(chk_list)
                # sys.exit()
            module_prob.extend(output)
            module_label.extend(label)

        # csv file로 fp, fn 경로 저장
        # df_chk = pd.DataFrame(chk_list)
        df_fp = pd.DataFrame(fp_path)
        df_fn = pd.DataFrame(fn_path)
        df_tp = pd.DataFrame(tp_path)
        df_tn = pd.DataFrame(tn_path)
        
        # df_chk.to_csv(f'{save_dir}/chk_list.csv', index=False, header=header_chk) 
        df_fp.to_csv(f'{save_dir}/fp_path.csv', index=False, header=header_chk) 
        df_fn.to_csv(f'{save_dir}/fn_path.csv', index=False, header=header_chk)
        df_tp.to_csv(f'{save_dir}/tp_path.csv', index=False, header=header_chk) 
        df_tn.to_csv(f'{save_dir}/tn_path.csv', index=False, header=header_chk) 
        
        module_prob_array = np.array(module_prob)
        module_label_array = np.array(module_label)
        
    for _, threshold in enumerate(threshold_list): 
        output_module = (module_prob_array > threshold).astype(np.int32) 
        result.append(evaluation(module_label_array, output_module, 'module', threshold))

result_df = pd.DataFrame(result)
result_df.to_csv(os.path.join(save_dir, 'module_result_origin.csv'), header=header, index=False)
print('The End!')
