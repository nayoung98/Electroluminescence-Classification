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
save_dir = os.path.join(ckpt_dir, f'epoch{load_best_epoch}')
# save_dir = "/home/sliver/SDN/Electroluminescence-Classification/dy/result/resnet50d_2"
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

threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# threshold_list = [0.5]

header = ['Threshold', 'Bal_acc', 'Acc', 'Precision', 'Recall', 'F1']
module_result, cell_result = [], []

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

def evaluation(gt, pred, prefix, threshold):
    print(f'=================== {prefix} Evaluation ===================')
    bal_acc, acc, precision, recall, f1 = balanced_metric(gt, pred) 
    print(f'Balanced Accuracy : {bal_acc}')
    print(f'Accuracy : {acc}')
    print(f'Precision : {precision}')
    print(f'Recall : {recall}')
    print(f'F1 : {f1}')
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

module_pred, cell_pred = np.array([]), np.array([])
module_gt, cell_gt = np.array([]), np.array([])
module_prob, cell_prob = np.array([]), np.array([])

chk_list = []
if  phase == 'test':
    ## Dataloader ##
    dataset_test = ELcellcropEvaluationDataset(data_dir=data_dir, cell_size=[150, 75], phase='test')
    # dataset_test = ELmulticellDataset(data_dir=data_dir, cell_size=[300, 150], phase='test')
    # dataset_test = ELcellEvaluationDataset(data_dir=data_dir, data_mode=data_mode, phase='test')
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    ## Models ##
    model.to(device)
    model.load_state_dict(torch.load(load_path))

    sigmoid = nn.Sigmoid()
    fp_path, fn_path, true_path = [], [], []
    header_chk = ['img_name', 'module_label', 'module_pred', 'module_prob', 'img_path']
    header_chk = ['filename', 'prob', 'pred', 'gt', 'img_path']
    header_cell = ['img_name', 'cell_number', 'pred', 'gt', 'cell_prob']
    cell_chk_list = []
    cell_len_chk_list = []
    # ---------------------------------- TEST ---------------------------------- #        
    with torch.no_grad():   
        model.eval()
        results = {}

        for i, data in enumerate(tqdm(loader_test, 0)):
            # forward pass
            img_name = data['filename']
        
            input = data['img_tensor']
            input = input.to(device)
                        
            b, cells, c, h, w = input.shape
            gt_module = data['label_module']
            label_module = data['label_module'].numpy()
            
            label = data['label_cell']
            label_test = [strlist_to_list(c) for c in label]
   
            label_test = np.array(list(itertools.chain(*label_test))).reshape(b, -1)
            
            # cell 라벨이 2인 경우 제외 코드
            valid_indices = label_test != 2
            # print(valid_indices.shape) # (1, cells)
            
            module_gt = np.append(module_gt,label_module.reshape(-1))
            cell_gt = np.append(cell_gt,label_test[valid_indices].reshape(-1))
            # cell_gt = np.append(cell_gt,label_test.reshape(-1))
      
            
            # prediction
            input = input.view(b*cells, c, h, w)
            
            output = model(input)
            output = sigmoid(output) # (b*cells, 1)
            output_cell = output.clone()
            output = output.view(b, cells)
            
            # cell prediction
            output_cell_prob = output_cell.detach().cpu().numpy() # (cells, 1)
             
            # module prediction
            output_module_prob, _ = torch.max(output, dim=1)
            output_module = (output_module_prob > 0.5).int()
            prob_module = output_module_prob.clone()
            output_module_prob = output_module_prob.view(b).detach().cpu().numpy()
            
            # # false data 확인
            # for j in range(len(output_module_prob)): # batch_size
            #     if output_module[j] != gt_module[j]:
            #         if output_module[j] > gt_module[j]: # false positive
            #             fp_path.append((img_name[j], gt_module[j].item(), output_module[j].item(), prob_module[j].item(), data['img_path'][j]))
            #         else: # false negative
            #             fn_path.append((img_name[j], gt_module[j].item(), output_module[j].item(), prob_module[j].item(), data['img_path'][j]))
            #     else:
            #         true_path.append((img_name[j], gt_module[j].item(), output_module[j].item(), prob_module[j].item(), data['img_path'][j]))
           
            # # cell label 2인 경우 제외하고 계산

            valid_out_cell = output.squeeze(0)[valid_indices]
            output_cell = valid_out_cell.clone()
            valid_out_cell = valid_out_cell.detach().cpu().numpy()
         
            # aggregate results
            module_prob = np.append(module_prob, output_module_prob)
            cell_prob = np.append(cell_prob, valid_out_cell)
            # cell_prob = np.append(cell_prob, output_cell_prob)
            # print(cell_prob)
            # print(len(cell_prob))
            
            # output_cell = (cell_prob.reshape(-1) > 0.5).astype(np.int32)
            # gt_cell = cell_gt.astype(np.int32)
            # # print(output_cell)
            # print(len(output_cell))
            # # print(output_cell[5])
           
            output_cell = (output_cell > 0.5).int()
            # print(output_cell)
            # sys.exit()
            
            # label = label_test_tensor[valid_indices]
            label = label_test[valid_indices].reshape(-1)
            # print(len(label))
            # label = label.detach().cpu().numpy()
            output_cell = output_cell.detach().cpu().numpy()

            # for j in range(len(output_cell)):
            #     if output_cell[j].item() == 1:
            #         cell_chk_list.append((img_name, j, output_cell[j].item(), label[j].item(), output_cell_prob[j].item()))
                    
            for j in range(len(output_cell)):
                if output_cell[j].item() == 1:
                    cell_chk_list.append((img_name, j, output_cell[j].item(), label[j].item(), valid_out_cell[j].item()))
            # print(output_cell[0]) 
            # print(label[0]) 
            # print(valid_out_cell[0]) 
            
            # print(len(label))
            # print(label[4].item())
            # print(output_cell)
            # print(cell_chk_list)
            # sys.exit()
            
            for k in range(b):
                image_stat = {}
                image_stat['module_gt'] = label_module.tolist()[k]
                image_stat['cell_gt'] = label_test.tolist()[k]
                image_stat['module_prob'] = output_module_prob.tolist()[k]
                image_stat['cell_prob'] = output_cell_prob.tolist()[k]
                
                results[img_name[k]] = image_stat
            
            cell_len_chk_list.append((len(output_cell), len(label)))
            
        # df_fp = pd.DataFrame(fp_path)
        # df_fn = pd.DataFrame(fn_path)
        # df_true = pd.DataFrame(true_path)
        
        # df_fp.to_csv(f'{save_dir}/fp_path.csv', index=False, header=header_chk) 
        # df_fn.to_csv(f'{save_dir}/fn_path.csv', index=False, header=header_chk)
        # df_true.to_csv(f'{save_dir}/true_path.csv', index=False, header=header_chk)

        cell_chk_df = pd.DataFrame(cell_chk_list)
        cell_chk_df.to_csv(os.path.join(save_dir, 'cell_chk_result.csv'), header=header_cell, index=False)
        
        
        cell_len_chk_df = pd.DataFrame(cell_len_chk_list)
        cell_len_chk_df.to_csv(os.path.join(save_dir, 'cell_len_chk.csv'), index=False)
        
        # print('The End!')
        # sys.exit()
        
        with open(os.path.join(save_dir, f"pred_result.json"), "w") as json_file:
            json.dump(results, json_file)
    
    for _, threshold in enumerate(tqdm(threshold_list)): 
        output_cell = (cell_prob.reshape(-1) > threshold).astype(np.int32)
        output_module = (module_prob.reshape(-1) > threshold).astype(np.int32)
        module_result.append(evaluation(module_gt, output_module, 'module', threshold))
        cell_result.append(evaluation(cell_gt, output_cell, 'cell', threshold))
            
    module_result_df = pd.DataFrame(module_result)
    module_result_df.to_csv(os.path.join(save_dir, 'module_result.csv'), header=header, index=False)

    cell_result_df = pd.DataFrame(cell_result)
    cell_result_df.to_csv(os.path.join(save_dir, 'cell_result.csv'), header=header, index=False)
print('The End!')