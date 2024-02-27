from tqdm import tqdm
import torch
import torch.nn as nn
 
from util import *

# training
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    
    print('==================== Training ====================')
    loss_list = []
    
    for i, data in enumerate(tqdm(trainloader, 0)):
        # forward pass
        input, label = data['img_tensor'].to(device), data['label'].to(device)
        output = model(input) 
        
        output = output.squeeze(-1)
        label = label.to(torch.float32) # batch_size
        
        # backward pass
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        # computate the losses
        loss_list += [loss.item()]
    
    return loss_list

def train_cell(model, trainloader, optimizer, criterion, device):
    model.train()
    sigmoid = nn.Sigmoid()
    
    print('==================== Training ====================')
    loss_list = []
    probs = []
    
    for i, data in enumerate(tqdm(trainloader, 0)):
        # forward pass
        input, label = data['img_tensor'].to(device), data['label'].to(device)
        
        b, cells, c, h, w = input.shape
        input = input.view(b*cells, c, h, w) # (156, 3, 600, 300)
        # print(input.shape)
        output = model(input) # (cells, 1)
        # print(output.shape)
        
        # probability
        output = sigmoid(output)
        
        # max
        output = torch.max(output)
        
        # avg
        # output = torch.mean(output)   
        
        output = output.unsqueeze(-1) # batch_size
        label = label.to(torch.float32) # batch_size
        
        # backward pass
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        # computate the losses
        loss_list += [loss.item()]
    
    return loss_list

def train_cell_cont(model, trainloader, optimizer, criterion, device):
    model.train()
    sigmoid = nn.Sigmoid()
    
    print('==================== Training ====================')
    loss_list = []
    probs = []
    
    for i, data in enumerate(tqdm(trainloader, 0)):
        # forward pass
        batch_input, label = data['img_tensor'].to(device), data['label'].to(device)
        b, cells, c, h, w = batch_input.shape # batch = 2로 고정
        
        # contrastive learning
        input1 = batch_input[0]
        input2 = batch_input[1]
        
        output1 = model(input1)
        output2 = model(input2)

        optimizer.zero_grad()
        
        loss1 = criterion(output1, output2) # contrastive loss
        print(loss1)
        input = batch_input.view(b*cells, c, h, w) # (2*156, 3, 600, 300)
        output = model(input) # (2*cells, 1)
        
        if output.size() != (b * cells, 1):
            raise ValueError("Invalid size of output tensor. Expected size: (batchsize * cells, 1)")

        # probability
        output = sigmoid(output)
        
        # max
        output= output.view(b, cells, 1)
        output, idx = torch.max(output, dim=1) # output : (b, 1)
        
        # avg
        # output = torch.mean(output)   
        
        output = output.unsqueeze(-1) # batch_size
        label = label.to(torch.float32) # batch_size
        
        loss2 = criterion(output, label) # classification loss
        
        total_loss = loss1 + loss2
        print(total_loss)
        total_loss.backward()
        optimizer.step()
        
        # computate the losses
        loss_list += [total_loss.item()]
    
    return loss_list

def train_mixup(model, trainloader, optimizer, criterion, device):
    model.train()
    sigmoid = nn.Sigmoid()
    
    print('==================== Training ====================')
    loss_list = []
    probs = []
    
    for i, data in enumerate(tqdm(trainloader, 0)):
        # forward pass
        input, label_a, label_b, lam = mixup_data(data['img_tensor'], data['label'], alpha=0.2)
        input, label_a, label_b = input.to(device), label_a.to(device), label_b.to(device)
        
        output = model(input) #(144,1)
        
        output = output.squeeze(-1) # batch_size
        label_a = label_a.to(torch.float32) # batch_size
        label_b = label_b.to(torch.float32) # batch_size
        
        # backward pass
        optimizer.zero_grad()
        loss = lam * criterion(output, label_a) + (1 - lam) * criterion(output, label_b)
        loss.backward()
        optimizer.step()
        
        # computate the losses
        loss_list += [loss.item()]
    
    return loss_list

# loss 추가
def train_cell_loss(model, trainloader, optimizer, criterion1, criterion2, device):
    model.train()
    sigmoid = nn.Sigmoid()
    
    print('==================== Training ====================')
    loss_list = []
    
    for i, data in enumerate(tqdm(trainloader, 0)):
        # forward pass
        input, label = data['img_tensor'].to(device), data['label'].to(device)
        
        b, cells, c, h, w = input.shape
        input = input.view(b*cells, c, h, w) # (156, 3, 600, 300)
        
        output = model(input) # (cells, 1)
        output = sigmoid(output) # (cells, 1)
        label = label.to(torch.float32) # batch_size
        
        # backward pass
        
        
        loss1 = 0.0
        for i in range(b*cells):
            cell_loss = criterion1(output[i], label) # BCE loss
            loss1 += cell_loss
        loss1 /= cells
        # print(loss1)
        
        output = torch.max(output) 
        output = output.unsqueeze(-1) # batch_size
       
        loss2 = criterion2(output, label) # BCE loss
        # print(loss2)
        
        # total_loss = 0.01 * loss1 + loss2 
        total_loss = loss1 + loss2
        # print(total_loss)
        
        total_loss.backward()
        optimizer.step()
        
        # computate the losses
        loss_list += [total_loss.item()]
    
    return loss_list

def train_cell_loss_non(model, trainloader, optimizer, criterion1, criterion2, device):
    model.train()
    sigmoid = nn.Sigmoid()
    
    print('==================== Training ====================')
    loss_list = []
    
    for i, data in enumerate(tqdm(trainloader, 0)):
        # forward pass
        input, label = data['img_tensor'].to(device), data['label'].to(device)
        
        b, cells, c, h, w = input.shape
        input = input.view(b*cells, c, h, w) # (cells, 3, 600, 300)
        
        output1, output2 = model(input) # (cells, 1)
    
        output1 = sigmoid(output1) # (cells, 1)   
        output2 = sigmoid(output2) # (cells, 1)
        label = label.to(torch.float32) # batch_size
        # print(label)
        
        # backward pass
        optimizer.zero_grad()
        
        loss1 = 0.0
        cnt = 0        
       
        if label == 0.0: # 정상 모듈
            for i in range(b*cells):
                cell_loss = criterion1(output1[i], label) # BCE loss
                loss1 += cell_loss
                cnt += 1
            loss1 /= cnt
            # loss1 /= cells
        # print(loss1)
        
        output2 = torch.max(output2)
        output2 = output2.unsqueeze(-1) # batch_size
        
        loss2 = criterion2(output2, label) # BCE loss
        # print(loss2)
        
        # total_loss = 0.01 * loss1 + loss2 
        total_loss = loss1 + loss2
        # print(total_loss)
        
        total_loss.backward()
        optimizer.step()
        
        # computate the losses
        loss_list += [total_loss.item()]
    
    return loss_list

# validation
def validate(model, validloader, criterion, device):
    model.eval()
    sigmoid = nn.Sigmoid()
    
    print('==================== Validation ====================')
    loss_list = []
    y_pred, y_true = [], []
    
    for i, data in enumerate(tqdm(validloader, 0)):
        # forward pass
        input, label = data['img_tensor'].to(device), data['label'].to(device)
        
        output = model(input)
        
        output = output.squeeze(-1)
        label = label.to(torch.float32)
        
        loss = criterion(output, label)
        loss_list += [loss.item()]
        
        threshold = 0.5
        output = sigmoid(output) 
        output = (output > threshold).int()
    
        label = label.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        
        y_true.extend(label)
        y_pred.extend(output)
    
    return y_true, y_pred, loss_list

# validation
def validate_cell(model, validloader, criterion, device):
    model.eval()
    sigmoid = nn.Sigmoid()
    
    print('==================== Validation ====================')
    loss_list = []
    y_pred, y_true = [], []
    
    for i, data in enumerate(tqdm(validloader, 0)):
        # forward pass
        input, label = data['img_tensor'].to(device), data['label'].to(device)
        b, cells, c, h, w = input.shape
        input = input.view(b*cells, c, h, w)
        
        output = model(input)
        
        output = sigmoid(output)
        
        # max
        output = torch.max(output)   
        # mean  
        # output = torch.mean(output)   
        
        output = output.unsqueeze(-1)
        label = label.to(torch.float32)
        
        loss = criterion(output, label)
        loss_list += [loss.item()]
        
        threshold = 0.5
        output = (output > threshold).int()
        
        label = label.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        
        y_true.extend(label)
        y_pred.extend(output)
    
    return y_true, y_pred, loss_list

# loss 추가
def validate_cell_loss(model, validloader, criterion1, criterion2, device):
    model.eval()
    sigmoid = nn.Sigmoid()
    
    print('==================== Validation ====================')
    loss_list = []
    y_pred, y_true = [], []
    
    for i, data in enumerate(tqdm(validloader, 0)):
        # forward pass
        input, label = data['img_tensor'].to(device), data['label'].to(device)
        b, cells, c, h, w = input.shape
        input = input.view(b*cells, c, h, w) # (156, 3, 600, 300)
        
        output = model(input) # (cells, 1)
        output = sigmoid(output) # (cells, 1)
        label = label.to(torch.float32) # batch_size
        
        loss1 = 0.0
        for i in range(b*cells):
            cell_loss = criterion1(output[i], label) # BCE loss
            loss1 += cell_loss
        loss1 /= cells
        
        output = torch.max(output) 
        output = output.unsqueeze(-1) # batch_size
       
        loss2 = criterion2(output, label) # BCE loss
        
        total_loss = loss1 + loss2

        # computate the losses
        loss_list += [total_loss.item()]
        
        threshold = 0.5
        output = (output > threshold).int()
        
        label = label.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        
        y_true.extend(label)
        y_pred.extend(output)
    
    return y_true, y_pred, loss_list

def validate_cell_loss_non(model, validloader, criterion1, criterion2, device):
    model.eval()
    sigmoid = nn.Sigmoid()
    
    print('==================== Validation ====================')
    loss_list = []
    y_pred, y_true = [], []
    
    for i, data in enumerate(tqdm(validloader, 0)):
        # forward pass
        input, label = data['img_tensor'].to(device), data['label'].to(device)
        b, cells, c, h, w = input.shape
        input = input.view(b*cells, c, h, w) # (156, 3, 600, 300)
        
        output1, output2 = model(input) # (cells, 1)
        
        output1 = sigmoid(output1) # (cells, 1)   
        output2 = sigmoid(output2) # (cells, 1)
        label = label.to(torch.float32) # batch_size
        
        loss1 = 0.0
        cnt = 0
        if label == 0.0: # 정상 모듈
            for i in range(b*cells):
                cell_loss = criterion1(output1[i], label) # BCE loss
                loss1 += cell_loss
                cnt += 1
            # loss1 /= cnt
            loss1 /= cells
        
        output2 = torch.max(output2) 
        output2 = output2.unsqueeze(-1) # batch_size
       
        loss2 = criterion2(output2, label) # BCE loss
        
        total_loss = loss1 + loss2

        # computate the losses
        loss_list += [total_loss.item()]
        
        threshold = 0.5
        output2 = (output2 > threshold).int()
        
        label = label.detach().cpu().numpy()
        output2 = output2.detach().cpu().numpy()
        
        y_true.extend(label)
        y_pred.extend(output2)
    
    return y_true, y_pred, loss_list

def validate_mixup(model, validloader, criterion, device):
    model.eval()
    sigmoid = nn.Sigmoid()
    
    print('==================== Validation ====================')
    loss_list = []
    y_pred, y_true = [], []
    
    for i, data in enumerate(tqdm(validloader, 0)):
        # forward pass
        input, label_a, label_b, lam = mixup_data(data['img_tensor'], data['label'], alpha=0.2)
        input, label_a, label_b = input.to(device), label_a.to(device), label_b.to(device)
        label = data['label'].to(device)
        
        output = model(input)
        
        output = output.squeeze(-1) # batch_size
        label_a = label_a.to(torch.float32)
        label_b = label_b.to(torch.float32) # batch_size
        label = label.to(torch.float32) # batch_size
        
        loss = lam * criterion(output, label_a) + (1 - lam) * criterion(output, label_b)
        loss_list += [loss.item()]
        
        threshold = 0.5
        output = sigmoid(output) 
        output = (output > threshold).int()
    
        label = label.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        
        y_true.extend(label)
        y_pred.extend(output)
    
    return y_true, y_pred, loss_list

# test
def inference(model, testloader, device):
    model.eval()
    sigmoid = nn.Sigmoid()
    
    print('==================== Inference ====================')
    y_pred, y_true = [], []
        
    for i, data in enumerate(tqdm(testloader, 0)):
        # forward pass
        input, label = data['img_tensor'].to(device), data['label'].to(device)
        output = model(input)
        output = sigmoid(output)
        
        output = output.squeeze(-1)
        label = label.to(torch.float32)
        
        threshold = 0.5
        output = (output > threshold).int()
        
        label = label.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        
        y_true.extend(label)
        y_pred.extend(output)
        
    return y_true, y_pred
