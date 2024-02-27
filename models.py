import torch
import torch.nn as nn
import torch.nn.functional as Func
import torchvision.models as models

from copy import deepcopy
from util import *
import timm
import math

class ResNet50_cell_maxprob(nn.Module):
    def __init__(self):
        super(ResNet50_cell_maxprob, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # self.downsample = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False),
        #     nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        # )
        # self.conv_block1 = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        #     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # )
        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        #     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # )

        # resnet 첫번째 conv1의 입력 채널=1로 변경, fc layer 변경 #
        self.conv1 = deepcopy(resnet.conv1)
        self.conv2 = deepcopy(resnet.conv1)
        self.conv1 = nn.Conv2d(1, 64, 7, 2)
        
        self.down_sampling1 = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.down_sampling2 = nn.Sequential(
            self.conv2,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.down_sampling = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2),
            nn.Conv2d(64, 64, 3, 2)
        )

        self.layer1 = deepcopy(resnet.layer1)
        self.layer2 = deepcopy(resnet.layer2)
        self.layer3 = deepcopy(resnet.layer3)
        self.layer4 = deepcopy(resnet.layer4)

        self.avgpool = deepcopy(resnet.avgpool)
        self.resnet.fc = nn.Linear(2048, 1, bias=True)

    def forward(self, x):
        batch_size, num_cells, c, w, h = x.size()

        x = x.view(batch_size*num_cells, c, w, h)
        x = self.down_sampling1(x)
        x = self.down_sampling2(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)
        x = x.view(batch_size*num_cells, -1)
        x = self.fc(x) # (batch_size*num_cells, 1)
        x = x.view(batch_size, num_cells)
        x = self.softmax(x)
        value, indices = torch.max(x, dim=1) # batch_size
    
        return value
    
class ResNet50_cell_avgprob(nn.Module):
    def __init__(self):
        super(ResNet50_cell_avgprob, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        )

        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.softmax = nn.Softmax()
        
        self.fc = nn.Linear(in_features=2048, out_features=1, bias=True)

    def forward(self, x):
        batch_size, num_cells, c, w, h = x.size()

        x = x.view(batch_size*num_cells, c, w, h)
        x = self.downsample(x) # (batch_size*num_cells, 64, 300/4=75, 600/4=150)
        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(batch_size*num_cells, -1)
        x = self.fc(x) # (batch_size*num_cells, 1)
        x = x.view(batch_size, num_cells)
        x = self.softmax(x)
        x = torch.mean(x, dim=1) # batch_size
    
        return x
    
class ResNet50_module_maxprob(nn.Module):
    def __init__(self):
        super(ResNet50_module_maxprob, self).__init__()
        # self.resnet = models.resnet50(pretrained=True)
        resnet = models.resnet50(pretrained=True)

        # resnet 첫번째 conv1의 입력 채널=1로 변경, fc layer 변경 #
        resnet.conv1 = nn.Conv2d(64, 64, 7, 2)

        self.down_sampling = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2),
            nn.Conv2d(64, 64, 7, 2)
        )
        
        self.conv_block = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = deepcopy(resnet.layer1)
        self.layer2 = deepcopy(resnet.layer2)
        self.layer3 = deepcopy(resnet.layer3)
        self.layer4 = deepcopy(resnet.layer4)

        self.avgpool = deepcopy(resnet.avgpool)
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.fc = deepcopy(resnet.fc)
        self.fc = nn.Linear(2048, 1, bias=True)

    def forward(self, x):
        b, c, w, h = x.size()
        x = self.down_sampling(x)

        x = self.conv_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)
        x = x.view(b, -1)
        x = self.fc(x) # (batch_size, 1)
        x = self.sigmoid(x)
        # x = self.softmax(x)
        value, indices = torch.max(x, dim=1) # batch_size
    
        return value

class ResNet50_module(nn.Module):
    def __init__(self):
        super(ResNet50_module, self).__init__()
        # self.resnet = models.resnet50(pretrained=True)
        resnet = models.resnet50(pretrained=True)

        # resnet 첫번째 conv1의 입력 채널=1로 변경, fc layer 변경 #
        resnet.conv1 = nn.Conv2d(1, 64, 7, 2)

        self.conv_block = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = deepcopy(resnet.layer1)
        self.layer2 = deepcopy(resnet.layer2)
        self.layer3 = deepcopy(resnet.layer3)
        self.layer4 = deepcopy(resnet.layer4)

        self.avgpool = deepcopy(resnet.avgpool)
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.fc = deepcopy(resnet.fc)
        self.fc = nn.Linear(2048, 1, bias=True)

    def forward(self, x):
        b, c, w, h = x.size()

        x = self.conv_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)
        x = x.view(b, -1)
        x = self.fc(x) # (batch_size, 1)
        x = self.sigmoid(x)
        # x = self.softmax(x)
        value, indices = torch.max(x, dim=1) # batch_size
    
        return value
    
# Resnet, VGG, CNN 그대로 불러오기
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Linear(2048, 1)
        self.net = resnet
    def forward(self, x):

        x = self.net(x)

        return x # (b, 1) 

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        resnet34 = models.resnet34(pretrained=True)
        in_features = resnet34.fc.in_features
        resnet34.fc = nn.Linear(in_features, 1)
        self.net = resnet34
    def forward(self, x):

        x = self.net(x)

        return x # (b, 1) 

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        in_features = resnet18.fc.in_features
        resnet18.fc = nn.Linear(in_features, 1)

        self.net = resnet18
    def forward(self, x):

        x = self.net(x)

        return x # (b, 1) 

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=True)
        in_features = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(in_features, 1)

        self.net = vgg16

    def forward(self, x):

        x = self.net(x)

        return x # (b, 1)
 
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg19 = models.vgg19_bn(pretrained=True)
        vgg19.classifier[6] = nn.Linear(4096, 1)

        self.net = vgg19

    def forward(self, x):

        x = self.net(x)

        return x # (b, 1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2)
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2)
                                    )
        self.fc1 = nn.Linear(7152128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        b, c, w, h = x.size()

        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(b, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x # (b, 1) 

class EfficientNetB2(nn.Module):
    def __init__(self):
        super(EfficientNetB2, self).__init__()

        effiB2 = models.efficientnet_b2(pretrained=True)
        in_features = effiB2.classifier[1].in_features
        effiB2.classifier[1] = nn.Linear(in_features, 1)

        self.net = effiB2
    
    def forward(self, x):

        x = self.net(x)

        return x
    
class EfficientNetB3(nn.Module):
    def __init__(self):
        super(EfficientNetB3, self).__init__()

        effiB3 = models.efficientnet_b3(pretrained=True)
        in_features = effiB3.classifier[1].in_features
        effiB3.classifier[1] = nn.Linear(in_features, 1)

        self.net = effiB3
    
    def forward(self, x):

        x = self.net(x)

        return x

class EfficientNetB4(nn.Module):
    def __init__(self):
        super(EfficientNetB4, self).__init__()

        effiB4 = models.efficientnet_b4(pretrained=True)
        in_features = effiB4.classifier[1].in_features
        effiB4.classifier[1] = nn.Linear(in_features, 1)

        self.net = effiB4
    
    def forward(self, x):
        x = self.net(x)

        return x
    
class Ensemble_concat(nn.Module):
    def __init__(self, path1, path2, path3):
        super(Ensemble_concat, self).__init__()
        
        # model load
        net1 = timm.create_model('resnext50d_32x4d.bt_in1k', pretrained=True, num_classes = 1)
        net2 = timm.create_model("resnet34d", pretrained=True, num_classes = 1)
        net3 = timm.create_model('seresnext50_32x4d.racm_in1k', pretrained=True, num_classes = 1)
        
        net1.load_state_dict(torch.load(path1))
        net2.load_state_dict(torch.load(path2))
        net3.load_state_dict(torch.load(path3))
          
        self.model1 = torch.nn.Sequential(*(list(net1.children())[:-1]))
        self.model2 = torch.nn.Sequential(*(list(net2.children())[:-1]))
        self.model3 = torch.nn.Sequential(*(list(net3.children())[:-1]))
        
        # fix parameters
        fix_parameters(self.model1)
        fix_parameters(self.model2)
        fix_parameters(self.model3)
        
        # for layer in self.model1.children():
        #     for param in layer.parameters():
        #         print(param.requires_grad)
        
        # for layer in self.model2.children(): 
        #     for param in layer.parameters():
        #         print(param.requires_grad)
        
        # for layer in self.model3.children():
        #     for param in layer.parameters():
        #         print(param.requires_grad)

            
        self.fc1 = nn.Linear(4608, 2304) # 4608 = 2048 + 512 + 2048
        self.fc2 = nn.Linear(2304, 1)
        
    def forward(self, x):
        feature1 = self.model1(x) # (b, 2048)
        feature2 = self.model2(x) # (b, 512)
        feature3 = self.model3(x) # (b, 2048)
        
        con_feature = torch.cat((feature1, feature2, feature3), dim=1) # (b, 2048 + 512 + 2048)
        
        x = self.fc1(con_feature) # (b, 1)
        x = self.fc2(x)
        
        return x

class Ensemble_concat2(nn.Module):
    def __init__(self, path1, path2, path3, path4, path5):
        super(Ensemble_concat2, self).__init__()
        
        # model load
        net1 = timm.create_model('resnext50d_32x4d.bt_in1k', pretrained=True, num_classes = 1)
        net2 = timm.create_model("resnet34d", pretrained=True, num_classes = 1)
        net3 = timm.create_model('seresnext50_32x4d.racm_in1k', pretrained=True, num_classes = 1)
        net4 = ResNet34()
        net5 = ResNet50()
        
        net1.load_state_dict(torch.load(path1))
        net2.load_state_dict(torch.load(path2))
        net3.load_state_dict(torch.load(path3))
        net4.load_state_dict(torch.load(path4))
        net5.load_state_dict(torch.load(path5))
        
        # fix parameters
        fix_parameters(net1)
        fix_parameters(net2)
        fix_parameters(net3)
        fix_parameters(net4)
        fix_parameters(net5)
        
        self.model1 = torch.nn.Sequential(*(list(net1.children())[:-1]))
        self.model2 = torch.nn.Sequential(*(list(net2.children())[:-1]))
        self.model3 = torch.nn.Sequential(*(list(net3.children())[:-1]))
        self.model4 = torch.nn.Sequential(*(list(net4.children())[:-1]))
        self.model5 = torch.nn.Sequential(*(list(net5.children())[:-1]))
        
        self.fc = nn.Linear(4045203, 1) 
        
    def forward(self, x):
        b, _, _, _ = x.size()
        
        feature1 = self.model1(x) # (b, 2048)
        feature2 = self.model2(x) # (b, 512)
        feature3 = self.model3(x) # (b, 2048)
        
        feature4 = self.model4(x) # (b, 1345329)
        feature4 = feature4.view(b, -1)
        # print(feature4.shape)
        feature5 = self.model5(x) # (b, 2695266)
        feature5 = feature5.view(b, -1)
        # print(feature5.shape)
        
        con_feature = torch.cat((feature1, feature2, feature3, feature4, feature5), dim=1) # (b, 2048 + 512 + 2048)
        # print(con_feature.shape)
        x = self.fc(con_feature) # (b, 1)
        
        return x

class ResNet34_att(nn.Module):
    def __init__(self):
        super(ResNet34_att, self).__init__()
        
        self.emb_size = 512
        
        net = timm.create_model('resnet34.a1_in1k', pretrained=True, num_classes = 1)
        self.model = torch.nn.Sequential(*(list(net.children())[:-1]))
        
        self.Q_layer = nn.Linear(self.emb_size, self.emb_size)
        self.K_layer = nn.Linear(self.emb_size, self.emb_size)
        self.V_layer = nn.Linear(self.emb_size, self.emb_size)
        
        self.fc_layer = nn.Linear(self.emb_size, 1)
        
    def forward(self, x):
        x = self.model(x) # (cells, emb_size)
        out1 = self.fc_layer(x) # (cells, 1)
        
        Q = self.Q_layer(x) # (cells, emb_size)
        K = self.K_layer(x) # (cells, emb_size)
        V = self.V_layer(x) # (cells, emb_size)
       
        # attention score
        energy = torch.matmul(Q, K.T) # (cells, cells)
        energy = energy / math.sqrt(self.emb_size) # scaling
        energy = Func.softmax(energy, dim=-1)
        
        out2 = torch.matmul(energy, V) # (cells, emb_size)
        out2 = self.fc_layer(out2) # (cells, 1)
        
        return out1, out2