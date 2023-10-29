# -*- coding: utf-8 -*-
# import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import models.classifiers.evolve as evolve

# from tqdm import tqdm

from torchvision import transforms as trans



def get_model(model_name,nclass,path_T=None,dataset='celeba'):
    print('------model_name',model_name)
    if model_name=="VGG16":
        model = VGG16(nclass)   
    elif model_name=="densenet121":
        model = densenet121(nclass)  
    elif model_name=="densenet161":
        model = densenet161(nclass)  
    elif model_name=="densenet169":
        model = densenet169(nclass)  
    elif model_name=="densenet201":
        model = densenet201(nclass)  
    elif model_name=="mobilenet_v3_large":
        model = mobilenet_v3_large(nclass)  
    elif model_name=="mobilenet_v3_small":
        model = mobilenet_v3_small(nclass)  
    elif model_name=="mobilenet_v2":
        model = Mobilenet_v2(nclass)     
    elif model_name =="Resnet18":
        model = ResNet18(nclass)
    elif model_name =="Resnet50":
        model = ResNet50(nclass)
    elif model_name=="FaceNet":
        print('FaceNet')
        model = FaceNet(nclass)
    elif "FaceNet64" in model_name:
        print('FaceNet')
        model = FaceNet64(nclass)
    elif model_name=="IR152":
        model = IR152(nclass)
    elif model_name=="VGG19":
        model = VGG19(nclass)
    elif model_name=="VGG13":
        model = VGG13(nclass)
    elif model_name=="VGG11":
        model = VGG11(nclass)
    elif model_name=="VGG19_wo_bn":
        model = VGG19_wo_bn(nclass)
    elif model_name=="VGG16_wo_bn":
        model = VGG16_wo_bn(nclass)
    elif model_name=="VGG13_wo_bn":
        model = VGG13_wo_bn(nclass)
    elif model_name=="VGG11_wo_bn":
        model = VGG11_wo_bn(nclass)
           
       
    elif model_name =="Resnet34":
        model = ResNet34(nclass)  
    
    elif model_name =="efficientnet_b0":
        model = EfficientNet_b0(nclass)   
    elif model_name =="efficientnet_b1":
        model = EfficientNet_b1(nclass)   
    elif model_name =="efficientnet_b2":
        model = EfficientNet_b2(nclass)  
    elif model_name =="efficientnet_b3":
        model = EfficientNet_b3(nclass)   
    elif model_name =="efficientnet_b4":
        model = EfficientNet_b4(nclass)  
    elif model_name =="efficientnet_b7":
        model = EfficientNet_b7(nclass)  

    elif model_name =="efficientnet_v2_l":
        model = EfficientNet_v2_l(nclass,dataset)   
    elif model_name =="efficientnet_v2_m":
        model = EfficientNet_v2_m(nclass,dataset)   
    elif model_name =="efficientnet_v2_s":
        model = EfficientNet_v2_s(nclass,dataset)
    if model_name =="VGG16_defense_MI":
        model = VGG16_defense_MI(nclass,True)   
      
    model = torch.nn.DataParallel(model).cuda()
    if path_T is not None: 
        
        ckp_T = torch.load(path_T)        
        model.load_state_dict(ckp_T['state_dict'], strict=True)
    return model

######################3333

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def de_preprocess(tensor):
    return tensor*0.5 + 0.5

rs112 = trans.Compose([
        de_preprocess,
        trans.ToPILImage(),
        trans.Resize(112),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
def resize112_batch(imgs_tensor):
    device = imgs_tensor.device
    resized_imgs = torch.zeros(len(imgs_tensor), 3, 112, 112)
    for i, img_ten in enumerate(imgs_tensor):
        resized_imgs[i] = rs112(img_ten.cpu())
    return resized_imgs.to(device)

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    return torch.pow(x.unsqueeze(1).expand(n, m, d) - y.unsqueeze(0).expand(n, m, d), 2).sum(2)

hflip = trans.Compose([
    de_preprocess,
    trans.ToPILImage(),
    trans.functional.hflip,
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs



def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class CrossEntropyLoss(_Loss):
    def forward(self, out, gt, mode="reg"):
        bs = out.size(0)
        loss = - torch.mul(gt.float(), torch.log(out.float() + 1e-7))
        if mode == "dp":
            loss = torch.sum(loss, dim=1).view(-1)
        else:
            loss = torch.sum(loss) / bs
        return loss

class BinaryLoss(_Loss):
    def forward(self, out, gt):
        bs = out.size(0)
        loss = - (gt * torch.log(out.float()+1e-7) + (1-gt) * torch.log(1-out.float()+1e-7))
        loss = torch.mean(loss)
        return loss



############################################ VGG ###################################################
def make_layers(cfg, batch_norm=False):
    blocks = []
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            blocks.append(nn.Sequential(*layers))
            layers = []
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return blocks

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG19(nn.Module):
    def __init__(self, n_classes):
        super(VGG19, self).__init__()
        model = torchvision.models.vgg19_bn(pretrained=True)
        
        self.feature = model.features


        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class VGG16_defense_MI(nn.Module):
    def __init__(self, n_classes, hsic_training=False, dp_training=False, dataset='celeba'):
        super(VGG16_defense_MI, self).__init__()

        self.hsic_training = hsic_training

        if self.hsic_training:
            blocks = make_layers(cfgs['D'], batch_norm=True)
            self.layer1 = blocks[0]
            self.layer2 = blocks[1]
            self.layer3 = blocks[2]
            self.layer4 = blocks[3]
            self.layer5 = blocks[4]

        else:
            model = torchvision.models.vgg16_bn(pretrained=True)
            self.feature = model.features

        if dataset == 'celeba':
            self.feat_dim = 512 * 2 * 2
        else:
            self.feat_dim = 512
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        if not dp_training:
            self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        if self.hsic_training:
            hiddens = []

            out = self.layer1(x)
            hiddens.append(out)

            out = self.layer2(out)
            hiddens.append(out)

            out = self.layer3(out)
            hiddens.append(out)

            out = self.layer4(out)
            hiddens.append(out)

            feature = self.layer5(out)
            feature = feature.view(feature.size(0), -1)
            feature = self.bn(feature)

            hiddens.append(feature)

            res = self.fc_layer(feature)
            #comment here
            # return hiddens, res
            
            return feature, res

        else:
            feature = self.feature(x)
            feature = feature.view(feature.size(0), -1)
            feature = self.bn(feature)

            res = self.fc_layer(feature)

            return feature, res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return out

class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out


class VGG16_vib(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_vib, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.k = self.feat_dim // 2
        self.n_classes = n_classes
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Linear(self.k, self.n_classes)
            
    def forward(self, x, mode="train"):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
       
        return [feature, out, mu, std]
    
    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
       
        return out

class VGG13(nn.Module):
    def __init__(self, n_classes):
        super(VGG13, self).__init__()
        model = torchvision.models.vgg13_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class VGG13_wo_bn(nn.Module):
    def __init__(self, n_classes):
        super(VGG13_wo_bn, self).__init__()
        model = torchvision.models.vgg13(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        # self.bn = nn.BatchNorm1d(self.feat_dim)
        # self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        # feature = self.bn(feature)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class VGG11(nn.Module):
    def __init__(self, n_classes):
        super(VGG11, self).__init__()
        model = torchvision.models.vgg11_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class VGG11_wo_bn(nn.Module):
    def __init__(self, n_classes):
        super(VGG11_wo_bn, self).__init__()
        model = torchvision.models.vgg11(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        # self.bn = nn.BatchNorm1d(self.feat_dim)
        # self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        # feature = self.bn(feature)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        # feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class VGG19_wo_bn(nn.Module):
    def __init__(self, n_classes):
        super(VGG19_wo_bn,  self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        
        self.feature = model.features


        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        # self.bn = nn.BatchNorm1d(self.feat_dim)
        # self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        # feature = self.bn(feature)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class VGG16_wo_bn(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_wo_bn,  self).__init__()
        model = torchvision.models.vgg16(pretrained=True)
        
        self.feature = model.features


        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        # self.bn = nn.BatchNorm1d(self.feat_dim)
        # self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        # feature = self.bn(feature)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out


############EfficientNet

class EfficientNet_b0(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b0, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b0(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        # print(model)
        # print('self.feature',self.feature)
        # exit()
        self.n_classes = n_classes
        self.feat_dim = 1280#5120#
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_b1(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b1, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b1(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_b2(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b2, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b2(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1408#5632#
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_b3(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b3, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b3(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1536
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_b4(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b4, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b4(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1792
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_b7(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b7, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b7(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 2560#10240#
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_v2_s(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_v2_s, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_s(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)

        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_v2_m(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_v2_m, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_m(pretrained=True)

        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_v2_l(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_v2_l, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_l(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        # self.feature = model.features
        self.n_classes = n_classes
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out


###########3 facenet
class EvolveFace(nn.Module):
    def __init__(self, num_of_classes, IR152):
        super(EvolveFace, self).__init__()
        if IR152:
            model = evolve.IR_152_64((64,64))
            # ckp_model = torch.load("./model_weights/eval_evolve/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth")
        else:
            model = evolve.IR_50_64((64,64))
            # load model weights
            # ckp_model = torch.load("./model_weights/eval_evolve/backbone_ir50_ms1m_epoch120.pth")
        # utils.load_my_state_dict(model, ckp_model)
        self.model = model
        self.feat_dim = 512
        self.num_classes = num_of_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(p=0.15),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))

        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_classes),)


    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self,x):
        #feature = self.feature(x)
        feature = self.model(x)
        feature = self.output_layer(feature)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)

        return  out

class FaceNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(FaceNet, self).__init__()
        self.feature = evolve.IR_50_112((112, 112))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def predict(self, x):
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out
            
    def forward(self, x):
        # print("input shape:", x.shape)
        # import pdb; pdb.set_trace()
        
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return [feat, out]

class FaceNet64(nn.Module):
    def __init__(self, num_classes = 1000):
        super(FaceNet64, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)
        return feat, out


####### resnet
class IR152(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return feat,out

class IR152_vib(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152_vib, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.k = self.feat_dim // 2
        self.n_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim = 1))

    def forward(self, x):
        feature = self.output_layer(self.feature(x))
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feature, out, iden, mu, st

class IR50(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR50, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim = 1))

    def forward(self, x):
        feature = self.output_layer(self.feature(x))
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feature, out, iden, mu, std

class IR50_vib(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR50_vib, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.n_classes = num_classes
        self.k = self.feat_dim // 2
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        feat = self.output_layer(self.feature(x))
        feat = feat.view(feat.size(0), -1)
        statis = self.st_layer(feat)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feat, out, iden, mu, std


class ResNet18(nn.Module):
    def __init__(self, num_of_classes):
        super(ResNet18, self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])
        # print(self.feature)
        self.feat_dim = 2048 * 1 * 1
        self.num_of_classes = num_of_classes
        self.fc_layer = nn.Linear(self.feat_dim, self.num_of_classes)
            # nn.Linear(2048, self.num_of_classes),)
            # nn.Sigmoid())


    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        # print(feature.shape)
        feature = feature.view(feature.size(0), -1)
        # print('feature',feature.shape)
        out, iden = self.classifier(feature)
        return feature, out

class ResNet34(nn.Module):
    def __init__(self, num_of_classes):
        super(ResNet34, self).__init__()
        model = torchvision.models.resnet34(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])
        # print(self.feature)
        self.feat_dim = 2048 * 1 * 1
        self.num_of_classes = num_of_classes
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(self.feat_dim, self.num_of_classes),)
            # nn.Linear(2048, self.num_of_classes),)
            # nn.Sigmoid())
        
        self.fc_layer = nn.Linear(self.feat_dim, self.num_of_classes)

    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        # print(feature.shape)
        feature = feature.view(feature.size(0), -1)
        # print('feature',feature.shape)
        out, iden = self.classifier(feature)
        return  feature, out

# class ResNet50(nn.Module):
#     def __init__(self, num_of_classes):
#         super(ResNet50, self).__init__()
#         model = torchvision.models.resnet50(pretrained=True)
#         self.feature = nn.Sequential(*list(model.children())[:-2])
#         # print(self.feature)
#         self.feat_dim = 8192 * 1 * 1
#         self.num_of_classes = num_of_classes
#         # self.fc_layer = nn.Sequential(
#         #     nn.Linear(self.feat_dim, self.num_of_classes),)
#             # nn.Linear(2048, self.num_of_classes),)
#             # nn.Sigmoid())
        
#         self.fc_layer = nn.Linear(self.feat_dim, self.num_of_classes)

#     def classifier(self, x):
#         out = self.fc_layer(x)
#         __, iden = torch.max(out, dim = 1)
#         iden = iden.view(-1, 1)
#         return out, iden

#     def forward(self, x):
#         feature = self.feature(x)
#         # print(feature.shape)
#         feature = feature.view(feature.size(0), -1)
#         # print('feature',feature.shape)
#         out, iden = self.classifier(feature)
#         return  feature, out


###densenet121


class densenet121(nn.Module):
    def __init__(self, n_classes):
        super(densenet121, self).__init__()
        model = torchvision.models.densenet.densenet121(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])

        # print('---self.feature ',self.feature )
        self.n_classes = n_classes
        self.feat_dim = 4096
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class densenet161(nn.Module):
    def __init__(self, n_classes):
        super(densenet161, self).__init__()
        model = torchvision.models.densenet.densenet161(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        # print('---self.feature ',self.feature )
        self.n_classes = n_classes
        self.feat_dim = 8832
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class densenet169(nn.Module):
    def __init__(self, n_classes):
        super(densenet169, self).__init__()
        model = torchvision.models.densenet.densenet169(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        # print('---self.feature ',self.feature )
        self.n_classes = n_classes
        self.feat_dim = 6656
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class densenet201(nn.Module):
    def __init__(self, n_classes):
        super(densenet201, self).__init__()
        model = torchvision.models.densenet.densenet201(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        # print('---self.feature ',self.feature )
        self.n_classes = n_classes
        self.feat_dim = 7680
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out


############3 mobilenet
class mobilenet_v3_large(nn.Module):
    def __init__(self, n_classes):
        super(mobilenet_v3_large, self).__init__()
        model = torchvision.models.mobilenetv3.mobilenet_v3_large(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        # print('---self.feature ',self.feature )
        self.n_classes = n_classes
        self.feat_dim = 960
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class mobilenet_v3_small(nn.Module):
    def __init__(self, n_classes):
        super(mobilenet_v3_small, self).__init__()
        model = torchvision.models.mobilenetv3.mobilenet_v3_small(pretrained=True)
        # print('model',model)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        # print('---self.feature ',self.feature )
        self.n_classes = n_classes
        self.feat_dim = 576
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out


class Mobilenet_v2(nn.Module):
    def __init__(self, num_of_classes):
        super(Mobilenet_v2, self).__init__()
        model = torchvision.models.mobilenet_v2(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])
        # print(self.feature)
        self.feat_dim = 12288
        self.num_of_classes = num_of_classes
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_of_classes),)
            # nn.Linear(2048, self.num_of_classes),)
            # nn.Sigmoid())


    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        # print(feature.shape)
        feature = feature.view(feature.size(0), -1)
        # print('feature',feature.shape)
        out, iden = self.classifier(feature)
        return   feature,out


# from models.classifiers.torchvision.models.convnext import convnext_small, convnext_tiny
# from torchvision.models.convnext import convnext_small, convnext_tiny
# class Convnext_small(nn.Module):
#     def __init__(self, n_classes):
#         super(Convnext_small, self).__init__()
#         model = convnext_small(weights='IMAGENET1K_V1')
#         self.feature = nn.Sequential(*list(model.children())[:-1])
#         self.n_classes = n_classes
#         self.feat_dim = 768
#         self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
#     def forward(self, x):
#         feature = self.feature(x)
#         feature = feature.view(feature.size(0), -1)
#         res = self.fc_layer(feature)
#         return  feature,res

#     def predict(self, x):
#         feature = self.feature(x)
#         feature = feature.view(feature.size(0), -1)
#         res = self.fc_layer(feature)
#         out = F.softmax(res, dim=1)

#         return feature,res


# class Convnext_tiny(nn.Module):
#     def __init__(self, n_classes):
#         super(Convnext_tiny, self).__init__()
#         model = convnext_tiny(weights='IMAGENET1K_V1')
#         self.feature = nn.Sequential(*list(model.children())[:-1])
#         self.n_classes = n_classes
#         self.feat_dim = 768
#         self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
#     def forward(self, x):
#         feature = self.feature(x)
#         feature = feature.view(feature.size(0), -1)
#         res = self.fc_layer(feature)
#         return  feature,res

#     def predict(self, x):
#         feature = self.feature(x)
#         feature = feature.view(feature.size(0), -1)
#         res = self.fc_layer(feature)
#         out = F.softmax(res, dim=1)

#         return feature,res







class ResNet50(nn.Module):
    def __init__(self, num_of_classes):
        super(ResNet50, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim =2048
        self.num_of_classes = num_of_classes        
        self.fc_layer = nn.Linear(self.feat_dim, self.num_of_classes)

    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        # print(feature.shape)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return  feature, out

