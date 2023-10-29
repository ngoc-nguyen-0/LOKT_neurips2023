from statistics import mode
from losses import completion_network_loss, noise_loss
from utils import *
from classify import *
from generator import *
from discri import *
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from torch.autograd import grad
import torchvision.transforms as transforms
import torch
import time
import random
import os, logging
import numpy as np
from attack import inversion, dist_inversion,dist_inversion_diff
from generator import Generator,GeneratorCXR
from argparse import  ArgumentParser


import utils
import KD_students

torch.manual_seed(9)
from recovery import prepare_parser,count_parameters
if __name__ == "__main__":
    global args, logger

    parser = prepare_parser()
    
    parser.add_argument('--model2', default='', type=str, help='model')
    parser.add_argument('--ckpt2', default='', type=str, help='ckpt')
    args = parser.parse_args()

    print("=> Using improved GAN:", args.improved_flag)
    print("Loading models")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args_json = load_json(json_file=args.file_path)
    
    n_classes=args_json['dataset']['n_classes']
    z_dim = 100
    ###########################################
    ###########     load model       ##########
    ###########################################
    if args.model2 =='':
        model_types_ = args_json['train']['model_types'].split(',')
        checkpoints = args_json['train']['cls_ckpts'].split(',')
        
        cid = args.classid.split(',')
    else:
        model_types_ = []
        checkpoints = []
        cid = []
        for i in range(200,201,10):
            model_types_.append(args.model2)
            checkpoints.append(args.ckpt2+'{}.pt'.format(i))
            cid.append(int(i/10-1))
    print('model_types_',model_types_)
    print('checkpoints',checkpoints)
    
    _, dataloader_test = init_dataloader(args_json, args_json['dataset']['test_file_path'], 64, mode="test")
    _, dataloader_train = init_dataloader(args_json, args_json['dataset']['train_file_path'], 64, mode="test")
    from KD_students import test
    import csv
    csv_file = 'Classifier_acc.csv'    
    
    # test
    

    dataset = args_json['dataset']['d_priv']
    
    #target and student classifiers
    fea_mean = []
    fea_logvar = []
    targetnets_diff = []
    for i in range(len(cid)):
        id_ = int(cid[i])
        print('Load classifier {}, ckpt {}'.format(model_types_[id_], checkpoints[id_]))
        model = KD_students.get_model(model_types_[id_],n_classes,checkpoints[id_])
        model = model.to(device)
        model = model.eval()
       
        top1,top5 = test(model, dataloader=dataloader_test)
        
        top1_train,top5_train = test(model, dataloader=dataloader_train)

        fields=['{}'.format(model_types_[id_]),
            '{}'.format(checkpoints[id_]),
            '{}'.format(args_json['dataset']['d_priv']),  
            '{}'.format(count_parameters(model)), 
            '{:.2f}'.format(top1_train),
            '{:.2f}'.format(top5_train),
            '{:.2f}'.format(top1),
            '{:.2f}'.format(top5)]
        with open(csv_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

  