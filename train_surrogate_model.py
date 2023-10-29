
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import torchvision
import os
import argparse
import sys
sys.path.append('./')

import wandb

wandb_record = None

from models import *

from utils import init_dataloader,load_json

from dataset import generated_dataset


from dataset import  InfiniteSamplerWrapper
from models.classifiers.classifier import get_model
from utils import load_json

from engine import test_acc_loss

device = torch.device("cuda")

import loader
import kornia

def get_model_accuracy(net, test_loader_):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader_):
            inputs, targets = inputs.to(device), targets.to(device)
            _,outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    acc = 100.*correct/total
    return acc

def decision(inputs,net):
    
    _,teacher_outputs = net(inputs)
    _, teacher_predicted = teacher_outputs.max(1)
    return teacher_predicted

def get_wandb(is_wandb, project_id,project='surrogate_model_tacgan_same_arch',exp_dir='.'):
    global wandb_record
    
    wandb_record = None
    if is_wandb == True:
        wandb_record = wandb.init(project=project, id=project_id,save_code=True,dir=exp_dir)
    return wandb_record

def train_generated_images(epoch,Surrogate_net,netG,optimizer,z_loader,args):

    global wandb_record

    # data augmentation module in stage-1 for the generated images
    aug_list = kornia.augmentation.container.ImageSequential(
        kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomRotation(5),
    )

    Surrogate_net.train()
    
    criterion = nn.CrossEntropyLoss()
    N_batch = int(30000/args.batch_size)
    # print('Iter: {}'.format( epoch*N_batch))
    for batch_idx in range(N_batch):
        # print('batch_idx',batch_idx)

        #create data
        with torch.no_grad():
            inputs_z, targets, gen_pseudo_y = next(z_loader)
            inputs_z = inputs_z.cuda()
            gen_pseudo_y = gen_pseudo_y.cuda()
            inputs = netG(inputs_z,gen_pseudo_y)            
            inputs_da = aug_list(inputs)          

            
        optimizer.zero_grad()

        _,Surrogate_outputs = Surrogate_net(inputs)
        _,Surrogate_outputs_da = Surrogate_net(inputs_da)
        loss = 0.5*criterion(Surrogate_outputs, targets) + 0.5*criterion(Surrogate_outputs_da, targets)
        
    

        loss.backward()
        optimizer.step()

        if args.is_wandb:
            wandb_record.log({'training loss':loss},step=batch_idx+ epoch*N_batch)
    return (epoch+1)*N_batch

# def cycle(iterable):
#     while True:
#         for x in iterable:
#             yield x



if __name__ == '__main__': 
    # global wandb_record   
    parser = argparse.ArgumentParser(description='PyTorch Clone Model Training')
    parser.add_argument('--gen_num_features', '-gnf', type=int, default=64,
                        help='Number of features of generator (a.k.a. nplanes or ngf). default: 64')
    parser.add_argument('--gen_dim_z', '-gdz', type=int, default=128,
                        help='Dimension of generator input noise. default: 128')
    parser.add_argument('--gen_bottom_width', '-gbw', type=int, default=4,
                        help='Initial size of hidden variable of generator. default: 4')
    parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',
                        help='Input noise distribution: normal (default) or uniform.')
    
    parser.add_argument('--dis_num_features', '-dnf', type=int, default=64,
                        help='Number of features of discriminator (a.k.a nplanes or ndf). default: 64')
    
    ######################333
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--max_epochs', default=200, type=int, help='max epochs')    
    parser.add_argument('--config_exp', default='', type=str, help='path of experiment config')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
    parser.add_argument('--is_wandb', action='store_true', help='is_wandb')   
    parser.add_argument('--attackid', default=-1, type=int)     
    
    parser.add_argument('--surrogate_model_id', default=1, type=int, help='Densenet121: 0 | Densenet161: 1 | Densenet169: 2')

    
    parser.add_argument('--is_general_gan', action='store_true', default=False,
                        help='Default: False')

    args = parser.parse_args()

        

    
    # global wandb_record
    loaded_args_exp = load_json(json_file=args.config_exp)
    loaded_args_data = load_json(json_file=loaded_args_exp['train']['config_dataset'])


    args.target = loaded_args_exp['train']['target']
    args.gen_data = loaded_args_exp['train']['gen_data']
    args.path_G = loaded_args_exp['train']['path_G']
    args.num_classes = loaded_args_data['dataset']['n_classes']
    models = loaded_args_exp['train']['models'].split(',')
    args.model = models[args.surrogate_model_id]
    
    if args.attackid>0:
        args.num_classes = args.attackid

    best_acc = 0  # best test accuracy
    best_val_acc = 0  # best test accuracy

    max_epochs = int(args.max_epochs)

    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True


    save_dir = os.path.join(loaded_args_exp['train']['result_dir'],loaded_args_data['dataset']['d_priv'],loaded_args_data['dataset']['d_pub'],loaded_args_exp['train']['target'],'S',args.model)
   
    os.makedirs(save_dir, exist_ok=True)
    log_path = "{}/attack_logs".format(save_dir)
    os.makedirs(log_path, exist_ok=True)
    log_file = "Surrogate_ckp_{}.txt".format(args.model)
    
    utils.Tee(os.path.join(log_path, log_file), 'w')
    
    print(args)

    print('--------',save_dir)
      
    Surrogate_net = get_model(args.model,args.num_classes)
    # netG,_ = loader.load_cgan(args,device,args.path_G) 
    # netG,_ = loader.load_cgan_wgan(args,device,args.path_G) 
    
    if args.is_general_gan == True:
        netG,_ = loader.load_cgan_wgan(args,device,args.path_G)
    else:
        netG,_ = loader.load_cgan(args,device,args.path_G)
    
   
    netG.eval()  

    print('==> Preparing data..')

                                                            
    _, testloader = init_dataloader(loaded_args_data, loaded_args_data['dataset']['test_file_path'], args.batch_size, mode="train", attackid = args.attackid)
        
    print("Surrogate at iteration 0, Acc = ", get_model_accuracy(Surrogate_net, testloader))
    
    print("max epochs = ", max_epochs)

    optimizer = optim.SGD(Surrogate_net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max_epochs)
    iterations = 0
    inital_lr = 0.01
    
    
    if args.is_wandb:
        project_id = "{}_{}_{}_{}".format(loaded_args_data['dataset']['d_priv'],loaded_args_data['dataset']['d_pub'],args.target,args.model)
   
        project = "surrogate_models"
        get_wandb(args.is_wandb,project_id,project,exp_dir =save_dir)

    print('---load synthetic dataset ---')
    gen_dataset = generated_dataset(args.gen_data)
    print('gen_dataset',len(gen_dataset))
    # exit()
    z_loader = iter(torch.utils.data.DataLoader(
        gen_dataset, args.batch_size,
        sampler=InfiniteSamplerWrapper(gen_dataset), #shuffle = True,
        drop_last=True)
    )


    model_path = os.path.join(save_dir,"{}_{}_e{}.pt".format(args.target,args.model,max_epochs))

    cls_ckpts = loaded_args_exp['train']['cls_ckpts'].split(',')
    cls_ckpts[args.surrogate_model_id] = model_path


    # add the surrogate model checkpoint to the config file
    loaded_args_exp['train']['cls_ckpts'] = ','.join(cls_ckpts)
    print('loaded_args_exp',loaded_args_exp['train']['cls_ckpts'])
    
    import json
        
    with open(args.config_exp, 'w') as f:
        json.dump(loaded_args_exp, f, indent=2)
    
    print('experiment',loaded_args_exp)

    for epoch in range(max_epochs):
        iter = train_generated_images(epoch,Surrogate_net,netG,optimizer,z_loader,args)        
              
        top1,top5,test_loss = test_acc_loss(Surrogate_net, dataloader=testloader)   

        print("Iter = {}, Learning rate : {}, top1 = {}, top5 = {}".format(iter, scheduler.get_lr()[0],top1,top5))

        
        if args.is_wandb:
            record_dict ={'lr':scheduler.get_lr()[0],'top1':top1,'top5':top5,'test_loss':test_loss}
            wandb_record.log(record_dict)   

        
        scheduler.step()

    torch.save({'state_dict':Surrogate_net.state_dict()}, os.path.join(save_dir,"{}_{}_e{}.pt".format(args.target,args.model,max_epochs)))

    #############33 test on private test set  
    top1,top5,test_loss = test_acc_loss(Surrogate_net, dataloader=testloader)    
    print("Test acc (private dataset): top 1 = {}, top 5 = {}".format(epoch,top1,top5))
