import logging
import numpy as np
import os
import random
import statistics
import time
import torch
from argparse import ArgumentParser
from kornia import augmentation

import json
import losses as L
import utils

from utils import save_tensor_images,load_json
import loader
import os  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.classifiers.classifier import get_model

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(42)


# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def inversion(args, G, T, E, iden, itr, lr=2e-2, iter_times=1500, num_seeds=5, clipz = 0.5):
    save_img_dir = os.path.join(args.save_dir, 'all_imgs_{}'.format(iter_times))
    save_z_dir = os.path.join(args.save_dir, 'all_z_{}_'.format(iter_times))
    success_dir = os.path.join(args.save_dir, 'success_imgs_{}'.format(iter_times))
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(save_z_dir, exist_ok=True)

    bs = iden.shape[0]
    iden = iden.view(-1).long().cuda()

    G.eval()
    # T.eval()
    E.eval()

    flag = torch.zeros(bs)
    no = torch.zeros(bs)  # index for saving all success attack images

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, 5))

    aug_list = augmentation.container.ImageSequential(
        augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        augmentation.ColorJitter(brightness=0.2, contrast=0.2),
        augmentation.RandomHorizontalFlip(),
        augmentation.RandomRotation(5),
    )

    for random_seed in range(num_seeds):
        outputs_z ='{}{}_{}_{}.pt'.format(save_z_dir,iden[0],random_seed,iter_times-1)
        
        print('outputs_z',outputs_z)
        r_idx = random_seed
        
        tf = time.time()
        if not os.path.exists(outputs_z):
            set_random_seed(random_seed)

            z = utils.sample_z(
                bs, args.gen_dim_z, device, args.gen_distribution
            )
            z.requires_grad = True
            
            torch.save({'z':z,'iden':iden},'{}{}_{}_{}.pt'.format(save_z_dir,iden[0],random_seed,0))
            optimizer = torch.optim.Adam([z], lr=lr)

            for i in range(iter_times):                
                fake = G(z, iden)
                inv_loss = 0
                for tn in T:
                    out1 = tn(aug_list(fake))[-1]
                    out2 = tn(aug_list(fake))[-1]

                    if z.grad is not None:
                        z.grad.data.zero_()

                    if args.inv_loss_type == 'ce':
                        inv_loss += L.cross_entropy_loss(out1, iden) + L.cross_entropy_loss(out2, iden)
                    elif args.inv_loss_type == 'margin':
                        inv_loss += L.max_margin_loss(out1, iden) + L.max_margin_loss(out2, iden)
                    elif args.inv_loss_type == 'poincare':
                        inv_loss += L.poincare_loss(out1, iden) + L.poincare_loss(out2, iden)
                inv_loss = inv_loss / len(T)
                
                optimizer.zero_grad()
                inv_loss.backward()
                optimizer.step()
                
                # z.data = torch.clamp(z.data,-clipz,clipz)
                inv_loss_val = inv_loss.item()

                if (i + 1) % 100 == 0:
                    with torch.no_grad():
                        fake_img = G(z, iden)
                        eval_prob = E(augmentation.Resize((112, 112))(fake_img))[-1]
                        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                        acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                        print("Iteration:{}\tInv Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, inv_loss_val, acc))
                    torch.save({'z':z,'iden':iden},'{}{}_{}_{}.pt'.format(save_z_dir,iden[0],random_seed,i))
        else:
            print('load',outputs_z)
            z_loader = torch.load(outputs_z)
            z = z_loader['z']            
            z = z.to(device)

        with torch.no_grad():
            fake = G(z, iden)
            eval_prob = E(augmentation.Resize((112, 112))(fake))[-1]
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

            cnt, cnt5 = 0, 0
            for i in range(bs):
                gt = iden[i].item()
                sample = G(z, iden)[i]
                # all_img_class_path = os.path.join(save_img_dir, str(gt))
                # if not os.path.exists(all_img_class_path):
                #     os.makedirs(all_img_class_path)
                save_tensor_images(sample.detach(),
                                   os.path.join(save_img_dir, "attack_iden_{}_{}.png".format(gt, r_idx)))

                if eval_iden[i].item() == gt:
                    seed_acc[i, r_idx] = 1
                    cnt += 1
                    flag[i] = 1
                    best_img = G(z, iden)[i]
                    # success_img_class_path = os.path.join(success_dir, str(gt))
                    # if not os.path.exists(success_img_class_path):
                    #     os.makedirs(success_img_class_path)
                    save_tensor_images(best_img.detach(), os.path.join(success_dir,
                                                                       "{}_attack_iden_{}_{}.png".format(itr, gt,
                                                                                                         int(no[i]))))
                    no[i] += 1
                _, top5_idx = torch.topk(eval_prob[i], 5)
                if gt in top5_idx:
                    cnt5 += 1

            interval = time.time() - tf
            print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))
            res.append(cnt * 1.0 / bs)
            res5.append(cnt5 * 1.0 / bs)
            torch.cuda.empty_cache()

    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
    acc_var = statistics.variance(res)
    acc_var5 = statistics.variance(res5)
    print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))

    return acc, acc_5, acc_var, acc_var5

if __name__ == "__main__":
    global args, logger

    parser = ArgumentParser(description='Stage-2: Image Reconstruction')
    parser.add_argument('--inv_loss_type', type=str, default='margin', help='ce | margin | poincare')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--iter_times', type=int, default=600)
    # Generator configuration
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
    # path
    parser.add_argument('--save_dir', type=str, default='tacgan')    
    parser.add_argument('--classid', type=str, default='')
    parser.add_argument('--config_exp', type=str, default='')
    parser.add_argument('--N', type=int, default=-1)
    
    parser.add_argument('--is_general_gan', action='store_true', default=False,
                        help='Default: False')

    
                        
    args = parser.parse_args()
    logger = get_logger()

    logger.info(args)
    logger.info("=> creating model ...")

    set_random_seed(42)
    cid = args.classid.split(',')

    loaded_args_exp = load_json(json_file=args.config_exp)
    loaded_args_dataset = load_json(json_file=loaded_args_exp['train']['config_dataset'])

    ######################### dataset
    print('loaded_args_dataset',loaded_args_dataset)
    args.data_name = loaded_args_dataset['dataset']['d_pub']
    args.num_classes = loaded_args_dataset['dataset']['n_classes']
    
    ######################## experiment
    model = loaded_args_exp['train']['models'].split(',')
    pathT = loaded_args_exp['train']['cls_ckpts'].split(',')
    path_G = loaded_args_exp['train']['path_G']
    args.save_dir = os.path.join(loaded_args_exp['train']['result_dir'],loaded_args_dataset['dataset']['d_priv'],loaded_args_dataset['dataset']['d_pub'],loaded_args_exp['train']['target'],args.save_dir, args.classid.replace(',','_'))
    os.makedirs(args.save_dir, exist_ok=True)
   
    print('saving results at {}'.format(args.save_dir))
    print('model',model)
    print('pathT',pathT)
    ####################### model loading
    # evaluation model
    E = get_model('FaceNet',loaded_args_dataset['dataset']['E_n_classes'],loaded_args_dataset['dataset']['path_E'])
    
    #target model   
    T = []
    for id_ in cid:
        i = int(id_)
        print(i,model[i],pathT[i])
        if model[i]=='T':
            net = loader.load_classifier(loaded_args_exp['train']['target'],args.num_classes,loaded_args_exp['train']['path_T'])
        elif model[i]=='D':
            # net = loader.load_classifier(model[i],args.num_classes,loaded_args_exp['train']['path_D'])
            
            _,net = loader.load_cgan_wgan(args,device,path_G,loaded_args_exp['train']['path_D'])
        else:
            net = loader.load_classifier(model[i],args.num_classes,pathT[i])
        net = net.eval()
        T.append(net)
    
    #tacgan
    print('path_G',path_G)
    # G,_ = loader.load_cgan(args,device,path_G)
    # G,_ = loader.load_cgan_wgan(args,device,path_G)

    if args.is_general_gan == True:
        G,_ = loader.load_cgan_wgan(args,device,loaded_args_exp['train']['path_G'])
    else:
        G,_ = loader.load_cgan(args,device,loaded_args_exp['train']['path_G'])
    
    if loaded_args_dataset['dataset']['d_priv'] == 'facescrub': # 200 identities
        bs = 50
        clipz = 0.5
        if args.N<0:
            N = 4
        else:
            N = args.N
    elif loaded_args_dataset['dataset']['d_priv'] == 'pubfig': # 50 identities
        bs = 50
        N = 1
        clipz = 0.5
    elif loaded_args_dataset['dataset']['d_priv'] == 'celeba': # evaluate on the first 300 identities only
        bs = 60
        clipz = 1.0
        if args.N<0:
            N = 5
        else:
            N = args.N
    E.eval()
    G.eval()


    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
    # save all config to the result dir
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))
    
    logger.info("=> Begin attacking ...")
    for i in range(1): # increase i if want to generate more, default 1 means 5 images/ID
        iden = torch.from_numpy(np.arange(bs))

        for idx in range(N):
            if idx >-1:
                print("--------------------- Attack batch [%s]------------------------------" % idx)
                # reconstructed private images

                acc, acc5, var, var5 = inversion(args, G, T, E, iden, itr=i, lr=args.lr, iter_times=args.iter_times,
                                                num_seeds=5, clipz=clipz)
                
                aver_acc += acc / N
                aver_acc5 += acc5 /N
                aver_var += var / N
                aver_var5 += var5 / N
            iden = iden + bs


    print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(aver_acc,
                                                                                                            aver_acc5,
                                                                                                            aver_var,
                                                                                                            aver_var5))


