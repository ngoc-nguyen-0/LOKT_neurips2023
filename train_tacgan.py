
import json
import os
import torch
import argparse
import losses as L
import utils
from dataset import   sample_from_data_wo_label, sample_from_gen,sample_from_gen_test
import loader
from utils import init_dataloader,load_json,test,decision

import wandb

wandb_record = None


def prepare_results_dir(args):
    """Makedir, init tensorboard if required, save args."""
    root = args.results_root# os.path.join(args.results_root,
                        #args.data_name, args.target_model)
    os.makedirs(root, exist_ok=True)

    

    train_image_root = os.path.join(root, "preview", "train")
    eval_image_root = os.path.join(root, "preview", "eval")
    os.makedirs(train_image_root, exist_ok=True)
    os.makedirs(eval_image_root, exist_ok=True)

    args.results_root = root
    args.train_image_root = train_image_root
    args.eval_image_root = eval_image_root

    if args.num_classes > args.n_eval_batches:
        args.n_eval_batches = args.num_classes
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size // 4

  
    with open(os.path.join(root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))
    return args

def get_wandb(is_wandb, project_id,project='tacgan_same_arch',exp_dir='.'):
    global wandb_record
    
    wandb_record = None
    print('-------is_wandb',is_wandb)
    if is_wandb == True:
        wandb_record = wandb.init(project=project, id=project_id, save_code =True, dir=exp_dir)
    return wandb_record

def get_args():
    parser = argparse.ArgumentParser(description='Stage-1: Train the Pseudo Label-Guided Conditional GAN')
    # Dataset configuration
    parser.add_argument('--batch_size', '-B', type=int, default=128,
                        help='mini-batch size of training data. default: 64')
    parser.add_argument('--eval_batch_size', '-eB', default=None,
                        help='mini-batch size of evaluation data. default: None')
    # Generator configuration
    parser.add_argument('--gen_num_features', '-gnf', type=int, default=64,
                        help='Number of features of generator (a.k.a. nplanes or ngf). default: 64')
    parser.add_argument('--gen_dim_z', '-gdz', type=int, default=128,
                        help='Dimension of generator input noise. default: 128')
    parser.add_argument('--gen_bottom_width', '-gbw', type=int, default=4,
                        help='Initial size of hidden variable of generator. default: 4')
    parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',
                        help='Input noise distribution: normal (default) or uniform.')
    # Discriminator (Critic) configuration
    parser.add_argument('--dis_num_features', '-dnf', type=int, default=64,
                        help='Number of features of discriminator (a.k.a nplanes or ndf). default: 64')
    
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Initial learning rate of Adam. default: 0.0002')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='beta1 (betas[0]) value of Adam. default: 0.0')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 (betas[1]) value of Adam. default: 0.9')
    # Training setting
    parser.add_argument('--seed', type=int, default=46,
                        help='Random seed. default: 46 (derived from Nogizaka46)')
    parser.add_argument('--max_iteration', '-N', type=int, default=19000,
                        help='Max iteration number of training. default: 20000')
    parser.add_argument('--n_dis', type=int, default=5,
                        help='Number of discriminator updater per generator updater. default: 5')
    parser.add_argument('--loss_type', type=str, default='hinge',
                        help='loss function name. hinge (default) or dcgan.')
    parser.add_argument('--cGAN', default=False, action='store_true')
    
    # Log and Save interval configuration
    parser.add_argument('--is_wandb', action='store_true', default=False,
                        help='If you like wandb. Default: False')
    parser.add_argument('--checkpoint_interval', '-ci', type=int, default=1000,
                        help='Interval of saving checkpoints (model and optimizer). default: 1000')
    parser.add_argument('--log_interval', '-li', type=int, default=100,
                        help='Interval of showing losses. default: 100')
    parser.add_argument('--eval_interval', '-ei', type=int, default=1000,
                        help='Interval for evaluation (save images and FID calculation). default: 1000')
    parser.add_argument('--n_eval_batches', '-neb', type=int, default=300,
                        help='Number of mini-batches used in evaluation. default: 100')
    # Model Inversion
    parser.add_argument('--alpha', type=float, default=0.2, help='weight of loss_c. default: 0.2')  
    parser.add_argument('--config_exp', type=str, default='path to dataset config')
    parser.add_argument('--classid', type=str, default='0')
    
    
    parser.add_argument('--is_general_gan', action='store_true', default=False,
                        help='Default: False')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    
    loaded_args_exp = load_json(json_file=args.config_exp)
    loaded_args_dataset = load_json(json_file=loaded_args_exp['train']['config_dataset'])


    args.target_model = loaded_args_exp['train']['target']
    args.path_T = loaded_args_exp['train']['path_T']
    print('--------------dataset --------------')
    print(loaded_args_dataset)

    args.data_name = loaded_args_dataset['dataset']['d_pub']
    args.num_classes = loaded_args_dataset['dataset']['n_classes']
    

    project_name = os.path.join(loaded_args_exp['train']['result_dir'],loaded_args_dataset['dataset']['d_priv'],loaded_args_dataset['dataset']['d_pub'],loaded_args_exp['train']['target'],'tacgan_{}_wgan'.format(args.alpha))
    args.results_root = project_name
    print('---------save results at: ',project_name)
    print("Target Model:", args.target_model)

    # CUDA setting
    if not torch.cuda.is_available():
        raise ValueError("Should buy GPU!")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True

    # Load public dataset
    _, train_loader = init_dataloader(loaded_args_dataset, loaded_args_dataset['dataset']['pub_file_path'], args.batch_size, mode="gan")

    # Test the accuracy of C \circ D on private dataset
    _, test_loader = init_dataloader(loaded_args_dataset, loaded_args_dataset['dataset']['test_file_path'], args.batch_size, mode="test")
 
    
    print(' prepared datasets...')
    print(' Number of training images: {}'.format(len(train_loader)))
    # Prepare directories.
    args = prepare_results_dir(args)

    # tacgan   
    # gen,dis = loader.load_cgan(args,device)
    if args.is_general_gan == True:
        gen,dis = loader.load_cgan_wgan(args,device)
    else:
        gen,dis = loader.load_cgan(args,device)
    
   

    # print('gen',gen)
    # print('dis',dis)
    # exit()
    #Target model resnet50: top 1 = 90.08167613636364, top 5 = 95.46132489669421
    target_model = loader.load_classifier(args.target_model,args.num_classes,args.path_T)
    # print(target_model.module)
    target_model.eval()
    
    # top1 = 90.08
    # top5 = 95.46
    top1,top5 = test(target_model, dataloader=test_loader)


    print("Target model {}: top 1 = {}, top 5 = {}".format(args.target_model,top1,top5))
    # exit()
    # load optimizer
    opt_gen = torch.optim.Adam(gen.parameters(), args.lr, (args.beta1, args.beta2))
    opt_dis = torch.optim.Adam(dis.parameters(), args.lr, (args.beta1, args.beta2))

    
    if args.is_wandb:
        project_id = "{}_{}_{}_{}".format(loaded_args_dataset['dataset']['d_priv'],loaded_args_dataset['dataset']['d_pub'],args.target_model,args.alpha)    
        print('project_id',project_id)    
        get_wandb(args.is_wandb,project_id, exp_dir = args.results_root)

    BCE_loss = torch.nn.BCELoss().cuda()
    y_real_, y_fake_ = torch.ones(args.batch_size, 1), torch.zeros(args.batch_size, 1)
    
    # train G,D without loss_C 
    # if loaded_args_dataset['dataset']['d_pub'] =='celeba' or loaded_args_dataset['dataset']['d_pub'] =='ffhq' :
    #     n_uncond = 1000     
    if loaded_args_dataset['dataset']['d_pub'] =='facescrub':
        n_uncond = 3000
    elif loaded_args_dataset['dataset']['d_pub'] =='pubfig':
        n_uncond = 6000
    elif loaded_args_dataset['dataset']['d_pub'] =='ffhq':
        n_uncond = 2000
    else:
        n_uncond = 1000
    iter_t = 0
    args.max_iteration = args.max_iteration + n_uncond
    print('n_uncond',n_uncond)
    iter_t = 0
    

    
    ####### save path_G and path_D to config
    loaded_args_exp['train']['path_G'] = os.path.join(
		args.results_root,
		'gen_{}_iter_{:07d}.pth.tar'.format(int(args.max_iteration/args.checkpoint_interval), args.max_iteration)
	)
    loaded_args_exp['train']['path_D'] = os.path.join(
		args.results_root,
		'dis_{}_iter_{:07d}.pth.tar'.format(int(args.max_iteration/args.checkpoint_interval), args.max_iteration)
	)

        
    with open(args.config_exp, 'w') as f:
        json.dump(loaded_args_exp, f, indent=2)
        
    print('experiment',loaded_args_exp)


    print('-----------------------Training-----------------------')
    for n_iter in range(1, args.max_iteration + 1):
        # ==================== Beginning of 1 iteration. ====================
        _l_g = .0
        cumulative_inv_loss = 0.
        cumulative_loss_dis = .0
        
        cumulative_gen_acc, cumulative_dis_acc, cumulative_dis_yhat_acc = 0,0,0

        target_correct = 0
        dis_y_correct = 0
        dis_yhat_correct = 0
        count = 0
        inv_loss = 0
        loss_c = 0
        if n_iter < n_uncond:
            alpha = 0
        else:
            alpha = args.alpha
        for i in range(args.n_dis):  # args.ndis=5, Gen update 1 time, Dis update ndis times.
            if i == 0:
                fake, pseudo_y, _ = sample_from_gen(args, device, args.num_classes, gen,args.batch_size)
                dis_fake,dis_class = dis(fake)
                # calc the loss of G
                loss_gen = BCE_loss(dis_fake, y_real_)
                loss_gen_all = loss_gen 

                # calc the loss_c
                if n_iter > n_uncond:
                    loss_c = L.cross_entropy_loss(dis_class, pseudo_y)
               
                    loss_gen_all += loss_c * alpha
                    cumulative_inv_loss += loss_c.item()
                # update the G
                gen.zero_grad()
                loss_gen_all.backward()
                opt_gen.step()
                _l_g += loss_gen.item()

                
            # sample the real images
            real = sample_from_data_wo_label(device, train_loader)
            
            with torch.no_grad():                
                # generate fake images
                fake, pseudo_y, _ = sample_from_gen(args, device, args.num_classes, gen,args.batch_size)
                iter_t +=1
                if n_iter > n_uncond:
                    
                    fake = fake.detach()
                    y_fake = decision(fake, target_model)
                    y_fake = y_fake.detach()                    
                    
                    ####### eval G
                    target_correct += y_fake.eq(pseudo_y.view_as(y_fake)).sum().item()
                
            

            # calc the loss of D
            dis_fake, dis_fake_class = dis(fake)
            dis_real, _ = dis(real)
            loss_fake = BCE_loss(dis_fake, y_fake_)
            loss_real = BCE_loss(dis_real, y_real_)
            loss_dis = loss_fake + loss_real
            if n_iter > n_uncond:                
                loss_c = L.cross_entropy_loss(dis_fake_class, y_fake)                
                loss_dis+= loss_c* alpha
            # update D
            dis.zero_grad()
            loss_dis.backward()
            opt_dis.step()

            cumulative_loss_dis += loss_dis.item()
            if n_iter > n_uncond:
                with torch.no_grad():
                    count += fake.shape[0]     
                    T_preds = dis_fake_class.max(1, keepdim=True)[1]                
                    ########## dis           
                    dis_y_correct += T_preds.eq(pseudo_y.view_as(T_preds)).sum().item() # predict of D = y
                    ########## dis to target                
                    dis_yhat_correct += T_preds.eq(y_fake.view_as(T_preds)).sum().item() # predict of D = \hat{y} = T(G(x,y))

                
            if n_iter % 10 == 0 and i == args.n_dis - 1:
                cumulative_loss_dis = cumulative_loss_dis/ args.n_dis
                if n_iter > n_uncond:
                    cumulative_gen_acc = target_correct*100.0/  count
                    cumulative_dis_acc = dis_y_correct*100.0/ count
                    cumulative_dis_yhat_acc = dis_yhat_correct*100.0/ count
        # ==================== End of 1 iteration. ====================
        if n_iter %10 == 0:
            if n_iter %100 == 0:
                if n_iter %1000 == 0:
                    with torch.no_grad():
                        top1,top5 = test(dis, dataloader=test_loader)                
                        print("Discriminator Epoch = {}: top 1 = {}, top 5 = {}".format(n_iter,top1,top5))

                if args.is_wandb:            
                    record_dict = {"Top1":top1,"Top5":top5, "dis_loss":cumulative_loss_dis,\
                            "gen_acc":cumulative_gen_acc,\
                            "dis_acc":cumulative_dis_acc,\
                            "dis_yhat_acc":cumulative_dis_yhat_acc,\
                            "gen_loss":_l_g,\
                            "gen_inv_loss": inv_loss}            
                    wandb_record.log(record_dict,step=iter_t)

            else:
                if args.is_wandb:            
                    record_dict = {"dis_loss":cumulative_loss_dis,\
                            "gen_acc":cumulative_gen_acc,\
                            "dis_acc":cumulative_dis_acc,\
                            "dis_yhat_acc":cumulative_dis_yhat_acc,\
                            "gen_loss":_l_g,\
                            "gen_inv_loss": inv_loss}            
                    wandb_record.log(record_dict,step=iter_t)


        if n_iter % args.log_interval == 0:
            print(
                'iteration: {:07d}/{:07d}, loss gen: {:05f}, loss dis {:05f}, inv loss {:05f}, target acc {:04f}, dis acc {:04f}'.format(
                    n_iter, args.max_iteration, _l_g, cumulative_loss_dis, cumulative_inv_loss,
                    cumulative_gen_acc, cumulative_dis_acc))
            # Save previews
            with torch.no_grad():
                fake, pseudo_y, _ = sample_from_gen_test(args, device, 300, gen) 
                utils.save_images(
                    n_iter, n_iter // args.checkpoint_interval, args.results_root,
                    args.train_image_root, fake, real
                )
           

        if n_iter % args.checkpoint_interval == 0:
            # Save checkpoints!z
            utils.save_checkpoints(
                args, n_iter, n_iter // args.checkpoint_interval,
                gen, opt_gen, dis, opt_dis
            )
    
if __name__ == '__main__':
    main()
