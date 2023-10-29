
import numpy as np
import os
import torch
from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
from utils import save_tensor_images
import loader
import os  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import load_json

from dataset import   sample_from_gen_with_targetID
def get_prob(net,img,iden):
    
    Softmax = torch.nn.Softmax()
    _, logit_ = net(img)
    softmax = Softmax(logit_)
    N = iden.shape[0]
    prob = torch.zeros(N)
    eval_iden = torch.argmax(logit_, dim=1).view(-1)
    
    is_success = np.zeros(N)
    for i in range(N): 
        prob[i]= softmax[i,iden[i]].item()    
        if eval_iden[i] == iden[i]:
            is_success[i] = 1
    return prob,is_success,eval_iden
    



if __name__ == "__main__":
    global args, logger

    parser = ArgumentParser(description='Generate data')
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
    
    # parser.add_argument('--N', type=int, default=100)                        
    parser.add_argument('--batch_size', type=int, default=250)
 
    parser.add_argument('--max_iter', type=int, default=2)
    parser.add_argument('--config_exp', type=str,default='')    

    
    parser.add_argument('--is_wandb', action='store_true', default=False,
                        help='If you like wandb. Default: False')
    parser.add_argument('--is_general_gan', action='store_true', default=False,
                        help='Default: False')
    args = parser.parse_args()

    
    loaded_args_exp = load_json(json_file=args.config_exp)
    loaded_args_dataset = load_json(json_file=loaded_args_exp['train']['config_dataset'])



    print('loaded_args',loaded_args_dataset)
    args.data_name = loaded_args_dataset['dataset']['d_pub']
    args.num_classes = loaded_args_dataset['dataset']['n_classes']
    args.path_G = loaded_args_exp['train']['path_G']
    args.path_T = loaded_args_exp['train']['path_T']
    args.target_model = loaded_args_exp['train']['target']

    # G,_ = loader.load_cgan(args,device,args.path_G)
    if args.is_general_gan == True:
        G,_ = loader.load_cgan_wgan(args,device,args.path_G)
    else:
        G,_ = loader.load_cgan(args,device,args.path_G)
    
   
    # G,_ = loader.load_cgan_wgan(args,device,args.path_G)

    T = loader.load_classifier(args.target_model,args.num_classes,args.path_T)
   
    T.eval()   
    G.eval()

    
    # N=args.N
    max_iter = args.max_iter
    gen_y = np.empty(0)
    total_api = 0


    fol_dir = os.path.join(loaded_args_exp['train']['result_dir'],loaded_args_dataset['dataset']['d_priv'],loaded_args_dataset['dataset']['d_pub'],loaded_args_exp['train']['target'],'generated_data_tacgan')
    data_filename = os.path.join(fol_dir,'{}_{}_{}'.format(args.target_model,loaded_args_dataset['dataset']['d_priv'],loaded_args_dataset['dataset']['d_pub']))
    
    os.makedirs(fol_dir, exist_ok=True)
    loaded_args_exp['train']['gen_data']  = data_filename+'.pt'
    
    ####### save generated data to config
    with open(args.config_exp, 'w') as f:
        json.dump(loaded_args_exp, f, indent=2)
        
    print('experiment',loaded_args_exp)




    for i in range(args.num_classes):

        iden = torch.from_numpy(np.zeros(args.batch_size)+i).to(torch.int).cuda()
        
        latent = np.empty(0)
        for iter in range(max_iter):
            
            with torch.no_grad():
                x,_,z = sample_from_gen_with_targetID(args, device,args.num_classes, G,i,args.batch_size)
                # save_tensor_images(x,'{}{}.png'.format(fol_dir,i))
                total_api += x.shape[0]
                _,_,eval_iden = get_prob(T,x,iden)
                
                if latent.shape[0] ==0:
                    latent = z
                    label = eval_iden
                else:
                    latent = torch.cat((latent,z),dim=0)   
                    label = torch.cat((label,eval_iden),dim=0)                   
                    print('---',i,z.shape,label.shape)                 
        pseudo_y = torch.from_numpy(np.zeros(latent.shape[0])+i).to(torch.int)
        if gen_y.shape[0] == 0:
            gen_y = label
            gen_z = latent
            gen_pseudo_y  = pseudo_y
        else:         
            gen_y =torch.cat((gen_y,label),dim=0)   
            gen_z =torch.cat((gen_z,latent),dim=0)
            gen_pseudo_y =torch.cat((gen_pseudo_y,pseudo_y),dim=0)
        print('-------',i,gen_y.shape,gen_pseudo_y.shape)

    torch.save({'y':gen_y,'z':gen_z, 'gen_pseudo_y':gen_pseudo_y,'total_api':total_api},data_filename +'.pt')
    
    ############## visualization
    data =torch.load(data_filename+'.pt')


    data = plt.hist(data['y'].cpu().numpy(),bins=args.num_classes)
    hist = data[0]
    x = data[1]
    print(hist.shape)
    sorted_hist = np.sort(hist)
    print(sorted_hist)
    plt.figure()
    plt.plot(sorted_hist)
    plt.bar(np.arange(0,args.num_classes), sorted_hist,1)            
    plt.show()
    
    plt.savefig(data_filename+'.png')


