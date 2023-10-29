
import torch
import numpy as np

from loader import *
import os
from evaluation_lib.fid import concatenate_list, gen_samples
import utils
from evaluation_lib.fid import concatenate_list

import statistics 

def get_fea_target(E,filepath,dataset ='celeba'):
   
    target_x = np.load(filepath+'.npy')
    label_x = np.load(filepath+'_label.npy')
    bs = 60
    N = target_x.shape[0]
    print('N',N)
    with torch.no_grad():
        for i in range(0,N,bs):     
            # print('i',i,int(N/bs)+1)
            if i + bs > N:
                img = target_x[i:N,:,:,:]   
            else:
                img = target_x[i:i+bs,:,:,:] 
            img = torch.from_numpy(img).cuda()  
            eval_fea,_ = E(utils.low2high(img))
         
            if i==0:
                all_fea = eval_fea.cpu().numpy()
                
            else:                        
                all_fea = np.concatenate((all_fea,eval_fea.cpu().numpy()))
            print('all_imgs',all_fea.shape)
    
    img_ids_path = '{}_fea_target_knn.npy'.format(dataset)
    np.save(img_ids_path,{'fea':all_fea,'label':label_x})
    return all_fea


def find_shortest_dist(fea_target,fea_fake):
    shortest_dist = 0
    pdist = torch.nn.PairwiseDistance(p=2)

    fea_target = torch.from_numpy(fea_target).cuda()
    fea_fake = torch.from_numpy(fea_fake).cuda()
    # print('---fea_fake.shape[0]',fea_fake.shape[0])
    for i in range(fea_fake.shape[0]):
        dist = pdist(fea_fake[i,:], fea_target)
        
        min_d = min(dist)
        
        # print('--KNN dist',min_d)
        shortest_dist = shortest_dist + min_d*min_d
    # print('--KNN dist',shortest_dist)

    return shortest_dist
def find_shortest_dist_list(fea_target,fea_fake):

    pdist = torch.nn.PairwiseDistance(p=2)

    fea_target = torch.from_numpy(fea_target).cuda()
    fea_fake = torch.from_numpy(fea_fake).cuda()
    # print('---fea_fake.shape[0]',fea_fake.shape[0])
    knn_dist = torch.zeros(fea_fake.shape[0])
    for i in range(fea_fake.shape[0]):
        dist = pdist(fea_fake[i,:], fea_target)
        
        min_d = min(dist)
        
        knn_dist[i] = min_d*min_d

    return knn_dist


def knn_dist(target_dir, fake_dir,n=300):
    knn = 0
    target = np.load(target_dir,allow_pickle=True)  
    fake = np.load(fake_dir,allow_pickle=True)  
    target_fea = target.item().get('fea')    
    target_y = target.item().get('label')
    fake_fea = fake.item().get('fea')
    fake_y = fake.item().get('label')

    fake_fea = concatenate_list(fake_fea)
    fake_y = concatenate_list(fake_y)
    
    N = fake_fea.shape[0]
    knn_list = []
    for id in range(n):
        id_f = fake_y == id
        id_t = target_y == id
        # print('id_f',sum(id_f),sum(id_t))
        if sum(id_f)>0 and sum(id_t)>0:
            fea_f = fake_fea[id_f,:]
            fea_t = target_fea[id_t]
            shorted_dist_list_ = find_shortest_dist_list(fea_t,fea_f)
            
            knn_list.append(shorted_dist_list_)
    
    knn_list = concatenate_list(knn_list)
    knn_list = np.float64(knn_list)
    knn = statistics.mean(knn_list)
    std = statistics.stdev(knn_list)
    return knn,std

def eval_KNN(dataset,fake_dir):   
    knn,std = knn_dist('./metadata/{}_fea_target_knn.npy'.format(dataset), fake_dir)
    print("KNN:{:.3f} std = {:.3f}".format(knn,std))
    return knn,std

#################3

def find_pair(fea_target,fea_fake):
    shortest_dist = 0
    pdist = torch.nn.PairwiseDistance(p=2)

    fea_target = torch.from_numpy(fea_target).cuda()
    fea_fake = torch.from_numpy(fea_fake).cuda()
    # print('---fea_fake.shape[0]',fea_fake.shape[0])
    min_dist = 100000
    
    for i in range(fea_fake.shape[0]):
        dist = pdist(fea_fake[i,:], fea_target)
        
        min_d = min(dist)
        # print(i,dist,min_d)
        if min_d<min_dist:
            index_fake = i
            index_real = torch.argmin(dist)
            # print('index_real',index_real)
            min_dist = min_d
    # print('index_fake, index_real, min_dist',index_fake, index_real.item(), min_dist.item())
    return index_fake, index_real.item(), min_dist.item()

from utils import save_tensor_images
def find_closest_pairs(save_dir,target_dir, fake_dir,filepath):

    target = np.load(target_dir,allow_pickle=True)  
    fake = np.load(fake_dir,allow_pickle=True)  
    target_fea = target.item().get('fea')    
    target_y = target.item().get('label')
    fake_fea = fake.item().get('sucessful_fea')
    fake_y = fake.item().get('label')
    target_x = np.load(filepath)
    # print('target_x',target_x)
    # exit()
    fake_x = fake.item().get('sucessful_imgs')


    fake_fea = concatenate_list(fake_fea)
    fake_y = concatenate_list(fake_y)
    fake_x = concatenate_list(fake_x)
    best_fake = np.zeros(300) -1
    best_priv = np.zeros(300) -1
    best_knn = np.zeros(300)+10000
    
    for id in range(300):
        id_f = fake_y == id
        if sum(id_f)>0:
            id_t = target_y == id
            fea_f = fake_fea[id_f,:]
            fea_t = target_fea[id_t]
            
            index_fake, index_real, knn_dist = find_pair(fea_t,fea_f)
            best_fake[id]=int(index_fake)
            best_priv[id]=int(index_real)
            best_knn[id] = knn_dist
        
    id_list = np.arange(300)
    index = np.argsort(best_knn)
    best_fake = best_fake[index]
    best_priv = best_priv[index]
    id_list = id_list[index]
    print(target_x.shape)
    for i in range(300):
        
        id_f = fake_y == id_list[i]
        if sum(id_f)>0:
            id_t = target_y == id_list[i]
            
            fake = fake_x[id_f]
            priv = target_x[id_t]
            # print('best_fake[i]',best_fake[i])
            
            fake = torch.from_numpy(fake[int(best_fake[i])]).cuda()
            priv =  torch.from_numpy(priv[int(best_priv[i])]).cuda()
            
            fake = utils.low2high(torch.unsqueeze(fake,0))
            priv = utils.low2high(torch.unsqueeze(priv,0))

            save_tensor_images(fake,'{}/{}_fake.png'.format(save_dir,i))
            save_tensor_images(priv,'{}/{}_priv.png'.format(save_dir,i))

def find_closest_pairs_visualization(target_dir, fake_x,fake_fea,fake_y,filepath):

    target = np.load(target_dir,allow_pickle=True)  
    # fake = np.load(fake_dir,allow_pickle=True)  
    target_fea = target.item().get('fea')    
    target_y = target.item().get('label')
    target_x = np.load(filepath)
    best_fake = np.zeros(fake_x.shape[0]) -1
    best_priv = np.zeros(fake_x.shape[0]) -1
    best_knn = np.zeros(fake_x.shape[0])+10000
    
    # print('target_x',target_x.shape)
    # exit()
    # print('fake_y',fake_y)
    for i in range(fake_y.shape[0]):
        id = fake_y[i]
        # print('id',id)
        id_f = fake_y == id
        if sum(id_f)>0:
            id_t = target_y == id
            fea_f = fake_fea[id_f,:]
            fea_t = target_fea[id_t]
            # print('fea_f',fea_f.shape,fea_t.shape)
            index_fake, index_real, knn_dist = find_pair(fea_t,fea_f)
            # print('----index_real',index_real)
            best_fake[i]=int(index_fake)
            best_priv[i]=int(index_real)
            best_knn[i] = knn_dist
        
    # id_list = np.arange(300)
    # index = np.argsort(best_knn)
    # best_fake = best_fake[index]
    # best_priv = best_priv[index]
    # # id_list = id_list[index
    # # ]
    # print('best_pri',best_pri)
    # print(best_fake)
    
    count = 0
    for i in range(fake_y.shape[0]):
        # print(i)
        if sum(id_f)>0:
            id_t = target_y == fake_y[i]
            
            fake = fake_x[i]
            priv = target_x[id_t]
        
        fake = torch.from_numpy(fake)
        priv =  torch.from_numpy(priv[int(best_priv[i])])
        fake = torch.unsqueeze(fake,0)
        priv = torch.unsqueeze(priv,0)
        
        if count == 0:
            priv_all= priv
            fake_all = fake
        else:
            priv_all = torch.cat((priv_all,priv))
            fake_all = torch.cat((fake_all,fake))

        count +=1
        # print('fake_all',fake_all.shape)
        # print('priv_all',priv_all.shape)
    return priv_all,fake_all

def find_samples(args):
    print("=> Using improved GAN:", args.improved_flag)
    print("Loading models")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args_json = utils.load_json(json_file=args.file_path)    
    from recovery import get_save_dir
    prefix = get_save_dir(args_json['train']['save_model_dir'],args)
    save_img_dir = "{}/imgs_{}/".format(prefix,args.loss)
    print('save_img_dir',save_img_dir)
    
    n_classes=args_json['dataset']['n_classes']
    z_dim = 100
    ###########################################
    ###########     load model       ##########
    ###########################################
    #target classifiers
    model_types_ = args_json['train']['model_types'].split(',')
    checkpoints = args_json['train']['cls_ckpts'].split(',')
    G,D = get_GAN(args_json['dataset']['d_priv'],args_json['dataset']['d_pub'],gan_type=args.dist_flag, 
                    gan_model_dir=args_json['train']['gan_model_dir'],
                    n_classes=n_classes,z_dim=100,target_model=model_types_[0],args=args)
    

    #evaluation classifier
    E = get_model(args_json['train']['eval_model'],n_classes,args_json['train']['eval_dir'])    
    dataset = args_json['dataset']['d_priv']
    E.eval()
    G.eval()
    save_dir = '{}{}/'.format(prefix,args.loss)
    # get_fea_target(E,'celeba_target_300ids')
    print('target done')
    
    fea_path,_ = gen_samples(G,E,dataset,save_dir,args.improved_flag,args.clipz,args.epoch)

    # (G,E,dataset,save_dir,improved_gan,clipz=True,epoch=2399):
    # fea_path,_ = gen_samples(G,E,save_dir,args.improved_flag)

    fea_path = fea_path + 'success.npy'

    save_dir = save_dir + "best_pairs/"
    
    os.makedirs(save_dir, exist_ok=True)
    find_closest_pairs(save_dir,'fea_target_300ids.npy', fea_path,'celeba_target_300ids.npy')
    return save_dir


def find_samples_demo(args):
    ############## get configs ##############
    
    args_json = load_json(json_file=args.configs)
    _, _, _, save_dir = get_save_dir(args_json['train']['save_model_dir'],args)

    ###########################################
    ###########     load model       ##########
    ###########################################
    
    print("Loading models")
    _, E, G, _, _, _, _ = loading(args,args_json)
    # save_dir = '{}/attack_demo/'.format(save_dir)
    
    fea_path,_ = gen_samples(G,E,save_dir,args.improved_flag,1,1)

    fea_path = fea_path + 'success.npy'

    save_dir = save_dir + "best_pairs/"
    
    os.makedirs(save_dir, exist_ok=True)
    find_closest_pairs(save_dir,'fea_target_300ids.npy', fea_path,'celeba_target_300ids.npy')
    return save_dir

