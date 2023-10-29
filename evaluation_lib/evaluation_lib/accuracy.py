
from utils import *
import torch
import os
import numpy as np


import utils
from loader import *
# from attack import attack_acc
# from recovery import get_GAN
import statistics 

from evaluation_lib.fid import concatenate_list,gen_samples

Softmax = torch.nn.Softmax()

def attack_acc(fake,iden,E):
    # if dataset =='celeba' or dataset =='ffhq' or dataset =='facescrub':
        # print('attack_acc celeba')
    eval_prob = E(utils.low2high(fake))[-1]
    # else:
    #     eval_prob = E(fake)[-1]
    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
    
    cnt, cnt5 = 0, 0
    bs = fake.shape[0]
    # print('correct id')
    for i in range(bs):
        gt = iden[i].item()
        if eval_iden[i].item() == gt:
            cnt += 1
            # print(gt)
        _, top5_idx = torch.topk(eval_prob[i], 5)
        if gt in top5_idx:
            cnt5 += 1
    # print('cnt',cnt)
    return cnt*100.0/bs, cnt5*100.0/bs

def eval_acc(E,dataset,fake_dir):
    
    aver_acc, aver_acc5, aver_std, aver_std5 = 0, 0, 0, 0
 
    E.eval()
    N_check = 0
    for i in range(1):       
        all_fake = np.load(fake_dir,allow_pickle=True)  
        all_imgs = all_fake.item().get('imgs')
        all_label = all_fake.item().get('label')
        with torch.no_grad():          

            for random_seed in range(len(all_imgs)):
                if random_seed % 5 == 0:
                    res, res5 = [], []
                #################### attack accuracy #################
                fake = all_imgs[random_seed]
                label = all_label[random_seed]
                # label = torch.from_numpy(label)
                fake = torch.from_numpy(fake)

                acc,acc5 = attack_acc(fake,label,E)

                res.append(acc)
                res5.append(acc5)
                
                if (random_seed+1)%5 == 0:      
                    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
                    std = statistics.stdev(res)
                    std5 = statistics.stdev(res5)
                    print("Top1/Top5:{:.3f}/{:.3f}, std top1/top5:{:.3f}/{:.3f}".format(acc, acc_5, std, std5))

                    aver_acc += acc 
                    aver_acc5 += acc5 
                    aver_std += std 
                    aver_std5 +=  std5 
                
    N = len(all_imgs)/5
    aver_acc = aver_acc / N
    aver_acc5 = aver_acc5 / N
    aver_std = aver_std / N
    aver_std5 =  aver_std5 / N
    print('Final acc', aver_acc,aver_acc5)
    return aver_acc, aver_acc5, aver_std, aver_std5


def acc_class(filename,fake_dir,E):
    
    E.eval()

    sucessful_fake = np.load(fake_dir + 'success.npy',allow_pickle=True)  
    sucessful_imgs = sucessful_fake.item().get('sucessful_imgs')
    sucessful_label = sucessful_fake.item().get('label')
    sucessful_imgs = concatenate_list(sucessful_imgs)
    sucessful_label = concatenate_list(sucessful_label)

    N_img = 5
    N_id = 300
    with torch.no_grad():
        acc = np.zeros(N_id)
        for id in range(N_id):                
            index = sucessful_label == id
            acc[id] = sum(index)
            
    acc=acc*100.0/N_img 
    print('acc',acc)
    csv_file = '{}acc_class.csv'.format(filename)
    print('csv_file',csv_file)
    import csv
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        for i in range(N_id):
            # writer.writerow(['{}'.format(i),'{}'.format(acc[i])])
            writer.writerow([i,acc[i]])

def eval_acc_class(E,dataset,fake_dir,csv_file,n=1000,n_imgs=5.0):
    from evaluation_lib.fid import get_successful_samples    
    _,successful_id= get_successful_samples(fake_dir,E,dataset)
    freq = np.zeros(n)
    successful_id = torch.cat(successful_id)
    print(successful_id.shape)
    for i in range(n):
        index = successful_id ==i
        freq[i]=sum(index)
    
    import csv         
    
    with open(csv_file, 'w') as f:  
        writer = csv.writer(f)   
        for i in range(n) :
            writer.writerow([i,freq[i]*100.0/n_imgs])        
        
   
def get_prob(net,img,iden):
    _, logit_ = net(img)
    softmax = Softmax(logit_)
    N = iden.shape[0]
    prob = np.zeros(N)
    eval_iden = torch.argmax(logit_, dim=1).view(-1)
    
    is_success = np.zeros(N)
    for i in range(N): 
        prob[i]= softmax[i,iden[i]].item()    
        if eval_iden[i] == iden[i]:
            is_success[i] = 1
    # print('logits/N, prob/N',logits/N, prob/N)
    return prob,is_success,eval_iden
    

# def eval_prob(E,dataset,fake_dir,csv_file,n=1000,n_imgs=5.0):
    
#     T = get_model(model_types_[0],n_classes,checkpoints[0],dataset)
#     T = T.to(device)
#     # print('{}: {} params'.format(model_types_[id_], count_parameters(model)))
#     T = T.eval()
#     cid = [6,7,8]
#     for i in range(len(cid)):
#         id_ = int(cid[i])
#         print('Load classifier {}, ckpt {}'.format(model_types_[id_], checkpoints[id_]))
#         model = get_model(model_types_[id_],n_classes,checkpoints[id_],dataset)
#         model = model.to(device)
#         # print('{}: {} params'.format(model_types_[id_], count_parameters(model)))
#         model = model.eval()
#         if i==0:
#             targetnets = [model]
#         else:
#             targetnets.append(model)
    
#     # filename = "{}/{}_".format(prefix,args.loss)
#     fake = np.load(fake_dir + 'full.npy',allow_pickle=True)  
#     imgs = fake.item().get('imgs')
#     labels = fake.item().get('label')



#     # csv_file = 'prob_class_{}_{}_{}'.format(model_types_[0],args.loss,args.lam)
#     if len(cid)>1:
#         csv_file = csv_file + '_KD'
#     if args.improved_flag == True:
#         csv_file = csv_file + '_kedmi'
#     else:
        
#         csv_file = csv_file +'_gmi'
#     if args.same_z != '':
#         csv_file = csv_file +'_same_z'
#     print('csv_file',csv_file)
#     import csv
    
#     # with open(csv_file+'_6.csv', 'a') as f:
        
#     #     writer = csv.writer(f)
#     #     for i in range(len(imgs)):
#     #         img = torch.from_numpy(imgs[i]).cuda()
#     #         iden = labels[i]
#     #         # eval_prob,eval_success = get_prob(E,utils.low2high(img),iden)
#     #         target_prob,target_success = get_prob(T,img,iden)
#     #         eval_prob,eval_success= get_prob(targetnets[0],img,iden)

#     #         for j in range(img.shape[0]):
#     #             writer.writerow([iden[j],eval_prob[j],eval_success[j],target_prob[j],target_success[j]])
#     # exit()



#     # exit()
#     thr = 0.9
#     img_thr = []
#     iden_thr = []
    

#     os.makedirs(csv_file+'_{}'.format(thr), exist_ok=True)


#     _, dataloader_test = init_dataloader(args_json, args_json['dataset']['train_file_path'], 100, mode="test")
#     from get_target import get_images_from_iden
    
#     for i in range(len(imgs)):
#         img = torch.from_numpy(imgs[i]).cuda()
#         iden = labels[i]
#         eval_prob,eval_success = get_prob(E,utils.low2high(img),iden)
#         target_prob,target_success = get_prob(T,img,iden)
#         idx = target_prob - eval_prob >=thr

#         if sum(idx)>0:
#             print('idx',sum(idx))
#             eval_prob=eval_prob[idx]
#             target_prob=target_prob[idx]
#             eval_success=eval_success[idx]
#             target_success=target_success[idx]
#             iden=iden[idx]
#             img=img[idx]
            
#             img_all = torch.cat((img,img),axis=0)
#             print('img',img.shape,img_all.shape)

#             img2 = get_images_from_iden(iden,dataloader_test,'cuda:0')
#             img_all = torch.cat((img2.cuda(),img),axis=0)
#             print('img',img.shape,img_all.shape)
#             img_thr.append(img)
#             iden_thr.append(iden)
#             save_tensor_images(img_all,'./'+csv_file+'_{}/{}.png'.format(thr,i),nrow= img.shape[0])
#         exit()
#     ###
    
#     ####baseline
#     args.classid='0'
#     args.same_z=''
#     prefix = get_save_dir(args_json['train']['save_model_dir'],args)
#     save_dir = '{}{}/'.format(prefix,args.loss)
#     print('save_dir',save_dir)
#     img_path2,_ = gen_samples(G,E,dataset,save_dir,args.improved_flag)

#     fake2 = np.load(img_path2 + 'full.npy',allow_pickle=True)  
#     imgs2 = fake2.item().get('imgs')
#     labels2 = fake2.item().get('label')
    
#     for i in range(len(imgs)):
#         img = torch.from_numpy(imgs[i]).cuda()
#         img2 = torch.from_numpy(imgs2[i]).cuda()
#         iden = labels[i]
#         eval_prob,eval_success = get_prob(E,utils.low2high(img),iden)
#         target_prob,target_success = get_prob(T,img,iden)

#         eval_prob2,eval_success2 = get_prob(E,utils.low2high(img2),iden)
#         target_prob2,target_success2 = get_prob(T,img2,iden)
        
#         idx = eval_success - eval_success2 ==0
#         print('-eval_success',eval_success,sum(eval_success))
#         print('-eval_success2',eval_success2,sum(eval_success2))
#         #KD
#         eval_prob=eval_prob[idx]
#         target_prob=target_prob[idx]
#         eval_success=eval_success[idx]
#         target_success=target_success[idx]
#         iden=iden[idx]
#         img=img[idx]
#         #baseline
#         eval_prob2=eval_prob2[idx]
#         target_prob2=target_prob2[idx]
#         eval_success2=eval_success2[idx]
#         target_success2=target_success2[idx]
#         iden2=iden
#         img2=img2[idx]
        
#         img_all = torch.cat((img2,img),axis=0)
        
#         print('img',img.shape,img2.shape,img_all.shape)
#         img3 = get_images_from_iden(iden,dataloader_test,'cuda:0')
#         img_all = torch.cat((img3.cuda(),img_all),axis=0)
#         print('img',img.shape,img_all.shape)
#         img_thr.append(img)
#         iden_thr.append(iden)
#         save_tensor_images(img_all,'./'+csv_file+'/{}_3.png'.format(i),nrow= img.shape[0])

#         exit()




#     print('csv_file',csv_file)


def plot_z(args):
    print("=> Using improved GAN:", args.improved_flag)
    print("Loading models")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args_json = load_json(json_file=args.file_path)

    
    now = datetime.now() # current date and time
    from recovery import get_save_dir
    prefix = get_save_dir(args_json['train']['save_model_dir'],args)
    save_img_dir = "{}/imgs_{}/".format(prefix,args.loss)
    print('save_img_dir',save_img_dir)
    log_path = "{}/invertion_logs".format(prefix)
    

    log_file = "invertion_logs_{}_{}.txt".format(args.loss,now.strftime("%m_%d_%Y_%H_%M_%S"))
    # utils.Tee(os.path.join(log_path, log_file), 'w')
    
    n_classes=args_json['dataset']['n_classes']
    
    dataset = args_json['dataset']['name']
    z_dim = 100
    ###########################################
    ###########     load model       ##########
    ###########################################
    #target classifiers
    model_types_ = args_json['train']['model_types'].split(',')
    checkpoints = args_json['train']['cls_ckpts'].split(',')
    cid = args.classid.split(',')

    G,D = get_GAN(args_json['dataset']['name'],args_json['dataset']['gan'],gan_type=args.dist_flag, 
                    gan_model_dir=args_json['train']['gan_model_dir'],
                    n_classes=n_classes,z_dim=100,target_model=model_types_[0],args=args)
    

    print('model_types_',model_types_)
    print('checkpoint',checkpoints)
    T = get_model(model_types_[0],n_classes,checkpoints[0],dataset)
    T = T.to(device)
    T = T.eval()

    
    #evaluation classifier
    E = get_model(args_json['train']['eval_model'],n_classes,args_json['train']['eval_dir'])    
    G.eval()
    E.eval()

    save_dir = '{}{}/'.format(prefix,args.loss)

    target_id=[7,11,12,19]
    from fid import get_z,concatenate_list
    z_all = []
    iden =[]

    import matplotlib.pyplot as plt

    from sklearn.decomposition import PCA


    pca = PCA(n_components=2)
    
    for loop in range(1):
            for i in range(5): #300 ides 
                for j in range(30): #30 images/iden
                    z, id_ = get_z(args.improved_flag,save_dir,loop,i,j)

                    z_all.append(z.detach().cpu().numpy())
                    iden.append(id_)
    z_all=concatenate_list(z_all)
    iden=concatenate_list(iden)
    color =['red','blue','black','green']
    thr = 0.6
    for i in range(60):
        # id = target_id[i]
        id=i
        index = iden == id
        
        z = z_all[index,:]
        
        pca.fit(z)
        z_pca = pca.fit_transform(z)
        print('---',z_pca.shape, sum(index))
        z = torch.from_numpy(z).cuda()
        fake = G(z)
        eval_prob,eval_success = get_prob(E,utils.low2high(fake),iden[index])
        target_prob,target_success = get_prob(T,fake,iden[index])
        
        plt.figure(i)
        for j in range(sum(index)):
            if eval_success[j] == 1 and target_success[j] == 1:
                if target_prob[j] - eval_prob[j]>= thr:
                    color_label= 1
                    label = 'target - eval >= {}'.format(thr)
                else:
                    color_label=2
                    label = 'target - eval < {}'.format(thr)
            elif eval_success[j]== 1:
                color_label=3
                label = 'unsuccessful samples'
            else:
                color_label=4
                
                label = 'unsuccessful samples for both'

        # print('z_pca',z_pca)
        # for k in range(z.shape[0]):
        #     # plt.scatter(z_pca[0],z_pca[1],c=color[color_label[i]],label=color_label[i],alpha=0.2)
            plt.scatter(z_pca[j, 0],z_pca[j, 1],c=color[color_label-1],label=label,alpha=0.2)

        plt.title('KD')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        filename='kd/{}.png'.format(i)
        plt.savefig(filename)


def acc_list(args):      
    print("=> Using improved GAN:", args.improved_flag)
    print("Loading models")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args_json = load_json(json_file=args.file_path)

    
    now = datetime.now() # current date and time
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
    G,D = get_GAN(args_json['dataset']['name'],args_json['dataset']['gan'],gan_type=args.dist_flag, 
                    gan_model_dir=args_json['train']['gan_model_dir'],
                    n_classes=n_classes,z_dim=100,target_model=model_types_[0],args=args)
    

    print('model_types_',model_types_)
    print('checkpoint',checkpoints)
    for i in range(1): # get only target classifier
        print(model_types_[i], checkpoints[i])
        model = get_model(model_types_[i],n_classes,checkpoints[i])
        model = model.to(device)
        model = model.eval()
        if i==0:
            targetnets = [model]
        else:
            targetnets.append(model)

    #evaluation classifier
    E = get_model(args_json['train']['eval_model'],n_classes,args_json['train']['eval_dir'])    
    G.eval()
    E.eval()
    dataset = args_json['dataset']['name']
    
    save_dir = '{}{}/'.format(prefix,args.loss)
    seed = 9
    #7 1380
    torch.manual_seed(seed)
    E.eval()
    from fid import get_z
    import csv 
    import os  
    csv_file = 'eval_10k{}_{}.csv'.format(dataset,model_types_[0])
    acc =[]
    
    # target_dir='fea_target_300ids.npy'
    # target = np.load(target_dir,allow_pickle=True)  
    # target_fea = target.item().get('fea')    
    # target_y = target.item().get('label')

    for epoch in range(0,10001,200):
        
        sucessful_attack = 0
        total_gen = 0
        for loop in range(1):
            for i in range(5): #300 ides 
                
                for j in range(5): #30 images/iden
                    if epoch == 0:
                        z, iden = get_z(args.improved_flag,save_dir,loop,i,j,epoch)
                    else:
                        z, iden = get_z(args.improved_flag,save_dir,loop,i,j,epoch-1)
                    if args.clipz==True:
                        z = torch.clamp(z,-1.0,1.0).float()
                    total_gen = total_gen + z.shape[0]
                    # calculate attack accuracy
                    with torch.no_grad():
                        fake = G(z.cuda())
                        print('save')
                        save_tensor_images(fake, os.path.join(save_dir, "gen_{}_{}.png".format(i,j)), nrow = 60)

                        if dataset =='celeba' or dataset =='ffhq' or dataset =='facescrub':
                            _,eval_prob = E(utils.low2high(fake))
                        else:
                            _,eval_prob = E(fake)
                              
                        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                        for id in range(iden.shape[0]):
                            if eval_iden[id]==iden[id]:
                                sucessful_attack +=1

                        
        acc.append(sucessful_attack*100.0/total_gen)
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        for i in range(len(acc)):
            fields = ['{}'.format(i*200), '{}'.format(acc[i])]
            writer.writerow(fields)
        
