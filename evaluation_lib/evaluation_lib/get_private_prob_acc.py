
import KD_students
from cosine import get_prob
import torch
import numpy as np
from skimage import io
from utils import save_tensor_images
eval_model ='VGG16'
# eval_dir ='./target_model/target_ckp/FaceNet_95.88.tar'
eval_dir ='./target_model/target_ckp/VGG16_88.26.tar'
n_classes = 1000
E = KD_students.get_model(eval_model,n_classes,eval_dir) 



Softmax = torch.nn.Softmax()
def get_prob(net,img,iden):
    _, logit_ = net(img)
    softmax = Softmax(logit_)
    N = img.shape[0]
    prob = np.zeros(N)
    eval_iden = torch.argmax(logit_, dim=1).view(-1)
    
    is_success = np.zeros(N)
    
    succ_prob = []
    for i in range(N): 
        prob[i]= softmax[i,iden].item()    
        if eval_iden[i] == iden:
            is_success[i] = 1
            succ_prob.append(prob[i])
    # print('logits/N, prob/N',logits/N, prob/N)
    acc = sum(is_success)*100.0/img.shape[0]
    print(sum(is_success),img.shape[0])
    # succ_prob = np.concatenate(succ_prob)
    if len(succ_prob)> 0:
        mean_prob = sum(succ_prob)/len(succ_prob)
    else:
        mean_prob = 0
    return mean_prob,acc

from torchvision import transforms
def get_processor():
    re_size = 64


    crop_size = 108
    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    
            
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

    proc = []
    proc.append(transforms.ToTensor())
    proc.append(transforms.Lambda(crop))
    proc.append(transforms.ToPILImage())
    proc.append(transforms.Resize((re_size, re_size)))
    proc.append(transforms.ToTensor())
    
    
        
    return transforms.Compose(proc)

import csv
csv_file =   'private_{}.csv'.format(eval_model)
processer = get_processor()

from utils import init_dataloader
dataset_test, _ = init_dataloader(args_json, args_json['dataset']['train_file_path'], 64, mode="test")
with open(csv_file, 'w') as f:  
    writer = csv.writer(f)
    for id in range(300):
        with torch.no_grad():

            path1='./GT_64x64/train_{}.npy'.format(id)

            img = np.load(path1,allow_pickle=True)  
            img = processer(img)
            img = torch.from_numpy(img).cuda()

            print(id,img.shape)
            mean_prob,acc = get_prob(E,img,id)
            writer.writerow(["{}".format(id),"{}".format(mean_prob),"{}".format(acc)])


        # l_id = img.item().get('l_id')
        # l_id_target = img.item().get('l_id_target')
        # # print(l_id)              
        # writer = csv.writer(f)
        # for i in range(l_id.shape[0]):
        #     if l_id_target[i]< 0.01 and l_id[i]>1:
        #         print(id,i,l_id[i],l_id_target[i])
        #         img_ = img.item().get('img')
        #         img_ = img_[i]
        #         img_ = torch.from_numpy(img_).cuda()
        #         save_tensor_images(img_, save_path+ "id{}_{}.png".format(id,i))
        #         writer.writerow(["{}".format(id),"{}".format(i),"{}".format(l_id[i]),"{}".format(l_id_target[i])])


