import argparse
import imp
from itertools import count

from utils import *
import torch
import numpy as np
import os

def get_image_from_iden(wanted_id, dataset,device):
    print(wanted_id)
    count = 0
    for _, (img, label) in enumerate(dataset):

        label = label.to(device)

        iter = label[:]==wanted_id
        if sum(iter)>0:
            
            # print(sum(iter))
            if count == 0:
                all_img =  torch.unsqueeze(img[iter][0],0)
            else:
                all_img = torch.cat((all_img, torch.unsqueeze(img[iter][0],0)),dim=0)
            # print('all_img',all_img.shape)
            count = count + 1
            # print(count)
            if count > 20:
                return all_img
    return all_img 
def get_images_from_iden(wanted_idens, dataset,device):
    imgs = torch.zeros(0)
    for iden in wanted_idens:
        img = get_image_from_iden(iden, dataset,device)
        imgs = torch.cat((imgs, img))

    return imgs

def get_image_from_iden2(wanted_id, dataset,device):
    print(wanted_id)
    count = 0
    for _, (img, label) in enumerate(dataset):
        label = label.to(device)

        iter = label[:]==wanted_id
        if sum(iter)>0:
            img_ = img[iter,:,:,:]
            # print('img_',img_.shape)
            # print(sum(iter))
            if count == 0:
                all_img = img_
            else:                
                all_img = torch.cat((all_img,img_))
            # print('all_img',all_img.shape)
            count = all_img.shape[0]
            print(count)
            if count > 50:
                return all_img
    return all_img 
if __name__ == "__main__":

    
   
    print("Loading models")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args_json = load_json(json_file='./config/celeba/classify.json')
    
    save_img_dir = "./GT/"
    
    os.makedirs(save_img_dir, exist_ok=True)
    
    dataset = ['train', 'test']
    img300ids= []
    iden =[]
    count = 0
    for data_ in dataset:
        _, dataloader_test = init_dataloader(args_json, args_json['dataset']['{}_file_path'.format(data_)], 100, mode="test")

            
        for _, (img, label) in enumerate(dataloader_test):
            label = label
            iter = label[:]<300
            if sum(iter)>0:
                img_ = img[iter,:,:,:]
                label_ = label[iter]
                if count == 0:
                    img300ids = img_
                    iden = label_
                    count = 1
                else:                
                    img300ids = torch.cat((img300ids,img_))
                    iden = torch.cat((iden,label_))
                print('all_img',img300ids.shape)
                
                print('label',iden.shape)

    np.save('{}_target_fid.npy'.format(dataset),img300ids)
    np.save('{}_target_fid_label.npy'.format(dataset),iden)          
    