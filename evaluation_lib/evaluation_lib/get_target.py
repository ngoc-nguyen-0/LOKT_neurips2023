import argparse
import imp

from utils import *
import torch
import numpy as np
import os
import utils
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
                return all_img
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
                return img_[0]
            else:                
                all_img = torch.cat((all_img,img_))
            # print('all_img',all_img.shape)
            count = all_img.shape[0]
            print(count)
            if count > 50:
                return all_img
    return all_img 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch GMI')
    
    parser.add_argument('--iden', type=str, default='295,75,141,201,211,224,243,284,289')
   
    args,_ = parser.parse_known_args()

   
    print("Loading models")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args_json = load_json(json_file='./config/celeba/celeba_VGG16.json')
    
    save_img_dir = "./GT/"
    
    os.makedirs(save_img_dir, exist_ok=True)
    iden = args.iden.split(',')
    dataset = ['train']#, 'test']
    for data_ in dataset:
        _, dataloader_test = init_dataloader(args_json, args_json['dataset']['{}_file_path'.format(data_)], 100, mode="test")
    
        for i in range(len(iden)):
            print('Wanted id: {}'.format(iden[i]))
            img = get_image_from_iden2(int(iden[i]),dataloader_test,'cuda:0')
            img=torch.unsqueeze(img, dim=0)
            print('img',img.shape)
            img = utils.low2high(img)
            save_path ="{}{}_private".format(save_img_dir,iden[i])
            # np.save(save_path,img)
            save_tensor_images(img, save_path+ ".png", nrow = 60)
        # for i in range(170,300,1):#len(iden)):
        #     print('Wanted id: {}'.format(i))
        #     img = get_image_from_iden2(int(i),dataloader_test,'cuda:0')
        #     save_path ="{}{}_{}".format(save_img_dir,data_,i)
        #     np.save(save_path,img)
        #     img = utils.low2high(img)
        #     save_tensor_images(img, save_path+ ".png", nrow = 60)
    # _, dataloader_test = init_dataloader(args_json, args_json['dataset']['{}_file_path'.format(data_)], 100, mode="test")
    # bs=60
    # _, dataloader_test = init_dataloader(args_json, args_json['dataset']['train_file_path'], 100, mode="test")
    
    # for i in range(1):
    #     iden = torch.from_numpy(np.arange(bs))

    #     for idx in range(5):
    #         print('iden',iden)    
    #         img = get_images_from_iden(iden,dataloader_test,'cuda:0')
    #         img = utils.low2high(img)
    #         print('img',img.shape)
    #         print(os.path.join(save_img_dir, "GT_{}.png".format(idx)))
    #         save_tensor_images(img.detach(), os.path.join(save_img_dir, "GT_{}.png".format(idx)), nrow = 60)
    #         iden +=bs