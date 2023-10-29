import numpy as np
import os
import torch
from PIL import Image

import utils

import torch.utils.data as data

def sample_from_data(args, device, data_loader):
    """Sample real images and labels from data_loader.

    Args:
        args (argparse object)
        device (torch.device)
        data_loader (DataLoader)

    Returns:
        real, y

    """

    real, y = next(data_loader)
    real, y = real.to(device), y.to(device)

    return real, y

def sample_from_data_wo_label(device, data_loader):
    """Sample real images and labels from data_loader.

    Args:
        args (argparse object)
        device (torch.device)
        data_loader (DataLoader)

    Returns:
        real, y

    """

    # real,_ = next(data_loader)
    real = next(data_loader)
    real = real.to(device)

    return real

def sample_from_gen_with_pseudo_y(args, device, gen,pseudo_y):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        gen (nn.Module)

    Returns:
        fake,  z

    """

    z = utils.sample_z(
        pseudo_y.shape[0], args.gen_dim_z, device, args.gen_distribution
    )
    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z

def sample_from_data_w_label(device, data_loader):
    """Sample real images and labels from data_loader.

    Args:
        args (argparse object)
        device (torch.device)
        data_loader (DataLoader)

    Returns:
        real, y

    """

    real,label = next(data_loader)
    real = real.to(device)
    label = label.to(device)
    return real,label


def sample_from_gen(args, device, num_classes, gen,batch_size):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        num_classes (int): for pseudo_y
        gen (nn.Module)

    Returns:
        fake, pseudo_y, z

    """

    z = utils.sample_z(
        batch_size, args.gen_dim_z, device, args.gen_distribution
    )
    pseudo_y = utils.sample_pseudo_labels(
        num_classes, batch_size, device
    )

    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z



def sample_from_gen_with_pseudo_y(args, device,  gen,pseudo_y):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        gen (nn.Module)

    Returns:
        fake,  z

    """

    z = utils.sample_z(
        pseudo_y.shape[0], args.gen_dim_z, device, args.gen_distribution
    )

    # pseudo_y2 = utils.sample_pseudo_labels(
    #     num_classes, args.batch_size, device
    # )
    # pseudo_y2[:] = pseudo_y.to(torch.int)
    # print('--pseudo_y',pseudo_y.to(torch.int))
    # print('--pseudo_y2',pseudo_y2)
    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z


def sample_from_gen_test(args, device, num_classes, gen):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        num_classes (int): for pseudo_y
        gen (nn.Module)

    Returns:
        fake, pseudo_y, z

    """

    z = utils.sample_z(
        num_classes, args.gen_dim_z, device, args.gen_distribution
    )
    pseudo_y =  torch.from_numpy(np.arange(num_classes))
    pseudo_y = pseudo_y % args.num_classes
    pseudo_y = pseudo_y.type(torch.long).to(device)


    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z

def sample_from_gen_with_targetID(args, device, num_classes, gen,ID,batch_size):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        num_classes (int): for pseudo_y
        gen (nn.Module)

    Returns:
        fake, pseudo_y, z

    """

    z = utils.sample_z(
        batch_size, args.gen_dim_z, device, args.gen_distribution
    )
    pseudo_y = utils.sample_pseudo_labels(
        num_classes, batch_size, device
    )
    pseudo_y[:] = ID

    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, args, root='', num_classes =1000, transform=None):
        super(FaceDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.images = []
        self.label = []
        

        self.path = self.root

        # num_classes = len([lists for lists in os.listdir(
        #     self.path) if os.path.isdir(os.path.join(self.path, lists))])
        filename = root + '/dataset.npy'
        print('filename',filename)
        if os.path.isfile(filename):
            data = np.load(filename, allow_pickle=True)
            self.images = data.item().get('x')
            self.label = data.item().get('y')
            self.label = torch.from_numpy(self.label).to(torch.long)
        else:
            for idx in range(num_classes):
                print('i',idx)
                class_path = os.path.join(self.path, str(idx))
                for _, _, files in os.walk(class_path):
                    for img_name in files:
                        image_path = os.path.join(class_path, img_name)
                        image = Image.open(image_path)
                        if args.data_name == 'facescrub':
                            if image.size != (64, 64):
                                image = image.resize((64, 64), Image.ANTIALIAS)
                        self.images.append(self.transform(image.copy()))
                        self.label.append(idx)
                        image.close()
            np.save(filename,{'x':self.images,'y':self.label})
        # exit()

    def __getitem__(self, index):
        img = self.images[index]
        label = self.label[index]
        # if self.transform != None:
        #     img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L5-L15
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L18-L26
class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31



class generated_dataset(data.Dataset):
    def __init__(self, gen_data):#,if_remove_top5=False,num_classes=1000):

        generated_dataset = torch.load(gen_data)
        self.label = generated_dataset['y']
        self.data = generated_dataset['z']
        self.pseudo = generated_dataset['gen_pseudo_y']
        
        self.num_img = self.data.shape[0]
        print('generated dat {}, num_img={}'.format(gen_data,self.num_img))


    def __getitem__(self, index):
        data_ = self.data[index]
        label = self.label[index]
        pseudo = self.pseudo[index]
        return data_,label,pseudo

    def __len__(self):
        return self.num_img