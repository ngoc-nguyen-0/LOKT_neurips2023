from tqdm import tqdm
import torch
import numpy as np
from scipy import linalg
from metrics import metric_utils
import utils
# from attack import reparameterize
from loader import *
import os
device = 'cuda:0'

# from kornia import augmentation

_feature_detector_cache = None
def get_feature_detector():
    global _feature_detector_cache
    if _feature_detector_cache is None:
        _feature_detector_cache = metric_utils.get_feature_detector(
            'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/'
            'metrics/inception-2015-12-05.pt', device)

    return _feature_detector_cache


def postprocess(x):
    """."""
    return ((x * .5 + .5) * 255).to(torch.uint8)


def run_fid(x1, x2):
    # Extract features
    x1 = run_batch_extract(x1, device)
    x2 = run_batch_extract(x2, device)

    npx1 = x1.detach().cpu().numpy()
    npx2 = x2.detach().cpu().numpy()
    mu1 = np.mean(npx1, axis=0)
    sigma1 = np.cov(npx1, rowvar=False)
    mu2 = np.mean(npx2, axis=0)
    sigma2 = np.cov(npx2, rowvar=False)
    frechet = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return frechet


def run_feature_extractor(x):
    assert x.dtype == torch.uint8
    assert x.min() >= 0
    assert x.max() <= 255
    assert len(x.shape) == 4
    assert x.shape[1] == 3
    feature_extractor = get_feature_detector()
    return feature_extractor(x, return_features=True)


def run_batch_extract(x, device, bs=500):
    z = []
    with torch.no_grad():
        for start in tqdm(range(0, len(x), bs), desc='run_batch_extract'):
            stop = start + bs
            x_ = x[start:stop].to(device)
            z_ = run_feature_extractor(postprocess(x_)).cpu()
            z.append(z_)
    z = torch.cat(z)
    return z


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, return_details=False):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    if not return_details:
        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
    else:
        t1 = diff.dot(diff)
        t2 = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return (t1 + t2), t1, t2


def get_z(save_dir,loop,i,epoch=600,bs = 60):

    outputs_z = "./{}/all_z_{}_{}_{}_{}.pt".format(save_dir,epoch,i*bs,loop,epoch-1)
    # outputs_z = "./{}/all_z{}_{}_{}.pt".format(save_dir,i*bs,loop,epoch-1)
    print('---outputs_z',outputs_z)
    z = torch.load(outputs_z)  
    z = z['z']
    iden = torch.from_numpy(np.arange(bs)) + bs*i
    # print('-iden',iden)
    # input()
    return z, iden
import os
from utils import save_tensor_images
def gen_samples(G,E,dataset,save_dir,Ns=-1,epoch=600):
    total_gen = 0
    
    all_imgs = []                            
    all_fea = []
    all_id = []
    
    E.eval()
    n_img = 5
    if dataset == 'celeba':
        if Ns<0:
            N = 5
        else:
            N = Ns

        n_img = 5
        bs = 60
    elif dataset == 'pubfig':
        N=1
        bs = 50
    
    elif dataset == 'facescrub':
        if Ns<0:
            N = 4
        else:
            N = Ns
        bs = 50

    img_ids_path = './{}/attack{}_e{}_{}_{}.npy'.format(save_dir,epoch, dataset,N,n_img)
    if not os.path.exists(img_ids_path):
        for loop in range(1):
            for i in range(N): #300 ides 
                for j in range(n_img): #5 images/iden
                    z, iden = get_z(save_dir,j,i,epoch,bs)
                    total_gen = total_gen + z.shape[0]
                    # calculate attack accuracy
                    with torch.no_grad():
                        fake = G(z.cuda(),iden.cuda())
                       
                        eval_fea,_ =  E(utils.low2high(fake))

                        
                        fake = fake.detach().cpu().numpy()
                        eval_fea = eval_fea.detach().cpu().numpy()  
                    
                        all_imgs.append(fake)
                        all_fea.append(eval_fea)
                        all_id.append(iden)  
        np.save(img_ids_path,{'imgs':all_imgs,'label':all_id,'fea':all_fea})
                
        
    return img_ids_path,total_gen


def get_successful_samples(fake_dir,E,dataset):
    
    all_fake = np.load(fake_dir,allow_pickle=True)  
    all_imgs = all_fake.item().get('imgs')
    all_label = all_fake.item().get('label')
    
    all_sucessful_imgs = []
    all_sucessful_id =[]
    E.eval()
    with torch.no_grad():
        
        for random_seed in range(len(all_imgs)):
            fake = all_imgs[random_seed]
            iden = all_label[random_seed]
            
            fake = torch.from_numpy(fake)
            _,eval_prob =   E(utils.low2high(fake))    
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
            sucessful_iden = []
            # failure_iden = []
            for id in range(iden.shape[0]):
                if eval_iden[id]==iden[id]:
                    sucessful_iden.append(id)

            fake = fake.detach().cpu().numpy()
        
            all_imgs.append(fake)
                
            if len(sucessful_iden)>0:                              
                sucessful_iden = np.array(sucessful_iden)                            
                sucessful_fake = fake[sucessful_iden,:,:,:]                    
                # sucessful_eval_fea = eval_fea[sucessful_iden,:]
                sucessful_iden = iden[sucessful_iden]
            else:
                # sucessful_fake = -1
                # sucessful_iden = -1
                # sucessful_eval_fea = -1
                sucessful_fake = []
                sucessful_iden = []
                # sucessful_eval_fea = []
            
            all_sucessful_imgs.append(sucessful_fake)
            all_sucessful_id.append(sucessful_iden)
    return all_sucessful_imgs,   all_sucessful_id   
def gen_samples2(G,E,dataset,save_dir,improved_gan):
    total_gen = 0
    img_ids_path = save_dir + 'attack_300ids'
    all_sucessful_imgs = []
    all_failure_imgs = []
    if not os.path.exists(img_ids_path+'full.npy'):
        for loop in range(1):
            for i in range(5): #300 ides 
                for j in range(5): #30 images/iden
                    z, iden = get_z(improved_gan,save_dir,loop,i,j)
                    total_gen = total_gen + z.shape[0]
                    # calculate attack accuracy
                    with torch.no_grad():
                        fake = G(z.cuda())
                        if dataset =='celeba' or dataset =='ffhq' or dataset =='facescrub':
                            eval_fea,eval_prob =  E(augmentation.Resize((112, 112))(fake))
                        else:
                            eval_fea,eval_prob = E(fake)
                        
                        ### successfully attacked samples       
                        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                        sucessful_iden = []
                        failure_iden = []
                        for id in range(iden.shape[0]):
                            if eval_iden[id]==iden[id]:
                                sucessful_iden.append(id)
                            else:
                                failure_iden.append(id)
                        fake = fake.detach().cpu().numpy()
                        eval_fea = eval_fea.detach().cpu().numpy()  
                        if i==0 and j==0:
                            all_imgs = fake                            
                            all_fea = eval_fea
                            all_id = iden
                        else:
                            all_imgs = np.concatenate((fake,all_imgs))
                            all_fea = np.concatenate((eval_fea,all_fea))
                            all_id = np.concatenate((iden,all_id))
                        if len(sucessful_iden)>0:                              
                            sucessful_iden = np.array(sucessful_iden)                            
                            sucessful_fake = fake[sucessful_iden,:,:,:]                    
                            sucessful_eval_fea = eval_fea[sucessful_iden,:]
                            sucessful_iden = iden[sucessful_iden]

                            if len(all_sucessful_imgs) == 0:
                                all_sucessful_imgs = sucessful_fake                            
                                all_sucessful_fea = sucessful_eval_fea
                                all_sucessful_id = sucessful_iden                                
                            else:  
                                all_sucessful_imgs = np.concatenate((sucessful_fake,all_sucessful_imgs))
                                all_sucessful_id = np.concatenate((sucessful_iden,all_sucessful_id))                            
                                all_sucessful_fea = np.concatenate((sucessful_eval_fea,all_sucessful_fea))
                        if len(failure_iden)>0: 
                            failure_iden = np.array(failure_iden)
                            failure_fake = fake[failure_iden,:,:,:]                    
                            failure_eval_fea = eval_fea[failure_iden,:]
                            failure_iden = iden[failure_iden]
                            if len(all_failure_imgs) == 0:

                                all_failure_imgs = failure_fake                            
                                all_failure_fea = failure_eval_fea
                                all_failure_id = failure_iden                                
                            else:           
                                all_failure_imgs = np.concatenate((failure_fake,all_failure_imgs))
                                all_failure_id = np.concatenate((failure_iden,all_failure_id))                            
                                all_failure_fea = np.concatenate((failure_eval_fea,all_failure_fea))

        np.save(img_ids_path+'full',{'imgs':all_imgs,'label':all_id,'fea':all_fea})
        np.save(img_ids_path+'success',{'sucessful_imgs':all_sucessful_imgs,'label':all_sucessful_id,'sucessful_fea':all_sucessful_fea})
        np.save(img_ids_path+'failure',{'failure_imgs':all_failure_imgs,'label':all_failure_id,'failure_fea':all_failure_fea})

    return img_ids_path,total_gen

def concatenate_list(listA):
    result = []
    for i in range(len(listA)):
        val = listA[i]
        if len(val)>0:
            if len(result)==0:
                result = listA[i]
            else:
                result = np.concatenate((result,val))
    return result

def eval_fid(E,dataset,fake_dir):
    #real data
    target_x = np.load('./metadata/{}_target_fid.npy'.format(dataset))

    # Load Samples
    print(fake_dir)
    
    fake,_= get_successful_samples(fake_dir,E,dataset)
    fake = concatenate_list(fake)

    print('correct samples {}'.format(fake.shape[0]))
    fake = torch.from_numpy(fake).cuda()
    target_x = torch.from_numpy(target_x).cuda()

    # FID
    # print('Fake ={} samples, Rea',fake.shape,target_x.shape)
    fid = run_fid(target_x, fake)
    print("FID:{:.3f}".format(fid))

    return fid, fake.shape[0]
        