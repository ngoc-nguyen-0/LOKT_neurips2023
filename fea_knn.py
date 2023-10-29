
from utils import init_dataloader,load_json

args_json = load_json(json_file='./config/dataset/ffhq_vggface2.json')
    

from inception_resnet_v1 import InceptionResnetV1
import torch
device = 'cuda'

E = InceptionResnetV1(classify=True, pretrained='vggface2').to(device)

_, dataloader_test = init_dataloader(args_json, './data_files/vggface2/testset.txt', 200, mode="test",attackid=1000)

all_labels = []
all_features = []

from kornia import augmentation
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True
with torch.no_grad():
    for i, (img, label) in enumerate(dataloader_test):
    
        print(i,len(dataloader_test))
        # index_i = label[:]<1000
        # if sum(index_i) > 0:
        x = img.to(device)
        # x = img[index_i]
        new_x = augmentation.Resize((160, 160))(x)
        fea,_ = E(new_x)
        # print('fea',fea.shape)
        all_features.append(fea.cpu())
        all_labels.append(label.cpu())

all_labels = torch.cat(all_labels)
all_features = torch.cat(all_features)
print(all_labels.shape)
print(all_features.shape)
import numpy as np
np.save('./metadata/vggface2_fea_target_knn.npy',{'label':all_labels.detach().cpu().numpy() ,'fea':all_features.detach().cpu().numpy()})


  
        

