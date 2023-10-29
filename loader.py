

from models.classifiers.classifier import *
from models.generators.resnet64 import ResNetGenerator,Generator,Generator_cond_wgp,Generator_cond_wgp2,Generator_cond_wgp3
from models.discriminators.snresnet64 import SNResNetProjectionDiscriminator,SNResNetConditionalDiscriminator,Discriminator_cond_wgp,SNResNetConditionalDiscriminator_vggface2
import torch.utils.data as data
import PIL

from torchvision import transforms
import torch
class ImageFolder(data.Dataset):
    def __init__(self, file_path, img_root, n_classes,dataset='celeba'):
        self.img_path = img_root	
        
        self.n_classes = n_classes
        self.dataset =dataset
        self.processor = self.get_processor()
        self.name_list, self.label_list = self.get_list(file_path) 
        self.label_list = np.array(self.label_list)

        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        print('self.num_img ',self.num_img )

    def get_processor(self):
        re_size =64
        if self.dataset=='celeba':
            crop_size = 108
            offset_height = (218 - crop_size) // 2
            offset_width = (178 - crop_size) // 2
        elif self.dataset == 'facescrub':
            crop_size = 108
            offset_height = (218 - crop_size) // 2
            offset_width = (178 - crop_size) // 2
        elif self.dataset == 'ffhq':
            # print('-------ffhq')
            crop_size = 88 #88
            offset_height = (128 - crop_size) // 2 
            offset_width = (128 - crop_size) // 2
        elif self.dataset == 'pubfig':
            
            crop_size = 67
            offset_height = (100 - crop_size) // 2
            offset_width = (100 - crop_size) // 2
                
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        proc = []
            
        proc.append(transforms.ToTensor())
        proc.append(transforms.Lambda(crop))

        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())
        
            
        return transforms.Compose(proc)
    
    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            img_name, iden = line.strip().split(' ')
            label_list.append(int(iden))
            name_list.append(img_name)
            

        return name_list, label_list


    def load_img(self):
        img_list = []
        
        # processer = self.get_processor()
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".png") or  img_name.endswith(".jpg") or  img_name.endswith(".jpeg") :
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert('RGB')
                # img_list.append(processer(img))
                img_list.append(img)
        return img_list


    def get_img_class(self, label):
        
        index = np.where(self.label_list == label)[0]
        iden = self.label_list[index]
        imgs,_ = self.__getitem__(index[0])
        imgs = imgs.unsqueeze(0)
        for i in range(1,index.shape[0],1):
            img,_= self.__getitem__(index[i])
            imgs = torch.cat((imgs,img.unsqueeze(0)),dim=0)
        return imgs, iden
    def __getitem__(self, index):
        # print('----index',index)
        processer = self.get_processor()

        img = self.image_list[index]
        img = processer(img)
        label = self.label_list[index]

        return img, label#,self.name_list[index]

    def __len__(self):
        return self.num_img

def load_gan(args,device,gen_ckpt_path=None,dis_ckpt_path=None):
    G = ResNetGenerator(
        args.gen_num_features, args.gen_dim_z, args.gen_bottom_width,
        activation=F.relu, num_classes=args.num_classes, distribution=args.gen_distribution
    ).to(device)

    D = SNResNetProjectionDiscriminator(args.dis_num_features, args.num_classes, F.relu).to(device)

    ######## load checkpoint
    if gen_ckpt_path!= None:
        gen_ckpt = torch.load(gen_ckpt_path)['model']
        G.load_state_dict(gen_ckpt)
        G = G.cuda()
    
    if dis_ckpt_path!= None:
        dis_ckpt = torch.load(dis_ckpt_path)['model']
        D.load_state_dict(dis_ckpt)
        D = D.cuda()


    return G,D     

def load_cgan(args,device,gen_ckpt_path=None,dis_ckpt_path=None):
    G = ResNetGenerator(
        args.gen_num_features, args.gen_dim_z, args.gen_bottom_width,
        activation=F.relu, num_classes=args.num_classes, distribution=args.gen_distribution
    ).to(device)

    # D = SNResNetConditionalDiscriminator_vggface2(args.dis_num_features, args.num_classes, F.relu).to(device)
    D = SNResNetConditionalDiscriminator(args.dis_num_features, args.num_classes, F.relu).to(device)

    ######## load checkpoint
    if gen_ckpt_path!= None:
        gen_ckpt = torch.load(gen_ckpt_path)['model']
        G.load_state_dict(gen_ckpt)
        G = G.cuda()
        # torch.save({'state_dict':gen_ckpt},gen_ckpt_path)
    
    if dis_ckpt_path!= None:
        dis_ckpt = torch.load(dis_ckpt_path)['model']
        
        # torch.save({'state_dict':dis_ckpt},dis_ckpt_path)
        D.load_state_dict(dis_ckpt)
        D = D.cuda()

    print('gen_ckpt_path',gen_ckpt_path)
    print('dis_ckpt_path',dis_ckpt_path)
    return G,D    

def load_cgan_wgan(args,device,gen_ckpt_path=None,dis_ckpt_path=None):
    print('???',args.num_classes)
    G = Generator_cond_wgp2(in_dim=args.gen_dim_z,num_classes=args.num_classes).to(device)
    # print(G)
    D = Discriminator_cond_wgp(3, num_classes=args.num_classes).to(device)
    # print(D)
    # exit()
    ######## load checkpoint
    if gen_ckpt_path!= None:
        gen_ckpt = torch.load(gen_ckpt_path)['model']
        G.load_state_dict(gen_ckpt)
        G = G.cuda()
        # torch.save({'state_dict':gen_ckpt},gen_ckpt_path)
    
    if dis_ckpt_path!= None:
        dis_ckpt = torch.load(dis_ckpt_path)['model']
        
        # torch.save({'state_dict':dis_ckpt},dis_ckpt_path)
        D.load_state_dict(dis_ckpt)
        D = D.cuda()

    print('gen_ckpt_path',gen_ckpt_path)
    print('dis_ckpt_path',dis_ckpt_path)
    return G,D    

def load_general_gan(args,device,gen_ckpt_path=None):
    G = Generator(100)
    
    G = torch.nn.DataParallel(G).to(device)
    ######## load checkpoint
    if gen_ckpt_path!= None:
        ckp_G = torch.load(gen_ckpt_path)
        G.load_state_dict(ckp_G['state_dict'], strict=True)
    return G
 
def load_classifier(model_name,nclass,path_T=None):
    print('------model_name',model_name,path_T)
    if model_name=="VGG16":
        model = VGG16(nclass)    
    if model_name=="MID":
        model = VGG16(nclass)   
    elif model_name=="densenet121":
        model = densenet121(nclass)  
    elif model_name=="densenet161":
        model = densenet161(nclass)  
    elif model_name=="densenet169":
        model = densenet169(nclass)  
    
    elif model_name=="convnext_tiny":
        model = Convnext_tiny(nclass)  
    elif model_name=="resnet50":
        model = ResNet50(nclass)  
    
    elif model_name=="FaceNet":
        model = FaceNet(nclass)
    elif "FaceNet64" in model_name:
        model = FaceNet64(nclass)
    elif model_name=="IR152":
        model = IR152(nclass)
   
            
    elif model_name =="BiDO":
        model = VGG16_defense_MI(nclass,True)   
           
    if model_name =="D":
        model  = SNResNetConditionalDiscriminator(64,nclass, F.relu)
        if path_T is not None:         
            ckp_T = torch.load(path_T)   
            model.load_state_dict(ckp_T['model'], strict=True)            
            model = torch.nn.DataParallel(model).cuda()
    else:        
        
        model = torch.nn.DataParallel(model).cuda()
        
        if path_T is not None:      
            ckp_T = torch.load(path_T)
            
            model.load_state_dict(ckp_T['state_dict'], strict=True)
            # torch.save({'state_dict':model.state_dict()},path_T)
            # exit()
            
            # print('ckp_T',ckp_T['state_dict'])
            # torch.save({'state_dict':ckp_T['state_dict']['state_dict']},path_T)
            # exit()
            # model.load_state_dict(ckp_T['state_dict'], strict=True)
            # model = torch.nn.DataParallel(model).cuda()
            # torch.save({'state_dict':model.state_dict()},'../target_model/target_ckp/ResNet50_sgq_50_1.tar')
        
        # a = model.state_dict()
        # print('a',a)
        # exit()
        # torch.save({'state_dict':model.state_dict()},path_T)
        # exit()
    return model
