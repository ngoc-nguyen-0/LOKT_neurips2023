
from utils import *
from classify import *
from generator import *
from discri import *
import torch
import time
import numpy as np


import KD_students

import statistics 
torch.manual_seed(9)
from recovery import prepare_parser
def test_class(model, dataset,nclass=300):
    tf = time.time()
    model.eval()
    acc = np.zeros(nclass)
    
    acc5 = np.zeros(nclass)
    mean_prob = np.zeros(nclass)
    std = np.zeros(nclass)
    with torch.no_grad():
        for id in range(nclass):  
            print(id)
            img, iden = dataset.get_img_class(id)

            img = img.to(device)
            iden = torch.from_numpy(iden).to(device)

            bs = img.size(0)
            iden = iden.view(-1)
            _,out_prob = model(img)
            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            acc[id]= torch.sum(iden == out_iden).item()/bs

            # print(out_iden)
            # print(out_prob)
            _, top5 = torch.topk(out_prob,5, dim = 1)
            for ind,top5pred in enumerate(top5):
                if iden[ind] in top5pred:
                    acc5[id] += 1.0
            acc5[id] = acc5[id]/bs
            succ_prob = []
            
            prob = np.zeros(bs)
            
            Softmax = torch.nn.Softmax()
            
            softmax = Softmax(out_prob)
            for i in range(bs): 
                print(i, softmax[i,id])
                prob[i]= softmax[i,id].item()    
                # if out_iden[i] == id:
                succ_prob.append(prob[i])
           
            if len(succ_prob)> 0:
                mean_prob[id] = sum(succ_prob)/len(succ_prob)
                std[id] =  statistics.stdev(succ_prob)
            
    return acc*100,acc5*100,mean_prob,std

if __name__ == "__main__":
    global args, logger

    parser = prepare_parser()
    
    parser.add_argument('--model2', default='', type=str, help='model')
    parser.add_argument('--ckpt2', default='', type=str, help='ckpt')
    args = parser.parse_args()

    print("=> Using improved GAN:", args.improved_flag)
    print("Loading models")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args_json = load_json(json_file=args.file_path)
    
    n_classes=args_json['dataset']['n_classes']
    z_dim = 100
    ###########################################
    ###########     load model       ##########
    ###########################################
    if args.model2 =='':
        model_types_ = args_json['train']['model_types'].split(',')
        checkpoints = args_json['train']['cls_ckpts'].split(',')
        
        cid = args.classid.split(',')
    # dataset_test, _ = init_dataloader(args_json, args_json['dataset']['test_file_path'], 64, mode="test")
    dataset_test, _ = init_dataloader(args_json, args_json['dataset']['train_file_path'], 64, mode="test")
    from KD_students import test
    import csv
    

    dataset = args_json['dataset']['d_priv']
    
    #target and student classifiers
    fea_mean = []
    fea_logvar = []
    targetnets_diff = []
    for i in range(len(cid)):
        id_ = int(cid[i])
        print('Load classifier {}, ckpt {}'.format(model_types_[id_], checkpoints[id_]))
        # exit()
        model = KD_students.get_model(model_types_[id_],n_classes,checkpoints[id_])
        
        csv_file = 'Classifier_acc_per_class_{}_{}_{}_{}_new.csv'.format(args_json['dataset']['d_priv'],args_json['dataset']['d_pub'],model_types_[0],id_)

        model = model.to(device)
        model = model.eval()

        top1,top5,mean_prob,std= test_class(model, dataset_test,n_classes)
        
        with open(csv_file, 'a') as f:
            
            writer = csv.writer(f)
            for j in range(n_classes):
                fields=['{}'.format(model_types_[id_]),
                    '{}'.format(j),   
                    '{}'.format(top1[j]),
                    '{}'.format(top5[j]),
                    '{}'.format(mean_prob[j]),
                    '{}'.format(std[j])]
                    

                writer.writerow(fields)

  