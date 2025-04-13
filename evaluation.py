from evaluation_lib.KNN_dist import eval_KNN,find_samples
from evaluation_lib.accuracy import eval_acc,eval_acc_class,acc_list
from evaluation_lib.fid import eval_fid,gen_samples
from loader import *
import csv 
import os  
import utils

from argparse import ArgumentParser

if __name__ == '__main__':
    
    parser = ArgumentParser(description='Evaluation')
    # Generator configuration
    parser.add_argument('--gen_num_features', '-gnf', type=int, default=64,
                        help='Number of features of generator (a.k.a. nplanes or ngf). default: 64')
    parser.add_argument('--gen_dim_z', '-gdz', type=int, default=128,
                        help='Dimension of generator input noise. default: 128')
    parser.add_argument('--gen_bottom_width', '-gbw', type=int, default=4,
                        help='Initial size of hidden variable of generator. default: 4')
    parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',
                        help='Input noise distribution: normal (default) or uniform.')
    
    parser.add_argument('--dis_num_features', '-dnf', type=int, default=64,
                        help='Number of features of discriminator (a.k.a nplanes or ndf). default: 64')
    ############
    parser.add_argument('--classid', type=str, default='0')
    parser.add_argument('--config_exp', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='0')   
    parser.add_argument('--eval_metric', type=str, default='cosine,knn', help='Evaluation metric')
    parser.add_argument('--iters_mi', default=600, type=int)
    parser.add_argument('--N', type=int, default=-1)

    parser.add_argument('--is_general_gan', action='store_true', default=False,
                        help='Default: False')

    args = parser.parse_args()
    metric = args.eval_metric.split(',')
    fid, nsamples = 0,0
    aver_acc, aver_acc5, aver_std, aver_std5 = 0,0,0,0
    knn, std_knn = 0,0
    dataset,model_types,save_dir ='','',''
    model_names = 'temp'
    model_checkpoints = 'temp'

    
    #################### loading
    print("Loading models")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_args_exp = utils.load_json(json_file=args.config_exp) 
    loaded_args_dataset = utils.load_json(json_file=loaded_args_exp['train']['config_dataset'])
    
    save_dir = os.path.join(loaded_args_exp['train']['result_dir'],loaded_args_dataset['dataset']['d_priv'],loaded_args_dataset['dataset']['d_pub'],loaded_args_exp['train']['target'],args.save_dir, args.classid.replace(',','_'))
    
    args.num_classes = loaded_args_dataset['dataset']['n_classes']
    args.path_G = loaded_args_exp['train']['path_G']
    ###########################################
    ###########     load model       ##########
    ###########################################
    import loader
    #evaluation classifier
    # G,_ = loader.load_cgan(args,device,args.path_G)
    # args.num_classes = 1000
    # G,_ = loader.load_cgan_wgan(args,device,args.path_G)
    
    if args.is_general_gan == True:
        G,_ = loader.load_cgan_wgan(args,device,loaded_args_exp['train']['path_G'])
    else:
        G,_ = loader.load_cgan(args,device,loaded_args_exp['train']['path_G'])
    

    E = get_model('FaceNet',loaded_args_dataset['dataset']['E_n_classes'],loaded_args_dataset['dataset']['path_E'])

    G.eval()
    E.eval()
    dataset = loaded_args_dataset['dataset']['d_priv']
    
    

    fake_dir,_ = gen_samples(G,E,dataset,save_dir,args.N,args.iters_mi)
    


    for metric_ in metric:
        if metric_ == 'fid':
            fid, nsamples = eval_fid(E,dataset,fake_dir)
        elif metric_ == 'cosine':
            aver_acc, aver_acc5, aver_std, aver_std5 = eval_acc(E,dataset,fake_dir)
        elif metric_ == 'knn':
            knn,std_knn = eval_KNN(dataset,fake_dir)
          
   
    csv_file = 'Evaluation_plg_{}.csv'.format(dataset)

    if not os.path.exists(csv_file):
        header = ['Save_dir', 'G', 'iters_mi',                    
                    'acc','std','acc5','std5',
                    'fid','knn','std_knn']
        with open(csv_file, 'w') as f:                
            writer = csv.writer(f)
            writer.writerow(header)

    method = 'plg'
    fields=['{}'.format(save_dir), 
            '{}'.format(args.path_G), 
            '{}'.format(args.iters_mi),
            '{:.2f}'.format(aver_acc),
            '{:.2f}'.format(aver_std),
            '{:.2f}'.format(aver_acc5),
            '{:.2f}'.format(aver_std5),
            '{:.2f}'.format(fid),
            '{:.2f}'.format(knn),
            '{:.2f}'.format(std_knn),]
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
