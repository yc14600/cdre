
import numpy as np
import tensorflow as tf
import edward as ed
import pandas as pd
import argparse
import gc

import time
import os
import sys
import prd.prd_score as prd


from utils.fid import FID_Evaluator
from utils.kid import KID_Evaluator
from estimators import KL_Loglinear_Estimator
from utils.train_util import one_hot_encoder,shuffle_data,shuffle_batches,condition_mean,load_cifar10,gen_class_split_data
from utils.test_util import *
from utils.data_util import *

from tensorflow.examples.tutorials.mnist import input_data


def eval_prd_segs(eval_data,ref_data,num_clusters,num_segs=4,num_runs=10):
    assert(np.log2(num_segs)==int(np.log2(num_segs)))
    rec_mean,prec_mean=[[] for i in range(num_segs)],[[] for i in range(num_segs)]
    for r in range(num_runs):
        eval_dist,ref_dist = prd._cluster_into_bins(eval_data,ref_data,num_clusters)
        eval_dist_segs = [eval_dist]
        for b in range(num_segs-1):
            eval_dist = eval_dist_segs.pop(0)
            h_eval_dist, l_eval_dist = sep_half_density(eval_dist)
            eval_dist_segs.append(h_eval_dist)
            eval_dist_segs.append(l_eval_dist)
        assert(len(eval_dist_segs)==num_segs)
        for s,edist in enumerate(eval_dist_segs):
            prec_s,rec_s = prd.compute_prd(edist,ref_dist=ref_dist)
            prec_mean[s].append(prec_s)
            rec_mean[s].append(rec_s)

    rec_mean = [np.mean(rec,axis=0) for rec in rec_mean]
    prec_mean = [np.mean(prec,axis=0) for prec in prec_mean]

    return prec_mean,rec_mean


def sep_half_density(eval_dist):
    sort_id = np.argsort(eval_dist)
    hids,lids = [sort_id[-1]],list(sort_id[:-1])
    si = -2
    while np.sum(eval_dist[hids])<0.5:
        #print(hids,lids)
        hids.append(sort_id[si])
        lids.remove(sort_id[si])
        si-=1
    print('mass of high part',np.sum(eval_dist[hids]))
    h_eval_dist,l_eval_dist = np.zeros_like(eval_dist),np.zeros_like(eval_dist)
    h_eval_dist[hids],l_eval_dist[lids] =  eval_dist[hids],eval_dist[lids]
    h_eval_dist /= h_eval_dist.sum()
    l_eval_dist /= l_eval_dist.sum()

    return h_eval_dist,l_eval_dist

    
def get_data(args, d_dim=784):
    dataset = args.dataset.lower()
    path = args.npath
    if dataset in ['mnist','fashion']:
        if path is None:
            path = '../datasets/mnist/' if dataset=='mnist' else '../datasets/fashion-mnist/'
        data = input_data.read_data_sets(path,one_hot=False) 
        ori_X = np.vstack((data.train.images,data.validation.images))
        ori_Y = np.concatenate((data.train.labels,data.validation.labels))
        ori_test_X = data.test.images
        ori_test_Y = data.test.labels
            
    elif dataset == 'ffhq':
        if path is None:
            path = '/home/yu/gits/data/ffhq/stylegan/real_samples.gz'
        real_samples = extract_data(path,shape=[d_dim],dtype=np.float32)
        print('ffhq real samples shape',real_samples.shape)
        ori_X = real_samples[:args.sample_size*args.T]
        ori_test_X = real_samples[-args.test_sample_size*args.T:]
        ori_Y, ori_test_Y = None, None

    else:
        raise NotImplementedError('NOT Supported Dataset!')

    return ori_X, ori_Y, ori_test_X, ori_test_Y


def define_feature_model(args,d_dim):

    if args.conv:
        if args.feature_type == 'classifier':
            net_shape = [[[4,4,1,64],[4,4,64,128]],[128*7*7,args.z_dim,args.T]]
            d_net_shape = None
        elif args.feature_type == 'VAE':
            net_shape = [[[4,4,1,64],[4,4,64,128]],[128*7*7,args.z_dim]]
            d_net_shape = [[args.z_dim,128*7*7],[[7,7,128],[14,14,64],[28,28,1]]]
    else:
        d_dim = ori_X.shape[1]      
        if args.feature_type == 'classifier':
            net_shape = [d_dim] + args.hidden_layers + [args.z_dim, args.T]
            d_net_shape = None
        elif args.feature_type == 'VAE':
            net_shape = args.hidden_layers           
            d_net_shape = args.hidden_layers if args.d_net_shape is None else args.d_net_shape

    return net_shape, d_net_shape

# estimate KL divergence for a static task
def static_kl_div(args,d_dim,nu_samples,de_samples,test_nu_samples,test_de_samples,eval_model):
        if args.extract_feature:
            nu_samples = eval_model.get_activations(nu_samples)
            de_samples = eval_model.get_activations(de_samples)
            test_nu_samples = eval_model.get_activations(test_nu_samples)
            test_de_samples = eval_model.get_activations(test_de_samples)
            d_dim = args.z_dim
        nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,d_dim],name='nu_ph')
        de_ph = tf.placeholder(dtype=tf.float32,shape=[None,d_dim],name='de_ph')
        r_net_shape = [d_dim]+args.hidden_layers+[1]#[d_dim,256,256,1]
        r_estimator = KL_Loglinear_Estimator(r_net_shape,nu_ph,de_ph)
        r_estimator.config_train(learning_rate=0.00001)
        with eval_model.sess.as_default():   
            tf.global_variables_initializer().run()
        r_estimator.learning(eval_model.sess,nu_samples,de_samples,test_nu_samples=test_nu_samples,min_epoch=100,\
                            test_de_samples=test_de_samples,epoch=1000,early_stop=True,print_e=50,batch_size=2000)
        test_samples = np.concatenate([de_samples,test_de_samples]) if args.test_sample_size > 0 else de_samples
        estimated_ratio = r_estimator.log_ratio(eval_model.sess,test_samples,test_samples).reshape(-1)
        np.savetxt(sub_dir+'sample_ratios.csv', estimated_ratio, delimiter=',')
        score_t = calc_divgenerce(args.divergence,[estimated_ratio],logr=True)

        return score_t


parser = argparse.ArgumentParser()

parser.add_argument('-T', default=10, type=int, help='number of tasks')
parser.add_argument('-dataset', default='MNIST', type=str, help='type of test dataset')
parser.add_argument('-dpath', default=None, type=str, help='data path of denorminator distribution')
parser.add_argument('-npath', default=None, type=str, help='data path of numerator distribution')
parser.add_argument('-sample_size', default=10000, type=int, help='number of samples of each class')
parser.add_argument('-test_sample_size', default=0, type=int, help='number of test samples of each class')
parser.add_argument('-batch_size', default=100, type=int, help='batch size')
parser.add_argument('-epoch', default=200, type=int, help='number of epochs')
parser.add_argument('-print_e', default=100, type=int, help='number of epochs for printing message')
parser.add_argument('-learning_rate', default=0.002, type=float, help='learning rate')
parser.add_argument('-warm_start', default='', type=str, help='specify the file path to load a trained model for feature extraction')
parser.add_argument('-result_path', default='./results/', type=str, help='specify the path for saving results')
parser.add_argument('-seed', default=0, type=int, help='random seed')
parser.add_argument('-hidden_layers', default=[512,256], type=str2ilist, help='size of hidden layers, no space between characters')
parser.add_argument('-grad_clip', default=None, type=str2flist, help='add clip for gradients')
parser.add_argument('-conv',default=False,type=str2bool,help='if True, use convolutional nets')
parser.add_argument('-test_model_type',default='WGAN',type=str,help='type of tested model')
parser.add_argument('-save_model',default=False,type=str2bool,help='save the trained model or not')
parser.add_argument('-model_type',default='bestdata',type=str,help='can be bestmodel,bestdata,continual,single,splitclass,splitsize,rv_split')
parser.add_argument('-extract_feature',default=True,type=str2bool,help='extract feature or not')
parser.add_argument('-score_type',default='fid',type=str,help='can be kid, fid, kl, prd')
parser.add_argument('-kid_degree',default=3,type=int,help='degree of kid kernel')
parser.add_argument('-feature_type',default='classifier',type=str,help='can be classifier, VAE')
parser.add_argument('-z_dim',default=64,type=int,help='dimension of transformed features')
parser.add_argument('-d_net_shape',default=None,type=str2ilist,help='decoder net shape of VAE')
parser.add_argument('-reg', default=None, type=str, help='regularization of feature model')
parser.add_argument('-divergence', default='KL', type=str, help='the divergence used to optimize the ratio model, one of [KL, Chi]')
parser.add_argument('-festimator', default=False, type=str2bool, help='use f-estimator')
parser.add_argument('-shuffled', default=False, type=str2bool, help='if model samples saved in a shuffled order')



args = parser.parse_args()
tf.set_random_seed(args.seed)
np.random.seed(args.seed)

path = config_result_path(args.result_path)

f_name = args.score_type+'_test_'+args.dataset+'_'+args.test_model_type+'_'+args.model_type+'-'.join(time.ctime().replace(':','').split(' '))+'/'
sub_dir = path+f_name
os.mkdir(sub_dir)

print(args)
with open(sub_dir+'configures.txt','w') as f:
    f.write(str(args))


if args.dataset.lower() in ['mnist','fashion']:
    ori_X, ori_Y, ori_test_X, ori_test_Y = get_data(args,d_dim=784)
    if args.conv:
        ori_X = ori_X.reshape(-1,28,28,1)
        ori_test_X = ori_test_X.reshape(-1,28,28,1)
        d_dim = list(ori_X.shape[1:]) 

    else:
        d_dim = ori_X.shape[1]      

    net_shape, d_net_shape = define_feature_model(args,d_dim)
            
elif args.dataset.lower() == 'ffhq':
    # use extracted feature already, disable feature extraction here
    args.conv = False
    args.extract_feature = False 
    args.shuffled = True 
    args.feature_type = 'classifier'
    d_dim = 2048
    ori_X, ori_Y, ori_test_X, ori_test_Y = get_data(args,d_dim=d_dim)
    net_shape, d_net_shape = None, None #features from pretrained inception-v3
    
else:
    raise NotImplementedError('NOT implemented dataset!')

print('d dim',d_dim)

out_dim = args.T if args.feature_type == 'classifier' else args.z_dim
if args.score_type in ['fid','kl','prd','prd_seg','prd_half']:
    eval_model = FID_Evaluator(d_dim,out_dim,net_shape,args.batch_size,conv=args.conv,clip=args.grad_clip,\
                                reg=args.reg,feature_type=args.feature_type,d_net_shape=d_net_shape,ipath=args.warm_start)
elif args.score_type == 'kid':
    eval_model = KID_Evaluator(d_dim,out_dim,net_shape,args.batch_size,conv=args.conv,clip=args.grad_clip,\
                                reg=args.reg,feature_type=args.feature_type,d_net_shape=d_net_shape,ipath=args.warm_start)
else:
    raise NotImplementedError('Score type not implemented.')

if args.extract_feature:
    saver = tf.train.Saver()
    if args.feature_type != 'inception':
        if args.warm_start:
            saver.restore(eval_model.sess,args.warm_start)
        else:
            if args.feature_type == 'classifier':
                ori_label = one_hot_encoder(ori_Y,args.T)
            else:
                ori_label = None
            eval_model.train_net(ori_X,ori_label,warm_start=False)

            if args.save_model:
                saver.save(eval_model.sess,sub_dir+'feature_model_'+args.feature_type)

score = []
if args.model_type in ['single','splitclass','splitsize']:
    print('dpath',args.dpath)
    _,nu_samples,de_samples,_,test_nu_samples,test_de_samples = gen_task_samples(args.T,args.sample_size,args.test_sample_size,\
                                                                                args.dpath,args.T,ori_X,ori_Y,ori_test_X,\
                                                                                ori_test_Y,model_type=args.model_type,shuffled=args.shuffled)

    print('check test shape',test_nu_samples.shape,test_de_samples.shape)
    if args.score_type == 'kl':
        score_t = static_kl_div(args,d_dim,nu_samples,de_samples,test_nu_samples,test_de_samples,eval_model)
    elif args.score_type == 'fid':
        score_t = eval_model.score(nu_samples,de_samples,extractf=args.extract_feature)
    elif args.score_type == 'kid':
        score_t = eval_model.score(nu_samples,de_samples,extractf=args.extract_feature,degree=args.kid_degree)

    elif args.score_type in ['prd','prd_seg','prd_half']:
        if args.extract_feature:
            nu_samples = eval_model.get_activations(nu_samples)
            de_samples = eval_model.get_activations(de_samples)
        if args.score_type == 'prd':
            score_t = prd.compute_prd_from_embedding(ref_data=nu_samples, eval_data=de_samples,num_clusters=100)
        elif args.score_type == 'prd_seg':
            score_t = eval_prd_segs(eval_data=de_samples,ref_data=nu_samples,num_clusters=100)
        elif args.score_type == 'prd_half':
            score_t = eval_prd_segs(eval_data=de_samples,ref_data=nu_samples,num_clusters=100,num_segs=2)


    print('task {0} {1}'.format(args.score_type,np.mean(score_t)))
    if args.score_type == 'kid':
        score.append([np.mean(score_t),np.std(score_t)])
    elif args.score_type in ['fid','kl','prd','prd_seg','prd_half']:
        score.append(score_t)
else:
    for t in range(args.T):

        _,nu_samples,de_samples,_,_,_ = gen_task_samples(t,args.sample_size,0,\
                                                    args.dpath,args.T,ori_X,ori_Y,ori_test_X,ori_test_Y,model_type=args.model_type)

        score_t = eval_model.score(nu_samples,de_samples,extractf=args.extract_feature)
        print('task {0} {1} {2}'.format(t,args.score_type,np.mean(score_t)))
        if args.score_type == 'kid':
            score.append([np.mean(score_t),np.std(score_t)])
        elif args.score_type == 'fid':
            score.append(score_t)
        else:
            raise TypeError('estimate KL div for continual tasks by cond_cl_ratio_test.py')

# save results
if 'prd' not in args.score_type:
    score = np.array(score)
    np.savetxt(sub_dir+args.score_type+'.csv', score, delimiter=',')
    
else:
    if args.score_type == 'prd' and args.model_type in ['single','splitclass','splitsize']:
        np.savetxt(sub_dir+args.score_type+'.csv', score[0], delimiter=',')
        prd.plot(score,out_path=os.path.join(sub_dir,'prd.pdf'))
    elif args.score_type in ['prd_seg','prd_half'] and args.model_type == 'single':
        score = score[0]
        for i,(prec,rec) in enumerate(zip(score[0],score[1])):
            ret = np.vstack([prec,rec])
            np.savetxt(sub_dir+args.score_type+str(i)+'.csv', ret, delimiter=',')
    else:
        for i,s in enumerate(score):
            s = np.array(s)
            np.savetxt(sub_dir+args.score_type+str(i)+'.csv', s, delimiter=',')
        
