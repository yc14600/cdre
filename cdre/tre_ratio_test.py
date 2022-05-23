#!/usr/bin/env python
# coding: utf-8




import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import gc

import time
import os
import sys

from base_models.gans import fGAN
from base_models.classifier import Classifier
from cl_models import Continual_VAE, Continual_DVAE
from estimators.tre_estimators import TRE_f_Estimator,TRE_KL_Loglinear_Estimator
from utils.train_util import one_hot_encoder,shuffle_data
from utils.test_util import *
from utils.data_util import *



def get_dist_list(d_dim,means,stds,dist_type='Normal'):
    dlist = []
    for mu, std in zip(means,stds):
        dt = gen_dist(d_dim,mu,std,dist_type=dist_type)
        dlist.append(dt)
  
    return dlist

def gen_tre_gaussian_samples(base_samples,dist):
    samples = base_samples*dist.scale + dist.mean
    return samples

def gen_tre_gaussian_task_samples(t,sample_size,test_sample_size,args,nu_dists,de_dists):
    nu_samples,de_samples,samples_c = [],[],[]
    if args.validation:
        t_nu_samples,t_de_samples,t_samples_c = [],[],[]
    else:
        t_nu_samples,t_de_samples,t_samples_c = None, None,None

    base_samples = gen_samples(sample_size,args.d_dim,0.,1.,dist_type='Normal')
    if args.validation:
        base_test_samples = gen_samples(test_sample_size,args.d_dim,0.,1.,dist_type='Normal')
    for c in range(t+1):
        sp_c = (np.ones(sample_size)*c).astype(np.int)
        sp_c = one_hot_encoder(sp_c,args.T)
        #print('check c',sp_c[:5])
        samples_c.append(sp_c)
           
        c_nu_samples = gen_tre_gaussian_samples(base_samples,nu_dists[c])
        c_de_samples = gen_tre_gaussian_samples(base_samples,de_dists[c])

        nu_samples.append(c_nu_samples)
        de_samples.append(c_de_samples)
    
        if args.validation:
            t_sp_c = (np.ones(test_sample_size)*c).astype(np.int)
            t_sp_c = one_hot_encoder(t_sp_c,args.T)
            t_samples_c.append(t_sp_c)

            tc_nu_samples = gen_tre_gaussian_samples(base_test_samples,nu_dists[c])
            tc_de_samples = gen_tre_gaussian_samples(base_test_samples,de_dists[c])
            
            t_nu_samples.append(tc_nu_samples)
            t_de_samples.append(tc_de_samples)

    
    ids = np.arange(sample_size*(t+1))
    np.random.shuffle(ids)                 
    samples_c = np.vstack(samples_c)[ids]
    nu_samples = np.vstack(nu_samples)[ids]
    de_samples = np.vstack(de_samples)[ids]

    if args.validation:
        ids = np.arange(test_sample_size*(t+1))
        np.random.shuffle(ids)
        t_samples_c = np.vstack(t_samples_c)[ids]
        t_nu_samples = np.vstack(t_nu_samples)[ids]
        t_de_samples = np.vstack(t_de_samples)[ids]

    return samples_c,nu_samples,de_samples,t_samples_c,t_nu_samples,t_de_samples


parser = argparse.ArgumentParser()
parser.add_argument('--d_dim', default=2, type=int, help='data dimension')
parser.add_argument('--dataset', default='gaussian', type=str, help='data set name')
parser.add_argument('--datapath', default='', type=str, help='data path when it is not gaussian')
parser.add_argument('--task_type', default='div', type=str, help='task type, div or regression')
parser.add_argument('--T', default=10, type=int, help='number of tasks')
parser.add_argument('--delta_mean', default=0.01, type=float, help='delta value for changing distribution parameters, \
                                if 0, it is randomly drawn from a uniform distribution U(0.005,0.025) at each step.')
parser.add_argument('--delta_std', default=0., type=float, help='delta value for changing standard deviation')
parser.add_argument('--delta_list', default=[], type=str2flist, help='the list of delta parameter for each task')
parser.add_argument('--sample_size', default=50000, type=int, help='number of samples')
parser.add_argument('--test_sample_size', default=10000, type=int, help='number of test samples')
parser.add_argument('--batch_size', default=2000, type=int, help='batch size')
parser.add_argument('--epoch', default=10000, type=int, help='number of epochs')
parser.add_argument('--print_e', default=100, type=int, help='number of epochs for printing message')
parser.add_argument('--learning_rate', default=0.00002, type=float, help='learning rate')
parser.add_argument('--reg', default=None, type=str, help='type of regularizer,can be l2 or l1')
parser.add_argument('--lambda_reg', default=1., type=float, help='Lagrange multiplier of regularity loss')
parser.add_argument('--early_stop', default=True, type=str2bool, help='if early stop when loss increases')
parser.add_argument('--validation', default=True, type=str2bool, help='if use validation set for early stop')
parser.add_argument('--constr', default=True, type=str2bool, help='if add continual constraints to the objective')
parser.add_argument('--lambda_constr', default=1., type=float, help='Lagrange multiplier of continual constraint')
parser.add_argument('--increase_constr', default=False, type=bool, help='increase Lagrange multiplier of continual constraint when number of tasks increase')
parser.add_argument('--warm_start', default='', type=str, help='specify the file path to load a trained model for task 0')
parser.add_argument('--result_path', default='./results/', type=str, help='specify the path for saving results')
parser.add_argument('--f_name', default='', type=str, help='specify the folder name for saving all result files,if empty, generated automatically with timestamp.')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--divergence', default='KL', type=str, help='the divergence used to optimize the ratio model, one of [KL, rv_KL, Pearson, Hellinger, Jensen_Shannon]')
parser.add_argument('--unlimit_samples', default=False, type=str2bool, help='unlimited number of samples')
parser.add_argument('--vis', default=False, type=str2bool, help='enable visualization')
parser.add_argument('--save_model',default=False, type=str2bool, help='if True, save task0 model')
parser.add_argument('--hidden_layers', default=[256,256], type=str2ilist, help='size of hidden layers, no space between characters')
parser.add_argument('--bayes', default=False, type=str2bool, help='enable Bayesian prior')
parser.add_argument('--local_constr', default=0., type=float, help='enable local estimator\'s constraint')
parser.add_argument('--num_components', default=1, type=int, help='generate samples from mixture Gaussian distributions if larger than 1,\
                                                                    each step removes one mode')
parser.add_argument('--component_weights',default=[],type=str2flist,help='component weights of mixture Gaussian')
parser.add_argument('--continual_ratio', default=True, type=str2bool, help='if False, estimate ratio by original data')
parser.add_argument('--festimator', default=False, type=str2bool, help='use f-estimator')
parser.add_argument('--cuda', default=False, type=str2bool, help='use cuda')
parser.add_argument('--restart', default=False, type=str2bool, help='restart in the process of cdre by some condition')
parser.add_argument('--restart_th', default=0.1, type=float, help='restart threshold')
parser.add_argument('--min_epoch',default=100,type=int,help='minimum number of epochs when using early_stop')
parser.add_argument('--save_ratios',default=True, type=str2bool, help='if True, save sample ratios for each task')

args = parser.parse_args()
tf.set_random_seed(args.seed)
np.random.seed(args.seed)

if not args.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if args.vis:
    import matplotlib as mtp
    mtp.rcParams['text.usetex'] = True
    mtp.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 
    mtp.rcParams['pdf.fonttype'] = 42
    mtp.rcParams['ps.fonttype'] = 42
    import matplotlib.pyplot as plt
    import seaborn as sns


dist = 'Normal'
decay = None #(1000,0.1)

path = config_result_path(args.result_path)

f_name = 'tre_ratio_test_d'+str(args.d_dim)+'_sd'+str(args.seed)+'_'+'-'.join(time.ctime().replace(':','').split(' '))+'/'
sub_dir = path+f_name
os.mkdir(sub_dir)

print(args)
with open(sub_dir+'configures.txt','w') as f:
    f.write(str(args))

if not args.continual_ratio:
    args.constr = False


if args.dataset == 'gaussian':
    # mixture Gaussian
    dmean = args.delta_mean 
    dstd = args.delta_std
    ori_nu_mean, ori_nu_std = 0., 1.

    nu_means = [ori_nu_mean + dmean * i for i in range(args.T)]
    nu_stds = [ori_nu_std + dstd * i for i in range(args.T)]

    de_means = [ori_nu_mean + dmean * (i+1) for i in range(args.T)]
    de_stds = [ori_nu_std + dstd * (i+1) for i in range(args.T)]

    nu_dists = get_dist_list(args.d_dim,nu_means,nu_stds,dist_type='Normal')
    de_dists = get_dist_list(args.d_dim,de_means,de_stds,dist_type='Normal')

else:
    raise NotImplementedError('not implemented test dataset.')

    
x_dim = args.d_dim
nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim],name='nu_ph')
de_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim],name='de_ph')
c_ph = tf.placeholder(dtype=tf.float32,shape=[None,args.T],name='c_ph')

prev_nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim],name='prev_nu_ph')
prev_de_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim],name='prev_de_ph')


net_shape = [x_dim] + args.hidden_layers + [args.T]
    


sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))


if args.festimator:
    tre_ratio_model = TRE_f_Estimator(net_shape=net_shape,nu_ph=nu_ph,de_ph=de_ph,\
                                               c_ph=c_ph,reg=args.reg,div_type=args.divergence,\
                                                lambda_reg=args.lambda_reg,\
                                                c_dim=args.T)
else:
    tre_ratio_model = TRE_KL_Loglinear_Estimator(net_shape=net_shape,nu_ph=nu_ph,de_ph=de_ph,\
                                               c_ph=c_ph,reg=args.reg,cl_constr=args.constr,\
                                                div_type=args.divergence,lambda_reg=args.lambda_reg,\
                                                c_dim=args.T)


tre_ratio_model.config_train(learning_rate=args.learning_rate,decay=decay)

   


saver = tf.train.Saver()
save_name = 'sample_ratios_t'

if args.dataset == 'gaussian':
    kl = [[],[],[]]
else:
    kl = []
divgergences = pd.DataFrame()
div_types = ['KL','rv_KL','Jensen_Shannon','Pearson','Hellinger']
sample_size = args.sample_size#int(args.sample_size/args.T)
test_sample_size = args.test_sample_size#int(args.test_sample_size/args.T)
tf.global_variables_initializer().run(session=sess)
prev_nu_samples,prev_de_samples,t_prev_nu_samples,t_prev_de_samples = None, None, None, None

sample_ratios = pd.DataFrame()

if args.dataset == 'gaussian':
    samples_c,nu_samples,de_samples,t_samples_c,t_nu_samples,t_de_samples = gen_toygaussian_task_samples(args.T-1,sample_size,test_sample_size,args,nu_dists,de_dists)
else:
    raise NotImplementedError('only gassian implemented')

        
tf.global_variables_initializer().run(session=sess)
batch_size = args.batch_size 
print('check shape',samples_c.shape,nu_samples.shape,de_samples.shape,t_de_samples.shape,t_nu_samples.shape)

# load checkpoint for first task
if len(args.warm_start) != 0:
    saver.restore(sess,args.warm_start)

else:


    losses,tlosses,terrs = tre_ratio_model.learning(sess,nu_samples,de_samples,samples_c,t_nu_samples,t_de_samples,\
                                                        t_samples_c,batch_size=batch_size,epoch=args.epoch,print_e=args.print_e,\
                                                        early_stop=args.early_stop,min_epoch=args.min_epoch)
       


for t in range(args.T): 
# save results
    test_samples = de_dists[t].sample(test_sample_size+sample_size)#de_samples#np.vstack([de_samples,t_de_samples])

    #test_samples_c = samples_c#np.vstack([samples_c,t_samples_c])

    if test_samples.shape[0] < batch_size:
        ids = np.random.choice(np.arange(test_samples.shape[0]),size=batch_size)
        test_samples = test_samples[ids]
        #test_samples_c = test_samples_c[ids]
            
    estimated_original_ratio = tre_ratio_model.original_log_ratio(sess,test_samples,test_samples,t)
    print('check ratio nan',np.isnan(estimated_original_ratio).any(),np.isnan(estimated_original_ratio).any())

       
    if args.festimator:
        odiv = args.divergence
    else:
        odiv = 'rv_KL'

    if args.dataset == 'gaussian':

        true_ratio = -de_dists[t].log_prob(test_samples) + nu_dists[0].log_prob(test_samples)
        true_step_ratio = -de_dists[t].log_prob(test_samples) + nu_dists[t].log_prob(test_samples)

        if args.save_ratios:
            sample_ratios['estimated_log_ratio'] = estimated_original_ratio
            sample_ratios['estimated_original_log_ratio'] = estimated_original_ratio
            sample_ratios['true_log_ratio'] = true_ratio
            sample_ratios['true_step_log_ratio'] = true_step_ratio
        
        true_kl = Gaussian_KL(de_dists[t],nu_dists[0],args.d_dim) 
        kl[0].append(true_kl)
        true_step_kl = Gaussian_KL(de_dists[t], nu_dists[t],args.d_dim) 
        kl[2].append(true_step_kl)
        print('true kls', true_kl,true_step_kl)
        
        est_kl = np.mean(- estimated_original_ratio)
        kl[1].append(est_kl)
        print('estimate kl', est_kl)
        

    else:
        raise NotImplementedError()


    if args.save_ratios:
        sample_ratios.to_csv(os.path.join(sub_dir,save_name+'_t'+str(t+1)+'.csv'),index=False)
            
# In[39]:
if args.dataset == 'gaussian':
    kl = np.array(kl)
    np.savetxt(os.path.join(sub_dir,'divergence_compare.csv'), kl, delimiter=',')










