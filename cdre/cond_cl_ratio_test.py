#!/usr/bin/env python
# coding: utf-8




import numpy as np
import tensorflow as tf
import edward as ed
import pandas as pd
import argparse
import gc

import time
import os
import sys

from base_models.gans import fGAN
from base_models.classifier import Classifier
from base_models.vae import VAE, Discriminant_VAE
from cl_models import Continual_VAE, Continual_DVAE
from estimators.cond_cl_estimators import Cond_Continual_LogLinear_Estimator,Cond_Continual_f_Estimator
from utils.train_util import one_hot_encoder,shuffle_data,shuffle_batches,condition_mean,load_cifar10,gen_class_split_data
from utils.test_util import *
from utils.data_util import *


from tensorflow.examples.tutorials.mnist import input_data
from scipy.stats import multivariate_normal, norm
from sklearn.random_projection import GaussianRandomProjection


parser = argparse.ArgumentParser()
parser.add_argument('--d_dim', default=2, type=int, help='data dimension')
parser.add_argument('--T', default=10, type=int, help='number of tasks')
parser.add_argument('--dataset', default='toy_gaussians', type=str, help='type of test dataset')
parser.add_argument('--dpath', default='./', type=str, help='path of model samples when dataset is not toy_gaussians')
parser.add_argument('--rdpath', default='/home/yu/gits/data/', type=str, help='path of real data samples when dataset is not toy_gaussians')
parser.add_argument('--delta_par', default=0.01, type=float, help='delta value for changing distribution parameters, \
                                if 0, it is randomly drawn from a uniform distribution U(0.005,0.025) at each step.')
parser.add_argument('--scale_shrink', default=1., type=float, help='if 1, decrease standard deviation at each step, \
                                                                    if -1, increase it, by the value of delta_par.\
                                                                    if 0, fix scale.')
parser.add_argument('--ori_nu_means',default=[],type=str2flist,help='the list of means of original distributions for toy Gaussian')
parser.add_argument('--ori_nu_stds',default=[],type=str2flist,help='the list of stds of original distributions for toy Gaussian')
parser.add_argument('--delta_list', default=[], type=str2flist, help='the list of delta parameter for each task')
parser.add_argument('--sample_size', default=50000, type=int, help='number of samples')
parser.add_argument('--test_sample_size', default=10000, type=int, help='number of test samples')
parser.add_argument('--batch_size', default=2000, type=int, help='batch size')
parser.add_argument('--epoch', default=10000, type=int, help='number of epochs')
parser.add_argument('--print_e', default=100, type=int, help='number of epochs for printing message')
parser.add_argument('--learning_rate', default=0.00001, type=float, help='learning rate')
parser.add_argument('--reg', default=None, type=str, help='type of regularizer,can be l2 or l1')
parser.add_argument('--lambda_reg', default=0., type=float, help='Lagrange multiplier of regularity loss')
parser.add_argument('--early_stop', default=True, type=str2bool, help='if early stop when loss increases')
parser.add_argument('--validation', default=True, type=str2bool, help='if use validation set for early stop')
parser.add_argument('--constr', default=True, type=str2bool, help='if add continual constraints to the objective')
parser.add_argument('--lambda_constr', default=1., type=float, help='Lagrange multiplier of continual constraint')
parser.add_argument('--increase_constr', default=False, type=bool, help='increase Lagrange multiplier of continual constraint when number of tasks increase')
parser.add_argument('--warm_start', default='', type=str, help='specify the file path to load a trained model for task 0')
parser.add_argument('--result_path', default='./results/', type=str, help='specify the path for saving results')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--divergence', default='KL', type=str, help='the divergence used to optimize the ratio model, one of [KL, Chi]')
parser.add_argument('--vis', default=False, type=str2bool, help='enable visualization')
parser.add_argument('--save_model',default=False, type=str2bool, help='if True, save task0 model')
parser.add_argument('--save_ratios',default=True, type=str2bool, help='if True, save sample ratios for each task')
parser.add_argument('--save_samples',default=False, type=str2bool, help='if True, save test sample for each task')
parser.add_argument('--hidden_layers', default=[256,256], type=str2ilist, help='size of hidden layers, no space between characters')
parser.add_argument('--bayes', default=False, type=str2bool, help='enable Bayesian prior')
parser.add_argument('--local_constr', default=0., type=float, help='enable local estimator\'s constraint')
parser.add_argument('--num_components', default=1, type=int, help='generate samples from mixture Gaussian distributions if larger than 1,\
                                                                    each step removes one mode')
parser.add_argument('--component_weights',default=[],type=str2flist,help='component weights of mixture Gaussian')
parser.add_argument('--continual_ratio', default=True, type=str2bool, help='if False, estimate ratio by original data')
parser.add_argument('--multihead', default=False, type=bool, help='one output unit for each condition')
parser.add_argument('--festimator', default=False, type=str2bool, help='use f-estimator')
parser.add_argument('--grad_clip', default=None, type=str2flist, help='add clip for gradients')
parser.add_argument('--conv',default=False,type=str2bool,help='if True, use convolutional nets')
parser.add_argument('--min_epoch',default=100,type=int,help='minimum number of epochs when using early_stop')
parser.add_argument('--model_type',default='continual',type=str,help='could be bestmodel,bestdata or continual')
parser.add_argument('--dim_reduction', default=None, type=str, help='reduce dimension before ratio estimation,could be vae or rand_proj')
parser.add_argument('--z_dim',default=128,type=int,help='dimension of latent feature space')
parser.add_argument('--random_encode',default=True,type=str2bool,help='if True, encode return qz, if False, encode return qz.loc')
parser.add_argument('--dvae_lamb',default=1e-10,type=float,help='lambda of discriminant VAE')

args = parser.parse_args()
tf.set_random_seed(args.seed)
np.random.seed(args.seed)

if args.vis:
    import matplotlib as mtp
    mtp.rcParams['text.usetex'] = True
    mtp.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 
    mtp.rcParams['pdf.fonttype'] = 42
    mtp.rcParams['ps.fonttype'] = 42
    import matplotlib.pyplot as plt
    import seaborn as sns

if args.model_type == 'bestdata':
    args.constr = False


dist = 'Normal'
decay = None #(1000,0.1)

path = config_result_path(args.result_path)

f_name = 'cond_cl_ratio_test_d'+str(args.d_dim)+'_sd'+str(args.seed)+'_'+'-'.join(time.ctime().replace(':','').split(' '))+'/'
sub_dir = path+f_name
os.mkdir(sub_dir)

print(args)
with open(sub_dir+'configures.txt','w') as f:
    f.write(str(args))

if not args.continual_ratio:
    args.constr = False


if args.dataset == 'toy_gaussians':
    # mixture Gaussian
    if args.ori_nu_means is None:
        ori_nu_means = [0. + k * 2. for k in range(args.T)]
        ori_nu_stds = [1.]*args.T
        #nu_means, nu_stds = [ori_nu_means[0]], [ori_nu_stds[0]]
    else:
        ori_nu_means = args.ori_nu_means
        ori_nu_stds = args.ori_nu_stds
    nu_means, nu_stds = [np.array(ori_nu_means[0])], [np.array(ori_nu_stds[0])]
    if args.delta_par == 0. :
        delta_par =  args.delta_list[0] 
    else:
        delta_par = args.delta_par
    de_means = [nu_means[i] + delta_par for i in range(len(nu_means))]
    de_stds = [nu_stds[i] - args.scale_shrink*delta_par for i in range(len(nu_stds))] 
    print('nu means',nu_means,'de means',de_means)
    nu_dist,de_dist = get_dists(args.d_dim,nu_means[0],nu_stds[0],de_means[0],de_stds[0])
    ori_nu_dists = [nu_dist]
    nu_dists = [nu_dist]
    de_dists = [de_dist]
        
elif args.dataset in ['mnist','fashion']:
    ori_data_dir = os.path.join(args.rdpath,args.dataset) 
    data = input_data.read_data_sets(ori_data_dir,one_hot=False) 
    
    ori_X = np.vstack((data.train.images,data.validation.images))
    ori_Y = np.concatenate((data.train.labels,data.validation.labels))
    ori_test_X = data.test.images
    ori_test_Y = data.test.labels
    if args.conv:
        ori_X = ori_X.reshape(-1,28,28,1)
        ori_test_X = ori_test_X.reshape(-1,28,28,1)
        if args.multihead:
            net_shape = [[[4,4,args.T+1,64],[4,4,64,128]],[128*7*7,1024,args.T]]
        else:
            net_shape = [[[4,4,args.T+1,64],[4,4,64,128]],[128*7*7,1024,1]]
        nu_ph = tf.placeholder(dtype=tf.float32,shape=[args.batch_size,28,28,args.T+1])
        de_ph = tf.placeholder(dtype=tf.float32,shape=[args.batch_size,28,28,args.T+1])
        c_ph = tf.placeholder(dtype=tf.float32,shape=[args.batch_size,args.T],name='c_ph')

        prev_nu_ph = tf.placeholder(dtype=tf.float32,shape=[args.batch_size,28,28,args.T+1])
        prev_de_ph = tf.placeholder(dtype=tf.float32,shape=[args.batch_size,28,28,args.T+1])

elif args.dataset == 'spec':
    dataset = np.load(args.dpath,allow_pickle=True)

else:
    raise NotImplementedError('not implemented test dataset.')

    

if not args.conv:
    if args.dim_reduction is None:
        x_dim = args.d_dim
    else:
        x_dim = args.z_dim

    nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim+args.T],name='nu_ph')
    de_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim+args.T],name='de_ph')
    c_ph = tf.placeholder(dtype=tf.float32,shape=[None,args.T],name='c_ph')

    prev_nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim+args.T],name='prev_nu_ph')
    prev_de_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim+args.T],name='prev_de_ph')

    if args.multihead:
        net_shape = [x_dim+args.T] + args.hidden_layers + [args.T]
    else:
        net_shape = [x_dim+args.T] + args.hidden_layers + [1]



sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))


if args.festimator:
    cl_ratio_model = Cond_Continual_f_Estimator(net_shape=net_shape,nu_ph=nu_ph,de_ph=de_ph,prev_nu_ph=prev_nu_ph,\
                                               prev_de_ph=prev_de_ph,c_ph=c_ph,conv=args.conv,reg=args.reg,cl_constr=args.constr,\
                                                div_type=args.divergence,lambda_reg=args.lambda_reg,lambda_constr=args.lambda_constr,\
                                                bayes=args.bayes,local_constr=args.local_constr,c_dim=args.T)
else:
    cl_ratio_model = Cond_Continual_LogLinear_Estimator(net_shape=net_shape,nu_ph=nu_ph,de_ph=de_ph,prev_nu_ph=prev_nu_ph,\
                                               prev_de_ph=prev_de_ph,c_ph=c_ph,conv=args.conv,reg=args.reg,cl_constr=args.constr,\
                                                div_type=args.divergence,lambda_reg=args.lambda_reg,lambda_constr=args.lambda_constr,\
                                                bayes=args.bayes,local_constr=args.local_constr,c_dim=args.T)


cl_ratio_model.estimator.config_train(learning_rate=args.learning_rate,decay=decay,clip=args.grad_clip)

   
if args.dim_reduction == 'vae':
    vtrainer = Continual_VAE(args.d_dim,args.z_dim,batch_size=200,e_net_shape=[512,512],d_net_shape=[256,256],epochs=100,reg='l2',sess=sess,prior_std=0.1)
elif args.dim_reduction == 'bvae':
    vtrainer = Continual_VAE(args.d_dim,args.z_dim,batch_size=200,e_net_shape=[512,512],d_net_shape=[256,256],epochs=100,reg='l2',sess=sess,prior_std=0.1,bayes=True)
elif args.dim_reduction == 'dvae':
    vtrainer = Continual_DVAE(args.d_dim,args.z_dim,batch_size=200,e_net_shape=[512,512],d_net_shape=[256,256],\
                                epochs=100,lamb=args.dvae_lamb,learning_rate=0.002,reg='l2')
elif args.dim_reduction == 'rand_proj':
    transformer = GaussianRandomProjection(n_components=args.z_dim)
elif args.dim_reduction == 'classifier':
    clss = Classifier(x_dim=args.d_dim,y_dim=1,net_shape=[512,args.z_dim],batch_size=200,epochs=100,reg='l2',lambda_reg=0.1)


saver = tf.train.Saver()
save_name = 'sample_ratios_t'

if args.dataset == 'toy_gaussians':
    kl = [[],[],[]]
else:
    kl = []
divgergences = pd.DataFrame()
div_types = ['KL','rv_KL','Jensen_Shannon','Pearson','Hellinger']
sample_size = args.sample_size#int(args.sample_size/args.T)
test_sample_size = args.test_sample_size#int(args.test_sample_size/args.T)
tf.global_variables_initializer().run(session=sess)
prev_nu_samples,prev_de_samples,t_prev_nu_samples,t_prev_de_samples = None, None, None, None
for t in range(args.T):
    sample_ratios = pd.DataFrame()

    if args.dataset == 'toy_gaussians':
        samples_c,nu_samples,de_samples,t_samples_c,t_nu_samples,t_de_samples = gen_toygaussian_task_samples(t,sample_size,test_sample_size,args,nu_dists,de_dists)
    elif args.dataset in ['mnist','fashion']:
        samples_c,nu_samples,de_samples,t_samples_c,t_nu_samples,t_de_samples = gen_task_samples(t,sample_size,test_sample_size,args.dpath,args.T,ori_X,ori_Y,ori_test_X,ori_test_Y,model_type=args.model_type)
        nu_dist, de_dist = None, None
        # dimension reduction before ratio estimation
        if args.dim_reduction in ['vae','dvae','bvae']:
            if args.dim_reduction in ['vae','bvae']:
                vtrainer.train(nu_samples)
            else:
                vtrainer.train(nu_samples,de_samples)
            if t>0:
                # encode input for previous estimator
                prev_nu_samples = vtrainer.prev_encode(nu_samples,random=args.random_encode)
                prev_de_samples = vtrainer.prev_encode(de_samples,random=args.random_encode)
                if t_nu_samples is not None:
                    t_prev_nu_samples = vtrainer.prev_encode(t_nu_samples,random=args.random_encode)
                    t_prev_de_samples = vtrainer.prev_encode(t_de_samples,random=args.random_encode)
            

            nu_samples = vtrainer.encode(nu_samples,random=args.random_encode)
            de_samples = vtrainer.encode(de_samples,random=args.random_encode)
            t_nu_samples = vtrainer.encode(t_nu_samples,random=args.random_encode)
            t_de_samples = vtrainer.encode(t_de_samples,random=args.random_encode)
            vtrainer.update_inference()

        elif args.dim_reduction == 'rand_proj':
            transformer.fit(np.vstack([nu_samples,de_samples]))
            if t>0:
                prev_nu_samples = prev_transformer.transform(nu_samples)
                prev_de_samples = prev_transformer.transform(de_samples)
                if t_nu_samples is not None:
                    t_prev_nu_samples = prev_transformer.transform(t_nu_samples)
                    t_prev_de_samples = prev_transformer.transform(t_de_samples)
            

            nu_samples = transformer.transform(nu_samples)
            de_samples = transformer.transform(de_samples)
            t_nu_samples = transformer.transform(t_nu_samples)
            t_de_samples = transformer.transform(t_de_samples)

        elif args.dim_reduction == 'classifier':
            clX = np.vstack([nu_samples,de_samples])
            clY = np.concatenate([np.ones([nu_samples.shape[0],1]),np.zeros([de_samples.shape[0],1])])
            clX,clY = shuffle_data(clX,clY)
            clss.fit(clX,clY)
            if t>0:
                # encode input for previous estimator
                prev_nu_samples = clss.extract_feature(nu_samples,prev=True)
                prev_de_samples = clss.extract_feature(de_samples,prev=True)
                if t_nu_samples is not None:
                    t_prev_nu_samples = clss.extract_feature(t_nu_samples,prev=True)
                    t_prev_de_samples = clss.extract_feature(t_de_samples,prev=True)
            
            nu_samples = clss.extract_feature(nu_samples)
            de_samples = clss.extract_feature(de_samples)
            t_nu_samples = clss.extract_feature(t_nu_samples)
            t_de_samples = clss.extract_feature(t_de_samples)
            clss.save_params()
    elif args.dataset == 'spec':
        nu_samples,de_samples,samples_c = [],[],[]
        if args.validation:
            t_nu_samples,t_de_samples,t_samples_c = [],[],[]
        for c in range(t+1):
            sp_c = (np.ones(sample_size)*c).astype(np.int)
            sp_c = one_hot_encoder(sp_c,args.T)
            #print('check c',sp_c[:5])
            samples_c.append(sp_c)
            
            c_nu_samples,c_de_samples = dataset[t][:args.sample_size,:args.d_dim],dataset[t+1][:args.sample_size,:args.d_dim]
            nu_samples.append(c_nu_samples)
            de_samples.append(c_de_samples)
    
            if args.validation:
                t_sp_c = (np.ones(test_sample_size)*c).astype(np.int)
                t_sp_c = one_hot_encoder(t_sp_c,args.T)
                t_samples_c.append(t_sp_c)
                tc_nu_samples,tc_de_samples = dataset[t][-args.test_sample_size:,:args.d_dim], dataset[t+1][-args.test_sample_size:,:args.d_dim]
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

            
    tf.global_variables_initializer().run(session=sess)
    batch_size = args.batch_size 
    print('check shape',samples_c.shape,nu_samples.shape,de_samples.shape,t_de_samples.shape,t_nu_samples.shape)

    # load checkpoint for first task
    if t==0 and len(args.warm_start) != 0:
        saver.restore(sess,args.warm_start)
    
    else:


        losses,tlosses,terrs = cl_ratio_model.learning(sess,nu_samples,de_samples,samples_c,t_nu_samples,t_de_samples,\
                                                            t_samples_c,batch_size=batch_size,epoch=args.epoch,print_e=args.print_e,\
                                                            nu_dist=nu_dist,de_dist=de_dist,early_stop=args.early_stop,min_epoch=args.min_epoch,\
                                                            prev_nu_samples=prev_nu_samples,prev_de_samples=prev_de_samples,\
                                                            t_prev_nu_samples=t_prev_nu_samples,t_prev_de_samples=t_prev_de_samples)
       
    if args.save_model:
        saver.save(sess,sub_dir+'model_task'+str(t))

   
    # save results
    test_samples = de_samples#np.vstack([de_samples,t_de_samples])
    test_samples_c = samples_c#np.vstack([samples_c,t_samples_c])

    if test_samples.shape[0] < batch_size:
        ids = np.random.choice(np.arange(test_samples.shape[0]),size=batch_size)
        test_samples = test_samples[ids]
        test_samples_c = test_samples_c[ids]


    if args.festimator and args.divergence == 'Pearson':

        if t > 0 and args.continual_ratio:
            estimated_original_ratio = cl_ratio_model.original_ratio(sess,test_samples,test_samples,test_samples_c)

        else:
            estimated_original_ratio = cl_ratio_model.estimator.ratio(sess,test_samples,test_samples,test_samples_c)

    else:

        if t > 0 and args.continual_ratio:
            estimated_original_ratio = cl_ratio_model.original_log_ratio(sess,test_samples,test_samples,test_samples_c)

        else:
            estimated_original_ratio = cl_ratio_model.estimator.log_ratio(sess,test_samples,test_samples,test_samples_c)
        
    if args.festimator:
        odiv = args.divergence
    else:
        odiv = 'rv_KL'

    if args.dataset == 'toy_gaussians':
        true_ratio,true_step_ratio = np.zeros(test_samples.shape[0]), np.zeros(test_samples.shape[0]) 
    
        for c in range(t+1):
            ids = test_samples_c[:,c]==1
            true_ratio[ids] = -de_dists[c].log_prob(test_samples[ids]) + ori_nu_dists[c].log_prob(test_samples[ids])
            true_step_ratio[ids] = -de_dists[c].log_prob(test_samples[ids]) + nu_dists[c].log_prob(test_samples[ids])

        if args.save_ratios:
            sample_ratios['true_ratio'] = true_ratio
            sample_ratios['true_step_ratio'] = true_step_ratio
        
        print('check ratio nan',np.isnan(estimated_original_ratio).any(),np.isnan(estimated_original_ratio).any())
        for div in div_types:
            true_ds = calc_divgenerce(div,[true_ratio,true_step_ratio],test_samples_c)
            logr = False if args.festimator and args.divergence == 'Pearson'else True
            est_ds = calc_divgenerce(div,[estimated_original_ratio],test_samples_c,logr=logr)
            divgergences['true_original_'+div] = true_ds[0]
            divgergences['true_step_'+div] = true_ds[1]
            divgergences['est_original_'+div] = est_ds[0]

            if div == odiv: #compatable for earlier code
                if 'odiv' == 'rv_KL':
                    kl[0].append(np.mean([Gaussian_KL(de_d,ori_nu_d,args.d_dim) for de_d, ori_nu_d in zip(de_dists,ori_nu_dists)]))
                    kl[2].append(np.mean([Gaussian_KL(de_d,nu_d,args.d_dim) for de_d, nu_d in zip(de_dists,nu_dists)]))

                else:
                    kl[0].append(true_ds[0][:t+1].mean())                    
                    kl[2].append(true_ds[1][:t+1].mean())    
                           
                kl[1].append(est_ds[0][:t+1].mean())
                print('avg true divs',kl[0][-1],'avg estimate divs',kl[1][-1])

    else:
        for div in div_types:
            logr = False if args.festimator and args.divergence == 'Pearson'else True
            est_ds = calc_divgenerce(div,[estimated_original_ratio],test_samples_c,logr=logr)
            divgergences['est_original_'+div] = est_ds[0]
            if div == odiv:
                kl.append(est_ds[0][:t+1].mean())        
                print('divs',div, est_ds[0])
                print('avg divs',kl[-1])
    
    divgergences.to_csv(sub_dir+'divs_t'+str(t+1)+'.csv',index=False)

    if args.save_ratios:
        sample_ratios['estimated_original_ratio'] = estimated_original_ratio.sum(axis=1)
        sample_ratios['sample_c'] = np.argmax(test_samples_c,axis=1)
        sample_ratios.to_csv(sub_dir+save_name+'_t'+str(t+1)+'.csv',index=False)
    if args.save_samples:
        np.save(sub_dir+'samples_t'+str(t+1)+'.csv',test_samples,allow_pickle=True)
           
    if t > 0 and args.continual_ratio:
        contr = cl_ratio_model.get_cl_constr()
        c_nu_samples, c_de_samples = cl_ratio_model.estimator.concat_condition(nu_samples[:batch_size],de_samples[:batch_size],samples_c[:batch_size])
        contr = sess.run(contr,feed_dict={cl_ratio_model.estimator.nu_ph:c_nu_samples,\
                                            cl_ratio_model.estimator.de_ph:c_de_samples,\
                                            cl_ratio_model.estimator.c_ph:samples_c[:batch_size],\
                                            cl_ratio_model.prev_nu_ph:c_nu_samples,\
                                            cl_ratio_model.prev_de_ph:c_de_samples})
                                            
    else: 
        contr = 1.
    print('constrain check',contr)

    # visualizations
    if args.vis:
        plt.plot(test_samples[:,0],true_ratio,'.')
        plt.plot(test_samples[:,0],estimated_original_ratio,'.')

        plt.xlabel('the first dimension of x',fontsize=15)
        lgd = plt.legend([r'$\log r^*_t$',r'$\log r_t$'],fontsize=15,bbox_to_anchor=(0.9, 1.2),
            ncol=3)
        plt.savefig(sub_dir+dist+'_cl_ratio_vis_d'+str(args.d_dim)+'_task'+str(t+1)+'.pdf',bbox_extra_artists=([lgd]), bbox_inches='tight')
        plt.close()

        if len(args.warm_start) == 0 or t > 0:
            N = len(losses)
            plt.plot(range(N),losses)
            plt.plot(range(N),tlosses)

            plt.xlabel('number of epochs',fontsize=12)
            lgd=plt.legend(['training loss','validation loss'],fontsize=12)
            plt.savefig(sub_dir+'loss_compare_t'+str(t)+'.pdf',bbox_extra_artists=([lgd]), bbox_inches='tight')
            plt.close()
    
    # update distributions and models
    if t < args.T - 1 :
        if args.dataset == 'toy_gaussians':
            nu_means,nu_stds,nu_dists,de_means,de_stds,de_dists = update_toygaussian_dists(args,t,ori_nu_means,ori_nu_stds,ori_nu_dists,nu_means,nu_stds,de_means,de_stds)
            print('nu means',nu_means,'de means',de_means)
        # update model loss 
        if args.continual_ratio:       
            cl_ratio_model.update_estimator(sess,t+1,increase_constr=args.increase_constr)

        if args.dim_reduction=='rand_proj':
            prev_transformer = transformer
            transformer = GaussianRandomProjection(n_components=args.z_dim)
        # clear memory    
        gc.collect()


kl = np.array(kl)
np.savetxt(sub_dir+'kl.csv', kl, delimiter=',')


if args.vis:
    plt.plot(range(1,args.T),kl[0])
    plt.plot(range(1,args.T),kl[1])
    plt.plot(range(1,args.T),kl[2])
    plt.plot(range(1,args.T),kl[3])
    plt.xlabel('t (index of tasks)',fontsize=14)
    plt.savefig(sub_dir+'KL_compare.pdf')
    plt.close()







