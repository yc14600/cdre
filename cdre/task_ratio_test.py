#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import edward as ed
import pandas as pd
import argparse
import gc

# In[2]:


# In[3]:

import time
import os
path = os.getcwd()
import sys
sys.path.append(path+'/../')


# In[4]:

from base_models.vae import VAE, Discriminant_VAE
from cl_models.cvae import Continual_VAE,Continual_DVAE
from estimators import LogLinear_Estimator, f_Estimator
from cl_estimators import Continual_LogLinear_Estimator,Continual_f_Estimator
from cond_cl_estimators import Cond_Continual_LogLinear_Estimator,Cond_Continual_f_Estimator
from utils.model_util import define_dense_layer
from utils.test_util import *
from base_models.mixture_models import MixDiagGaussian

# In[5]:
from scipy.stats import multivariate_normal, norm
from tensorflow.examples.tutorials.mnist import input_data

# In[13]:


class generator(object):
    def __init__(self,mean,std,d_dim):
        self.mean = mean
        self.std = std
        self.d_dim = d_dim
        
    def draw_samples(self,sample_size):
        return gen_samples(sample_size,self.d_dim,self.mean,self.std)
    

# In[7]:
parser = argparse.ArgumentParser()
parser.add_argument('-d_dim', default=2, type=int, help='data dimension')
parser.add_argument('-T', default=10, type=int, help='number of tasks')
parser.add_argument('-sample_size', default=50000, type=int, help='number of samples')
parser.add_argument('-test_sample_size', default=10000, type=int, help='number of test samples')
parser.add_argument('-batch_size', default=2000, type=int, help='batch size')
parser.add_argument('-epoch', default=10000, type=int, help='number of epochs')
parser.add_argument('-print_e', default=100, type=int, help='number of epochs for printing message')
parser.add_argument('-learning_rate', default=0.00002, type=float, help='learning rate')
parser.add_argument('-reg', default=None, type=str, help='type of regularizer,can be l2 or l1')
parser.add_argument('-lambda_reg', default=1., type=float, help='Lagrange multiplier of regularity loss')
parser.add_argument('-early_stop', default=True, type=str2bool, help='if early stop when loss increases')
parser.add_argument('-validation', default=True, type=str2bool, help='if use validation set for early stop')
parser.add_argument('-constr', default=True, type=str2bool, help='if add continual constraints to the objective')
parser.add_argument('-lambda_constr', default=1., type=float, help='Lagrange multiplier of continual constraint')
parser.add_argument('-warm_start', default='', type=str, help='specify the file path to load a trained model for task 0')
parser.add_argument('-result_path', default='./results/', type=str, help='specify the path for saving results')
parser.add_argument('-seed', default=0, type=int, help='random seed')
parser.add_argument('-divergence', default='KL', type=str, help='the divergence used to optimize the ratio model, one of [KL, Chi]')
parser.add_argument('-save_model',default=False, type=str2bool, help='if True, save task0 model')
parser.add_argument('-hidden_layers', default=[256,256], type=str2ilist, help='size of hidden layers, no space between characters')
parser.add_argument('-local_constr', default=0., type=float, help='enable local estimator\'s constraint')
parser.add_argument('-continual_ratio', default=True, type=str2bool, help='if False, estimate ratio by original data')
parser.add_argument('-festimator', default=False, type=str2bool, help='use f-estimator')
parser.add_argument('-dataset', default='mnist', type=str, help='type of dataset, can be mnist, fashion.')
parser.add_argument('-dim_reduction', default=None, type=str, help='reduce dimension before ratio estimation,could be vae or rand_proj')
parser.add_argument('-z_dim',default=64,type=int,help='dimension of latent feature space')
parser.add_argument('-dvae_lamb',default=1e-10,type=float,help='lambda of discriminant VAE')
parser.add_argument('-vae_lamb_reg',default=1e-3,type=float,help='lambda of regularization loss of VAE or DVAE')
parser.add_argument('-min_epoch',default=50,type=int,help='minimum number of epochs when using early_stop')
parser.add_argument('-grad_clip', default=None, type=str2flist, help='add clip for gradients')
parser.add_argument('-conditional', default=False, type=str2bool, help='type of estimator, conditional or not.')
parser.add_argument('-conv',default=False,type=str2bool,help='if True, use convolutional nets')
parser.add_argument('-start_task',default=0,type=int,help='index of start task')
parser.add_argument('-pairwise', default=False, type=str2bool, help='estimate pairwise ratio of tasks.')


args = parser.parse_args()

tf.set_random_seed(args.seed)
np.random.seed(args.seed)

decay = None #(1000,0.1)

if args.result_path[-1] != '/':
    path = args.result_path+'/'
else:
    path = args.result_path

if not os.path.exists(path):
    os.makedirs(path)

f_name = 'task_ratio_test_'+args.dataset+'_start'+str(args.start_task)+'-'.join(time.ctime().replace(':','').split(' '))+'/'
sub_dir = path+f_name
os.mkdir(sub_dir)

print(args)
with open(sub_dir+'configures.txt','w') as f:
    f.write(str(args))

if not args.continual_ratio:
    args.constr = False

if args.pairwise:
    args.conditional = False
    args.continual_ratio = False
    args.constr = False
    args.start_task = 0
    args.dim_reduction = 'vae'
    tcs = [(i,j) for i in range(args.T) for j in range(i+1,args.T)]
else:
    tcs = []


# In[8]:

if args.dataset in ['mnist','fashion']:
    ori_data_dir = '../datasets/mnist/' if args.dataset== 'mnist' else '../datasets/fashion-mnist/'
    data = input_data.read_data_sets(ori_data_dir,one_hot=False) 
    
    ori_X = np.vstack((data.train.images,data.validation.images))
    ori_Y = np.concatenate((data.train.labels,data.validation.labels))
    ori_test_X = data.test.images
    ori_test_Y = data.test.labels

if args.dim_reduction is None:
    x_dim = args.d_dim
else:
    x_dim = args.z_dim

# In[16]:


sess = ed.get_session()

# In[18]:
if args.conditional:
    nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim+args.T],name='nu_ph')
    de_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim+args.T],name='de_ph')

    prev_nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim+args.T],name='prev_nu_ph')
    prev_de_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim+args.T],name='prev_de_ph')
    c_ph = tf.placeholder(dtype=tf.float32,shape=[None,args.T],name='c_ph')
    net_shape = [x_dim+args.T] + args.hidden_layers + [args.T]
    if args.festimator:
        cl_ratio_model = Cond_Continual_f_Estimator(net_shape=net_shape,nu_ph=nu_ph,de_ph=de_ph,prev_nu_ph=prev_nu_ph,\
                                                prev_de_ph=prev_de_ph,c_ph=c_ph,conv=args.conv,reg=args.reg,cl_constr=args.constr,\
                                                div_type=args.divergence,lambda_reg=args.lambda_reg,lambda_constr=args.lambda_constr,\
                                                local_constr=args.local_constr,c_dim=args.T)
    else:
        cl_ratio_model = Cond_Continual_LogLinear_Estimator(net_shape=net_shape,nu_ph=nu_ph,de_ph=de_ph,prev_nu_ph=prev_nu_ph,\
                                                prev_de_ph=prev_de_ph,c_ph=c_ph,conv=args.conv,reg=args.reg,cl_constr=args.constr,\
                                                div_type=args.divergence,lambda_reg=args.lambda_reg,lambda_constr=args.lambda_constr,\
                                                local_constr=args.local_constr,c_dim=args.T)

else:
    nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim],name='nu_ph')
    de_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim],name='de_ph')

    prev_nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim],name='prev_nu_ph')
    prev_de_ph = tf.placeholder(dtype=tf.float32,shape=[None,x_dim],name='prev_de_ph')
    net_shape = [x_dim] + args.hidden_layers + [1]
    if args.festimator: 
        cl_ratio_model = Continual_f_Estimator(net_shape=net_shape,nu_ph=nu_ph,de_ph=de_ph,prev_nu_ph=prev_nu_ph,\
                                               prev_de_ph=prev_de_ph,reg=args.reg,cl_constr=args.constr,\
                                                div_type=args.divergence,lambda_reg=args.lambda_reg,\
                                                lambda_constr=args.lambda_constr,local_constr=args.local_constr)
    else:
        cl_ratio_model = Continual_LogLinear_Estimator(net_shape=net_shape,nu_ph=nu_ph,de_ph=de_ph,prev_nu_ph=prev_nu_ph,\
                                                prev_de_ph=prev_de_ph,reg=args.reg,cl_constr=args.constr,\
                                                    div_type=args.divergence,lambda_reg=args.lambda_reg,\
                                                    lambda_constr=args.lambda_constr,local_constr=args.local_constr)



cl_ratio_model.estimator.config_train(learning_rate=args.learning_rate,decay=decay,clip=args.grad_clip)

# In[19]:
if args.dim_reduction == 'vae':
    if args.conv:
        ori_X = ori_X.reshape(-1,28,28,1)
        ori_test_X = ori_test_X.reshape(-1,28,28,1)
        d_dim = [28,28,1]
        d_net_shape = [[args.z_dim,512,128*7*7],[[7,7,128],[14,14,64],[28,28,1]]]
        e_net_shape = [[[4,4,1,64],[4,4,64,128]],[128*7*7,512,args.z_dim]]
    else:
        e_net_shape=[512,512]
        d_net_shape=[256,256]
        d_dim = args.d_dim
    
    vtrainer = Continual_VAE(d_dim,args.z_dim,batch_size=200,e_net_shape=e_net_shape,d_net_shape=d_net_shape,prior_std=.1,\
                                epochs=100,reg='l2',learning_rate=0.002,lamb_reg=args.vae_lamb_reg,conv=args.conv)
elif args.dim_reduction == 'dvae':
    vtrainer = Continual_DVAE(args.d_dim,args.z_dim,batch_size=200,e_net_shape=[512,512],d_net_shape=[256,256],\
                                epochs=100,lamb=args.dvae_lamb,learning_rate=0.002,reg='l2',lamb_reg=args.vae_lamb_reg)

if args.pairwise:
    vtrainer.train(ori_X)
# In[20]:


saver = tf.train.Saver()

# In[28]:

save_name = 'sample_ratios_t'



kl = [[],[],[]]

sample_size = args.sample_size
test_sample_size = args.test_sample_size
batch_size = args.batch_size
prev_nu_samples,prev_de_samples,t_prev_nu_samples,t_prev_de_samples = None, None, None, None
T = int(args.T * (args.T - 1) * 0.5 ) if args.pairwise else args.T-1
for t in range(args.start_task,T):
    sample_ratios = pd.DataFrame()
    if args.conditional:
        model_type = 'taskratio_cond'  
    elif args.pairwise:
        model_type = 'taskratio_pw'
    else:
        model_type = 'taskratio'
    
    samples_c,nu_samples,de_samples,t_samples_c,t_nu_samples,t_de_samples = gen_task_samples(t,sample_size,test_sample_size,None,args.T,\
                                                                            ori_X,ori_Y,ori_test_X,ori_test_Y,model_type=model_type,tcs=tcs)

    # dimension reduction before ratio estimation
    if args.dim_reduction in ['vae','dvae']:
        if not args.pairwise:
            if args.dim_reduction == 'vae':
                vtrainer.train(np.vstack([nu_samples[:args.sample_size],de_samples[:args.sample_size]]))
                #vtrainer.train(nu_samples)
            else:
                vtrainer.train(nu_samples,de_samples)
            if t>args.start_task:
                # encode input for previous estimator
                prev_nu_samples = vtrainer.prev_encode(nu_samples)
                prev_de_samples = vtrainer.prev_encode(de_samples)
                if t_nu_samples is not None:
                    t_prev_nu_samples = vtrainer.prev_encode(t_nu_samples)
                    t_prev_de_samples = vtrainer.prev_encode(t_de_samples)
        

        nu_samples = vtrainer.encode(nu_samples)
        de_samples = vtrainer.encode(de_samples)
        t_nu_samples = vtrainer.encode(t_nu_samples)
        t_de_samples = vtrainer.encode(t_de_samples)
        if not args.pairwise:
            vtrainer.update_inference()

    tf.global_variables_initializer().run()
    if args.conditional:
        losses,tlosses,terrs = cl_ratio_model.learning(sess,nu_samples,de_samples,samples_c,t_nu_samples,t_de_samples,\
                                                            t_samples_c,batch_size=batch_size,epoch=args.epoch,print_e=args.print_e,\
                                                            early_stop=args.early_stop,min_epoch=args.min_epoch,\
                                                            prev_nu_samples=prev_nu_samples,prev_de_samples=prev_de_samples,\
                                                            t_prev_nu_samples=t_prev_nu_samples,t_prev_de_samples=t_prev_de_samples)

    else:
        losses,tlosses,terrs = cl_ratio_model.learning(sess,nu_samples,de_samples,t_nu_samples,t_de_samples,\
                                                batch_size=batch_size,epoch=args.epoch,print_e=args.print_e,\
                                                early_stop=args.early_stop,min_epoch=args.min_epoch,\
                                                prev_nu_samples=prev_nu_samples,prev_de_samples=prev_de_samples,\
                                                t_prev_nu_samples=t_prev_nu_samples,t_prev_de_samples=t_prev_de_samples)
    if args.save_model and t==args.start_task:
        saver.save(sess,sub_dir+'model_start_task')


    # save results
    test_samples = de_samples
    if args.conditional:
        test_samples_c = samples_c 
        estimated_ratio = cl_ratio_model.estimator.log_ratio(sess,test_samples,test_samples,c=test_samples_c).reshape(-1)
    else:
        estimated_ratio = cl_ratio_model.estimator.log_ratio(sess,test_samples,test_samples).reshape(-1)
  

    if args.conditional:
        test_samples_c = samples_c 
        estimated_ratio = cl_ratio_model.estimator.log_ratio(sess,test_samples,test_samples,c=test_samples_c)
        if t > 0 and args.continual_ratio:
            estimated_original_ratio = cl_ratio_model.original_log_ratio(sess,test_samples,test_samples,c=test_samples_c)
        else:
            estimated_original_ratio = estimated_ratio 

        sample_ratios['estimated_step_ratio'] = estimated_ratio.sum(axis=1)
        sample_ratios['estimated_original_ratio'] = estimated_original_ratio.sum(axis=1)
        sample_ratios['sample_c'] = np.argmax(test_samples_c,axis=1)
        est_ds = calc_divgenerce('Jensen_Shannon',[estimated_original_ratio,estimated_ratio],test_samples_c,logr=True)
        kl[0].append(est_ds[0][:t+1].mean())        
        kl[1].append(est_ds[1][:t+1].mean())
        print('avg JS',kl[0][-1],kl[1][-1])
    else:
        estimated_ratio = cl_ratio_model.estimator.log_ratio(sess,test_samples,test_samples).reshape(-1)

        if t > args.start_task and args.continual_ratio:
            estimated_original_ratio = cl_ratio_model.original_log_ratio(sess,test_samples,test_samples).reshape(-1)
        else:
            estimated_original_ratio = estimated_ratio 

        sample_ratios['estimated_step_ratio'] = estimated_ratio
        sample_ratios['estimated_original_ratio'] = estimated_original_ratio
        est_ds = calc_divgenerce('Jensen_Shannon',[estimated_original_ratio,estimated_ratio],logr=True)
        kl[0].append(est_ds[0])        
        kl[1].append(est_ds[1])

    print('JS',est_ds[0],est_ds[1])
    sample_ratios.to_csv(sub_dir+save_name+'_t'+str(t+1)+'.csv',index=False)
    
    if t > args.start_task and args.continual_ratio:
        contr = cl_ratio_model.get_cl_constr()
        if args.conditional:
            c_nu_samples, c_de_samples = cl_ratio_model.estimator.concat_condition(nu_samples[:batch_size],de_samples[:batch_size],samples_c[:batch_size])
            feed_dict={cl_ratio_model.estimator.nu_ph:c_nu_samples,\
                        cl_ratio_model.estimator.de_ph:c_de_samples,\
                        cl_ratio_model.estimator.c_ph:samples_c[:batch_size]}
        else:
            feed_dict={cl_ratio_model.estimator.nu_ph:nu_samples,cl_ratio_model.estimator.de_ph:de_samples}

        contr = sess.run(contr,feed_dict)
    else: 
        contr = 1.
    kl[2].append(contr)
    print('constrain check',contr)
    

    # update distributions and model
    if t < args.T - 2 :
        # update model loss 
        if args.conditional:
            cl_ratio_model.update_estimator(sess,t+1)
        elif not args.pairwise:      
            cl_ratio_model.update_estimator(sess)
    # clear memory    
    gc.collect()

# In[39]:


kl = np.array(kl)
np.savetxt(sub_dir+'divergence_compare.csv', kl, delimiter=',')









