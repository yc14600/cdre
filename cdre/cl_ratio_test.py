


import numpy as np
import tensorflow as tf
import pandas as pd
import argparse

import time
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt


from estimators import LogLinear_Estimator,Continual_LogLinear_Estimator,Continual_f_Estimator
from utils.model_util import define_dense_layer,LinearRegression
from utils.test_util import *
from base_models.mixture_models import MixDiagGaussian

from scipy.stats import multivariate_normal, norm



class generator(object):
    def __init__(self,mean,std,d_dim):
        self.mean = mean
        self.std = std
        self.d_dim = d_dim
        
    def draw_samples(self,sample_size):
        return gen_samples(sample_size,self.d_dim,self.mean,self.std)
    

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

args = parser.parse_args()

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

tf.set_random_seed(args.seed)
np.random.seed(args.seed)
decay = None #(1000,0.1)


if args.result_path[-1] != '/':
    path = args.result_path+'/'
else:
    path = args.result_path

if not os.path.exists(path):
    os.makedirs(path)

if args.f_name:
    f_name = args.f_name
else:
    f_name = 'cl_ratio_test_d'+str(args.d_dim)+'_'+'-'.join(time.ctime().replace(':','').split(' '))+'/'
sub_dir = os.path.join(path,f_name)
if not os.path.exists(sub_dir):
    os.mkdir(sub_dir)

print(args)
with open(os.path.join(sub_dir,'configures.txt'),'w') as f:
    f.write(str(args))

if not args.continual_ratio:
    args.constr = False


# dist = 'Normal'
if args.dataset == 'gaussian':
    if args.num_components == 1:
        delta_mean = args.delta_mean #if args.delta_mean != 0. else args.delta_list[0]#np.random.uniform(-0.5,0.5)
        ori_nu_mean, ori_nu_std = 0., 1.
        de_mean = ori_nu_mean + delta_mean 
        de_std = ori_nu_std + args.delta_std 
        nu_mean, nu_std = ori_nu_mean, ori_nu_std

        nu_dist,de_dist = get_dists(args.d_dim,nu_mean,nu_std,de_mean,de_std)

    else:
        # mixture Gaussian
        ori_nu_mean = [0. + k * 3. for k in range(args.num_components)]
        ori_nu_std = [1.]*args.num_components
        nu_mean, nu_std = ori_nu_mean, ori_nu_std
        de_mean = ori_nu_mean[:-1]
        de_std = ori_nu_std[:-1]
        if len(args.component_weights) > 0:
            nu_pi = args.component_weights
            de_pi = normalize(nu_pi[:-1])
        else:
            nu_pi = [1./len(nu_mean)]*len(nu_mean)
            de_pi = [1./len(de_mean)]*len(de_mean)

        print('check mean',ori_nu_mean,de_mean)
        print('check pi',nu_pi,de_pi)

        nu_dist,de_dist = get_dists(args.d_dim,nu_mean,nu_std,de_mean,de_std,nu_pi,de_pi)
    ori_nu_dist = nu_dist

elif args.dataset == 'stock':
    dataset = np.load(args.datapath)
             

if args.task_type == 'regression':  
    print('define regression true func')
    def true_f(x):
        return 1.2*np.power(x,3)-2.4*np.power(x,2)+3.6*x

d_dim = args.d_dim



nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,d_dim],name='nu_ph')
de_ph = tf.placeholder(dtype=tf.float32,shape=[None,d_dim],name='de_ph')

prev_nu_ph = tf.placeholder(dtype=tf.float32,shape=[None,d_dim],name='prev_nu_ph')
prev_de_ph = tf.placeholder(dtype=tf.float32,shape=[None,d_dim],name='prev_de_ph')


sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

net_shape = [d_dim] + args.hidden_layers + [1]

if args.festimator: 
    cl_ratio_model = Continual_f_Estimator(net_shape=net_shape,nu_ph=nu_ph,de_ph=de_ph,prev_nu_ph=prev_nu_ph,\
                                               prev_de_ph=prev_de_ph,reg=args.reg,cl_constr=args.constr,\
                                                div_type=args.divergence,lambda_reg=args.lambda_reg,\
                                                lambda_constr=args.lambda_constr,bayes=args.bayes,local_constr=args.local_constr)
else:
    cl_ratio_model = Continual_LogLinear_Estimator(net_shape=net_shape,nu_ph=nu_ph,de_ph=de_ph,prev_nu_ph=prev_nu_ph,\
                                               prev_de_ph=prev_de_ph,reg=args.reg,cl_constr=args.constr,\
                                                div_type=args.divergence,lambda_reg=args.lambda_reg,\
                                                lambda_constr=args.lambda_constr,bayes=args.bayes,local_constr=args.local_constr)


cl_ratio_model.estimator.config_train(learning_rate=args.learning_rate,decay=decay)

saver = tf.train.Saver()

save_name = 'sample_ratios_t'

kl = [[],[],[],[],[]]
sample_ratios = pd.DataFrame()

for t in range(args.T):
    restart = False
    if args.unlimit_samples:
        nu_generator = generator(mean=nu_mean,std=nu_std,d_dim=args.d_dim)
        de_generator = generator(mean=de_mean,std=de_std,d_dim=args.d_dim)

    #if args.continual_ratio:
    if args.dataset == 'gaussian':
        nu_samples,de_samples = get_samples(args.sample_size,nu_dist,de_dist)
    elif args.dataset == 'stock':
        # remove the 1st dimenstion which is time stamp
        nu_samples = dataset[t][:args.sample_size,1:]
        t_nu_samples = dataset[t][-args.test_sample_size:,1:]
        de_samples = dataset[t+1][:args.sample_size,1:]
        t_de_samples = dataset[t+1][-args.test_sample_size:,1:] 
    #else:
    #    nu_samples,de_samples = get_samples(args.sample_size,nu_dist,de_dist,de_sample_size=args.sample_size)
    
        
    print('check sample',nu_samples.shape)
    if args.validation: 
        if args.dataset == 'gaussian':
            t_nu_samples,t_de_samples = get_samples(args.test_sample_size,nu_dist,de_dist)

    else:
        t_nu_samples,t_de_samples = None, None    
    
    tf.global_variables_initializer().run(session=sess)

    # load checkpoint for first task
    if t==0 and len(args.warm_start) != 0:
        saver.restore(sess,args.warm_start)
    
    else:

        if args.unlimit_samples:
            losses,tlosses,terrs = cl_ratio_model.estimator.learning(sess,nu_generator,de_generator,t_nu_samples,t_de_samples,\
                                                                batch_size=args.batch_size,epoch=args.epoch,print_e=args.print_e,\
                                                                nu_dist=nu_dist,de_dist=de_dist,early_stop=args.early_stop)
        else:
            losses,tlosses,terrs = cl_ratio_model.estimator.learning(sess,nu_samples,de_samples,t_nu_samples,t_de_samples,\
                                                                batch_size=args.batch_size,epoch=args.epoch,print_e=args.print_e,\
                                                                nu_dist=None,de_dist=None,early_stop=args.early_stop)
    if args.save_model:
        saver.save(sess,os.path.join(sub_dir,'model_task'+str(t)))


    # save results
    test_samples = np.vstack([de_samples,t_de_samples]) #de_samples #
    estimated_ratio = cl_ratio_model.estimator.log_ratio(sess,test_samples,test_samples).reshape(-1)
    if t > 0 and args.continual_ratio:
        estimated_original_ratio = cl_ratio_model.original_log_ratio(sess,test_samples,test_samples).reshape(-1)
    else:
        estimated_original_ratio = estimated_ratio
    
    sample_ratios['estimated_log_ratio'] = estimated_ratio
    sample_ratios['estimated_original_log_ratio'] = estimated_original_ratio
    
    if args.task_type == 'div':
        if args.dataset == 'gaussian':
            true_ratio = -de_dist.log_prob(test_samples) + ori_nu_dist.log_prob(test_samples)
            true_step_ratio = -de_dist.log_prob(test_samples) + nu_dist.log_prob(test_samples)
            sample_ratios['true_log_ratio'] = true_ratio
            sample_ratios['true_step_log_ratio'] = true_step_ratio

            true_kl = Gaussian_KL(de_dist,ori_nu_dist,args.d_dim) 
            kl[0].append(true_kl)
            true_step_kl = Gaussian_KL(de_dist, nu_dist,args.d_dim) 
            kl[2].append(true_step_kl)
            print('true kls', true_kl,true_step_kl)
        
        est_kl = np.mean(- estimated_original_ratio)
        kl[1].append(est_kl)
        if args.restart and est_kl > args.restart_th:
            print('restart at task {}'.format(t))
            restart = True
        step_kl = np.mean(- estimated_ratio)
        kl[3].append(step_kl)
        print('kls', est_kl,step_kl)
        if t > 0 and args.continual_ratio:
            contr = cl_ratio_model.get_cl_constr()
            contr = sess.run(contr,feed_dict={cl_ratio_model.estimator.nu_ph:nu_samples,cl_ratio_model.estimator.de_ph:de_samples})
        else: 
            contr = 1.
        kl[4].append(contr)
        print('constrain check',contr)


    elif args.task_type == 'regression':
        if t == 0:
            nu_y = true_f(nu_samples)
            nu_y += 1.5*(0.5 - np.random.uniform(size=nu_y.shape))  # add noise
            ori_nu_samples = nu_samples

            y_ph = tf.placeholder(dtype=tf.float32,shape=[None,1],name='y_ph')
            w_ph = tf.placeholder(dtype=tf.float32,shape=[None,1],name='w_ph')
            rg_model = LinearRegression(x_ph=nu_ph,y_ph=y_ph,in_dim=args.d_dim,out_dim=1,Bayes=False,logistic=False,w_ph=w_ph)
            rg_model.config_train_opt(learning_rate=0.001)
            rg_model2 = LinearRegression(x_ph=nu_ph,y_ph=y_ph,in_dim=args.d_dim,out_dim=1,Bayes=False,logistic=False,w_ph=1.)
            rg_model2.config_train_opt(learning_rate=0.001)

        elif t == args.T-1:
            de_y = true_f(de_samples)
            de_y += 1.5*(0.5 - np.random.uniform(size=de_y.shape))
            ratio = np.exp(estimated_original_ratio).reshape(-1,1)
            print('check ratio',np.max(ratio),np.min(ratio))
            rg_model.fit(num_iter=20000,x_train=de_samples,y_train=de_y,batch_size=200,sess=sess,weights=ratio,print_iter=1000)
            rg_model2.fit(num_iter=20000,x_train=de_samples,y_train=de_y,batch_size=200,sess=sess,weights=None,print_iter=1000)
            
            #### plot ####
            X_plot = np.linspace(0, 3, 1000)[:, None]
            stime = time.time()
            py = sess.run(rg_model.y,feed_dict={rg_model.x_ph:X_plot})
            py2 = sess.run(rg_model2.y,feed_dict={rg_model2.x_ph:X_plot})
            sns.set_style('darkgrid')
            lw = 2
            plt.scatter(ori_nu_samples[:100], nu_y[:100],marker='>', label=r'$D_1$')
            plt.scatter(de_samples[:100], de_y[:100],marker='*', label=r'$D_t$')
            plt.plot(X_plot, py, color='turquoise', lw=lw,
                    label=r'$LR+IW:D_t$')
            plt.plot(X_plot, py2, color='red', lw=lw,
                    label=r'$LR:D_t$')
            plt.legend(loc="best",  scatterpoints=1,fontsize=12)
            plt.ylim(-2, 15)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(os.path.join(sub_dir,'cv_shift_regression.pdf'))

    sample_ratios.to_csv(os.path.join(sub_dir,save_name+'_t'+str(t+1)+'.csv'),index=False)
    # visualizations
    if args.vis:
        plt.plot(test_samples[:,0],true_ratio,'.')
        plt.plot(test_samples[:,0],estimated_original_ratio,'.')
        plt.plot(test_samples[:,0],estimated_ratio,'.')

        plt.xlabel('the first dimension of x',fontsize=15)
        lgd = plt.legend([r'$\log r^*_t$',r'$\log r_t$',r'$\log r_{\theta_t}$'],fontsize=15,bbox_to_anchor=(0.9, 1.2),
            ncol=3)
        plt.savefig(os.path.join(sub_dir,args.task_type+'_cl_ratio_vis_d'+str(args.d_dim)+'_task'+str(t+1)+'.pdf'),bbox_extra_artists=([lgd]), bbox_inches='tight')
        plt.close()

        if len(args.warm_start) == 0 or t > 0:
            N = len(losses)
            plt.plot(range(N),losses)
            plt.plot(range(N),tlosses)

            plt.xlabel('number of epochs',fontsize=12)
            lgd=plt.legend(['training loss','validation loss'],fontsize=12)
            plt.savefig(os.path.join(sub_dir,'loss_compare_t'+str(t)+'.pdf'),bbox_extra_artists=([lgd]), bbox_inches='tight')
            plt.close()
    
    # update distributions and model
    if t < args.T - 1 :
        if args.dataset=='gaussian':
            if args.continual_ratio:
                nu_mean, nu_std = de_mean, de_std


            if args.num_components > 1:
                if args.continual_ratio:
                    nu_pi = de_pi
                de_mean = de_mean[:-1]
                print('check de mean',de_mean)
                de_std = de_std[:-1]        
                de_pi = normalize(de_pi[:-1])
                print('component weights',nu_pi,de_pi)
                #print('de mean',de_mean,'de std',de_std)
                nu_dist,de_dist = get_dists(args.d_dim,nu_mean,nu_std,de_mean,de_std,nu_pi,de_pi)
            else:
                #if args.delta_mean == 0. :
                #    delta_mean =  args.delta_list[t+1] #np.random.uniform(-0.5,0.5)

                print('delta par',args.delta_mean)
                            
                de_mean += delta_mean
                de_std += args.delta_std 
                print(nu_mean,nu_std,de_mean,de_std,ori_nu_mean,ori_nu_std)
                nu_dist,de_dist = get_dists(args.d_dim,nu_mean,nu_std,de_mean,de_std)
   
                

        # update model loss 
        if args.continual_ratio:    
            cl_ratio_model.update_estimator(sess,increase_constr=args.increase_constr,nu_samples=nu_samples,de_samples=de_samples,restart=restart)

# In[39]:
if args.dataset == 'gaussian':
    kl = np.array(kl)
else:
    kl = np.vstack([kl[1],kl[3],kl[4]])
np.savetxt(os.path.join(sub_dir,'divergence_compare.csv'), kl, delimiter=',')



# In[45]:
if args.vis:
    plt.plot(range(1,args.T+1),kl[0])
    plt.plot(range(1,args.T+1),kl[1])
    plt.plot(range(1,args.T+1),kl[2])
    plt.plot(range(1,args.T+1),kl[3])
    plt.xlabel('t (index of tasks)',fontsize=14)
    #lgd=plt.legend([r'$D_{KL}(P(x)||P_{\theta_t}(x))$',r'$\widehat{D}_{KL}(P(x)||P_{\theta_t}(x))$',\
    #                r'$D_{KL}(P_{\theta_{t-1}}(x)||P_{\theta_t}(x))$',r'$\widehat{D}_{KL}(P_{\theta_{t-1}}(x)||P_{\theta_t}(x))$'],fontsize=14)#,bbox_to_anchor=(1., 1.)
    plt.savefig(os.path.join(sub_dir,'KL_compare.pdf'))
    plt.close()





