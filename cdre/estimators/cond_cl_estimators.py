from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import scipy as sp
import pandas as pd
import six
import tensorflow as tf
from abc import ABC, abstractmethod
from utils.train_util import config_optimizer, get_next_batch,concat_cond_data,one_hot_encoder
from base_models.gans import GAN
from .cl_estimators import Continual_Estimator,Continual_LogLinear_Estimator,Continual_f_Estimator
from .cond_estimators import Cond_KL_Loglinear_Estimator,Cond_Loglinear_Estimator,Cond_f_Estimator



class Cond_Continual_LogLinear_Estimator(Continual_LogLinear_Estimator):

    def __init__(self,net_shape,nu_ph,de_ph,prev_nu_ph,prev_de_ph,c_ph,coef=None,conv=False,\
                    batch_norm=False,ac_fn=tf.nn.relu,reg=None,batch_train=False,\
                    cl_constr=True,div_type='KL',lambda_reg=1.,lambda_constr=1.,\
                    bayes=False,local_constr=False,c_dim=10,*args,**kargs):
        
        self.t = 0
        super(Cond_Continual_LogLinear_Estimator,self).__init__(net_shape,nu_ph,de_ph,prev_nu_ph,prev_de_ph,coef,conv,\
                    batch_norm,ac_fn,reg,batch_train,cl_constr,div_type,lambda_reg,lambda_constr,bayes,local_constr,c_dim,c_ph)

        self.estimator.max_c = 1    
        return
    
    def build_estimator(self,net_shape,nu_ph,de_ph,coef=None,conv=False,batch_norm=False,\
                        ac_fn=tf.nn.relu,reg=None,batch_train=False,div_type='KL',lambda_reg=1.,\
                        bayes=False,local_constr=False,c_dim=10,c_ph=None,*args,**kargs):
        
        if batch_norm:
            raise NotImplementedError('Batch norm not supported!')

        if div_type == 'KL':
            return Cond_KL_Loglinear_Estimator(net_shape,nu_ph,de_ph,c_ph,coef,conv,batch_norm,ac_fn,reg,\
                                            batch_train=batch_train,lambda_reg=lambda_reg,bayes=bayes,\
                                            constr=local_constr,c_dim=c_dim)

        else:
            raise NotImplementedError

    
    def update_estimator(self,sess,t,increase_constr=False):
        self.t = t
        self.estimator.max_c = t+1
        if increase_constr:
            self.lambda_constr = self.lambda_constr * (t+1) / t
            print('update lambda_ce',self.lambda_constr)
        self.prev_nu_r,self.prev_de_r = self.save_prev_estimator(sess)
        self.prev_nu_r = tf.clip_by_value(self.prev_nu_r,-60.,60.)
        self.prev_de_r = tf.clip_by_value(self.prev_de_r,-60.,60.)

        c_mask = np.ones([self.estimator.c_dim,1],dtype=np.float32)
        c_mask[t] = 0.
        c_mask = tf.matmul(self.estimator.c_ph, c_mask)
        #print('check update',c_mask.shape,self.prev_de_r.shape)

        self.estimator.nu_r = self.estimator.nu_H[-1] - self.prev_nu_r * c_mask
        self.estimator.de_r = self.estimator.de_H[-1] - self.prev_de_r * c_mask
        self.estimator.nu_r = tf.clip_by_value(self.estimator.nu_r,-60.,60.)
        self.estimator.de_r = tf.clip_by_value(self.estimator.de_r,-60.,60.)
        self.update_train(self.estimator)

        return


    def get_cl_constr(self):
        estimator = self.estimator

        de_r = estimator.de_r * self.estimator.c_ph
        prev_nu_r = self.prev_nu_r * self.estimator.c_ph
        de_h = estimator.de_H[-1] * self.estimator.c_ph
        #print('check constr',self.t)

        constr = tf.square(tf.div(self.estimator.exp_de_mean(de_r)*self.estimator.exp_de_mean(prev_nu_r),\
                                self.estimator.exp_de_mean(de_h)) - 1.)
        
        # doesn't use this constrain for current task
        c_mask = np.ones(self.estimator.c_dim).astype(np.float32)
        c_mask[self.t] = 0.
        constr *= c_mask

        c_num = tf.reduce_sum(self.estimator.c_ph,axis=0)
        c_num *= c_mask
        c_prop = c_num /tf.reduce_sum(c_num)


        return tf.reduce_sum(constr*c_prop) #only add this constrain when t > 1

    
    def original_ratio(self,sess,x,x_de,c,*args,**kargs):
            
        r = self.estimator.ratio(sess,x,x_de,c,\
                                    nu_r=self.estimator.nu_H[-1],\
                                    de_r=self.estimator.de_H[-1])        
        
        return r


    def original_log_ratio(self,sess,x,x_de,c,*args,**kargs):
        
        r = self.estimator.log_ratio(sess,x,x_de,c,\
                                    nu_r=self.estimator.nu_H[-1],\
                                    de_r=self.estimator.de_H[-1])
        return r

    

    def prev_ratio(self,sess,x,x_de,c,*args,**kargs):

        x, x_de = self.estimator.concat_condition(x,x_de,c)  

        return super(Cond_Continual_LogLinear_Estimator,self).prev_ratio(sess,x,x_de)   


    def learning(self,sess,nu_samples,de_samples,samples_c,test_nu_samples=None,test_de_samples=None,\
                test_samples_c=None,batch_size=64,epoch=50,print_e=1,nu_dist=None,de_dist=None,early_stop=False,\
                tol=0.,update_feed_dict=None,min_epoch=50,prev_nu_samples=None,prev_de_samples=None,\
                t_prev_nu_samples=None,t_prev_de_samples=None,*args,**kargs):

        self.prev_map = {}
        if prev_nu_samples is not None:   
            print('check prev shape',prev_nu_samples.shape,prev_de_samples.shape,samples_c.shape) 
            prev_nu_samples,prev_de_samples = self.estimator.concat_condition(prev_nu_samples,prev_de_samples,samples_c)                   
            self.prev_map[id(samples_c)] = (prev_nu_samples,prev_de_samples)
            if test_nu_samples is not None:
                t_prev_nu_samples, t_prev_de_samples = self.estimator.concat_condition(t_prev_nu_samples,t_prev_de_samples,test_samples_c)                   
                self.prev_map[id(test_samples_c)] = (t_prev_nu_samples,t_prev_de_samples)
        
        return self.estimator.learning(sess,nu_samples,de_samples,samples_c,test_nu_samples,test_de_samples,test_samples_c,batch_size,epoch,print_e,\
                                    nu_dist,de_dist,early_stop,tol,self.update_feed_dict,min_epoch,*args,**kargs)


    def update_feed_dict(self,nu_samples,de_samples,ii,batch_size,samples_c,*args,**kargs):

        feed_dict = {self.estimator.is_training:is_training} if self.estimator.batch_norm else {}
        ii_bk = ii

        nu_batch,_,ii = get_next_batch(nu_samples,batch_size,ii,repeat=True)
        de_batch,_,__ = get_next_batch(de_samples,batch_size,ii_bk,repeat=True)

        if len(self.prev_map)>0:
            prev_nu_batch,_,__ = get_next_batch(self.prev_map[id(samples_c)][0],batch_size,ii_bk,repeat=True)
            prev_de_batch,_,__ = get_next_batch(self.prev_map[id(samples_c)][1],batch_size,ii_bk,repeat=True)
            feed_dict.update({self.estimator.nu_ph:nu_batch,self.estimator.de_ph:de_batch,\
                                self.prev_nu_ph:prev_nu_batch,self.prev_de_ph:prev_de_batch})
        else:

            feed_dict.update({self.estimator.nu_ph:nu_batch,self.estimator.de_ph:de_batch})

        c_batch,_,__ = get_next_batch(samples_c,batch_size,ii_bk,repeat=True)
        feed_dict.update({self.estimator.c_ph:c_batch})

        return feed_dict,ii


class Cond_Continual_f_Estimator(Continual_f_Estimator,Cond_Continual_LogLinear_Estimator):

    def __init__(self,net_shape,nu_ph,de_ph,prev_nu_ph,prev_de_ph,c_ph,coef=None,conv=False,\
                    batch_norm=False,ac_fn=tf.nn.relu,reg=None,batch_train=False,\
                    cl_constr=True,div_type='KL',lambda_reg=1.,lambda_constr=1.,\
                    bayes=False,local_constr=False,c_dim=10,*args,**kargs):
        
        self.t = 0
        Continual_f_Estimator.__init__(self,net_shape,nu_ph,de_ph,prev_nu_ph,prev_de_ph,coef,conv,\
                    batch_norm,ac_fn,reg,batch_train,cl_constr,div_type,lambda_reg,lambda_constr,bayes,local_constr,c_dim,c_ph)

        self.estimator.max_c = 1    

        return

    def learning(self,sess,nu_samples,de_samples,samples_c,test_nu_samples=None,test_de_samples=None,\
                test_samples_c=None,batch_size=64,epoch=50,print_e=1,nu_dist=None,de_dist=None,early_stop=False,\
                tol=0.,update_feed_dict=None,min_epoch=50,prev_nu_samples=None,prev_de_samples=None,\
                t_prev_nu_samples=None,t_prev_de_samples=None,*args,**kargs):

        return Cond_Continual_LogLinear_Estimator.learning(self,sess,nu_samples,de_samples,samples_c,test_nu_samples,test_de_samples,\
                test_samples_c,batch_size,epoch,print_e,nu_dist,de_dist,early_stop,\
                tol,update_feed_dict,min_epoch,prev_nu_samples,prev_de_samples,\
                t_prev_nu_samples,t_prev_de_samples,*args,**kargs)

    
    def update_feed_dict(self,nu_samples,de_samples,ii,batch_size,samples_c,*args,**kargs):

        return Cond_Continual_LogLinear_Estimator.update_feed_dict(self,nu_samples,de_samples,ii,batch_size,samples_c,*args,**kargs)

    
    def build_estimator(self,net_shape,nu_ph,de_ph,coef=None,conv=False,batch_norm=False,\
                        ac_fn=tf.nn.relu,reg=None,batch_train=False,div_type='KL',lambda_reg=1.,\
                        bayes=False,local_constr=False,c_dim=10,c_ph=None,*args,**kargs):
        
        if batch_norm:
            raise NotImplementedError('Batch norm not supported!')

        print('div',div_type)
        return Cond_f_Estimator(net_shape,nu_ph,de_ph,c_ph,coef,conv,batch_norm,ac_fn,\
                reg, batch_train=batch_train, lambda_reg=lambda_reg,bayes=bayes,constr=local_constr,\
                divergence=div_type, c_dim=c_dim)


    def update_estimator(self,sess,t):

        self.t = t
        self.estimator.max_c = t+1
        self.estimator.constr = self.estimator.constr * (t+1) / t
        print('update local_constr',self.estimator.constr)
        self.prev_nu_r,self.prev_de_r = self.save_prev_estimator(sess)

        c_mask = np.ones([self.estimator.c_dim,1],dtype=np.float32)
        c_mask[t] = 0.
        c_mask = tf.matmul(self.estimator.c_ph, c_mask)
        c_mask2 = 1. - c_mask
        #print('check update',c_mask.shape,self.prev_de_r.shape)

        if self.div_type == 'KL':
            self.estimator.nu_r = (self.estimator.nu_H[-1] - self.prev_nu_r + 1.)*c_mask + self.estimator.nu_H[-1]*c_mask2
            self.estimator.de_r = (self.estimator.de_H[-1] - self.prev_de_r + 1.)*c_mask + self.estimator.de_H[-1]*c_mask2
        elif self.div_type == 'Pearson': 
            nr = tf.clip_by_value((self.estimator.nu_H[-1]+2.)/(self.prev_nu_r+2.),-1e15,1e15)
            dr = tf.clip_by_value((self.estimator.de_H[-1]+2.)/(self.prev_de_r+2.),-1e15,1e15)

            self.estimator.nu_r = (2.*(nr - 1.))*c_mask + self.estimator.nu_H[-1]*c_mask2
            self.estimator.de_r = (2.*(dr - 1.))*c_mask + self.estimator.de_H[-1]*c_mask2
        else:
            self.estimator.nu_r = self.estimator.nu_H[-1] - self.prev_nu_r*c_mask 
            self.estimator.de_r = self.estimator.de_H[-1] - self.prev_de_r*c_mask
            #self.estimator.nu_r = tf.clip_by_value(self.estimator.nu_r,-30.,30.)
            #self.estimator.de_r = tf.clip_by_value(self.estimator.de_r,-30.,30.)
        self.update_train(self.estimator)

        return
    

    def original_ratio(self,sess,x,x_de,c,*args,**kargs):
            
        r = self.estimator.ratio(sess,x,x_de,c,\
                                    nu_r=self.estimator.nu_H[-1],\
                                    de_r=self.estimator.de_H[-1])        
        
        return r


    def original_log_ratio(self,sess,x,x_de,c,*args,**kargs):
        
        r = self.estimator.log_ratio(sess,x,x_de,c,\
                                    nu_r=self.estimator.nu_H[-1],\
                                    de_r=self.estimator.de_H[-1])
        return r

