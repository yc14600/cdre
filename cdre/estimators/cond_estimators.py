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
from base_models.gans import GAN,fGAN
from base_models.ratio_fgan import Ratio_fGAN
from .estimators import KL_Loglinear_Estimator, LogLinear_Estimator,f_Estimator

class Cond_Loglinear_Estimator(LogLinear_Estimator):

    def __init__(self,net_shape,nu_ph,de_ph,c_ph,coef=None,conv=False,batch_norm=False,ac_fn=tf.nn.relu,\
                reg=None,scope='ratio',batch_train=False, lambda_reg=1.,bayes=False,constr=0.,\
                c_dim=10, *args,**kargs):
        
        self.c_ph = c_ph
        self.c_dim = c_dim
        self.max_c = 1
        super(Cond_Loglinear_Estimator,self).__init__(net_shape,nu_ph,de_ph,coef,conv,batch_norm,ac_fn,reg,scope,\
                batch_train,lambda_reg,bayes,constr)
        
        return



    def concat_condition(self,x_nu,x_de,c):

        x_nu = concat_cond_data(x_nu,c,one_hot=False,conv=self.conv)
        if x_nu is not x_de:
            x_de = concat_cond_data(x_de,c,one_hot=False,conv=self.conv)
        return x_nu,x_de


    def log_ratio(self,sess,x_nu,x_de,c,nu_r=None,de_r=None,coef=None,concat_c=True,*args,**kargs):
        if concat_c:
            x_nu, x_de = self.concat_condition(x_nu,x_de,c)
         
        nu_r = self.nu_r if nu_r is None else nu_r
        de_r = self.de_r if de_r is None else de_r
        coef = self.coef if coef is None else coef

        nu_r  = nu_r * self.c_ph
        de_r = de_r * self.c_ph

        de_mean = self.exp_de_mean(de_r)
        log_r = nu_r - tf.log(de_mean) 
        log_r *= self.c_ph
        
        
        if self.nu_ph.shape[0].value is None:
            feed_dict = {self.is_training:False} if self.batch_norm else {}
            feed_dict.update({self.nu_ph:x_nu,self.de_ph:x_de,self.c_ph:c})
            rt = sess.run(log_r,feed_dict)

        else:
            batch_size = self.nu_ph.shape[0].value
            ii = 0
            iters = int(np.ceil(x_nu.shape[0]/batch_size))
            rt = np.zeros([x_nu.shape[0],self.c_ph.shape[-1].value])

            for _ in range(iters):
                start = ii
                feed_dict,ii = self.update_feed_dict(x_nu,x_de,ii,batch_size,c)
                end = ii if ii < x_nu.shape[0] and ii > start else x_nu.shape[0]

                rt[start:end] = sess.run(log_r,feed_dict)[:end-start]
        
        rt[np.isnan(rt)*(c==0)] = 0.
        return rt


    def ratio(self,sess,x_nu,x_de,c,nu_r=None,de_r=None,coef=None,*args,**kargs):
        log_r = self.log_ratio(sess,x_nu,x_de,c,nu_r,de_r,coef)
        return np.exp(log_r)


    def learning(self,sess,nu_samples,de_samples,samples_c,test_nu_samples=None,test_de_samples=None,\
                test_samples_c=None,batch_size=64,epoch=50,print_e=1,nu_dist=None,de_dist=None,early_stop=False,\
                tol=0.,update_feed_dict=None,min_epoch=50,concat_c=True,*args,**kargs):
        if concat_c:
            nu_samples, de_samples = self.concat_condition(nu_samples,de_samples,samples_c)
            test_nu_samples, test_de_samples = self.concat_condition(test_nu_samples,test_de_samples,test_samples_c)
        losses,tlosses,true_errs = super(Cond_Loglinear_Estimator,self).learning(sess,nu_samples,de_samples,test_nu_samples,test_de_samples,\
                batch_size,epoch,print_e,None,None,early_stop,tol,update_feed_dict,min_epoch,samples_c,test_samples_c)

        return losses,tlosses,true_errs


    def update_feed_dict(self,nu_samples,de_samples,ii,batch_size,samples_c=None,*args,**kargs):

        ii_bk = ii
        feed_dict,ii = super(Cond_Loglinear_Estimator,self).update_feed_dict(nu_samples,de_samples,ii,batch_size)
        if samples_c is not None:
            c_batch,_,__ = get_next_batch(samples_c,batch_size,ii_bk,repeat=True)
            feed_dict.update({self.c_ph:c_batch})

        return feed_dict,ii

    def exp_de_mean(self,de_r):        
        de_mean = self.cond_mean(tf.exp(de_r)*self.c_ph)
        return de_mean

    def cond_mean(self,x):
        mask = np.zeros(self.c_dim).astype(np.float32)
        mask[self.max_c:] = 1. # to avoid log0
        tmp = tf.reduce_sum(self.c_ph,axis=0)
        return (tf.reduce_sum(x,axis=0)+mask)/(tmp+mask)

class Cond_KL_Loglinear_Estimator(Cond_Loglinear_Estimator):
    
    def set_loss(self):
        nu_r = self.nu_r * self.c_ph
        de_r = self.de_r * self.c_ph

        c_num = tf.reduce_sum(self.c_ph,axis=0)
        c_prop = c_num /tf.reduce_sum(c_num)

        nu_mean = self.cond_mean(nu_r) 
        de_mean = self.exp_de_mean(de_r)
        lloss = -nu_mean + tf.log(de_mean)
        if self.constr:
            #todo: need check
            print('local constr',self.constr)
            ir = self.exp_de_mean(de_r) * tf.exp(-1.* nu_r) * self.c_ph
            loss = lloss + tf.square(self.cond_mean(ir) - 1.)    
        else:
            loss = lloss
        
        self.ll_loss = tf.reduce_sum(lloss*c_prop) # for early stop
        loss = loss * c_prop
       
        # average over all conditions
        loss = tf.reduce_sum(loss)

        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())

        return loss + self.lambda_reg * reg_loss 


class Cond_f_Estimator(Cond_Loglinear_Estimator,f_Estimator):

    def __init__(self,net_shape,nu_ph,de_ph,c_ph,coef=None,conv=False,batch_norm=False,ac_fn=tf.nn.relu,\
                reg=None,scope='ratio',batch_train=False, lambda_reg=1.,bayes=False,constr=0.01,\
                divergence='KL', c_dim=10, *args,**kargs):
        self.c_ph = c_ph
        self.c_dim = c_dim
        self.max_c = 1
        f_Estimator.__init__(self,net_shape,nu_ph,de_ph,coef,conv,batch_norm,ac_fn,reg,scope,\
                batch_train,lambda_reg,bayes,constr,divergence)
        
        return

    def ratio(self,sess,x_nu,x_de,c,nu_r=None,de_r=None,coef=None,concat_c=True,*args,**kargs):
        if concat_c:
            x_nu, x_de = self.concat_condition(x_nu,x_de,c)
        
        nu_r = self.nu_r if nu_r is None else nu_r
        nu_r  = nu_r * self.c_ph

        idf_gf = Ratio_fGAN.get_idf_gf(self.divergence)
        r = idf_gf(nu_r)*self.c_ph
        '''
        feed_dict={self.nu_ph:x_nu,self.c_ph:c}
        if self.batch_norm:
            feed_dict.update({self.is_training:False})
        '''
        if self.nu_ph.shape[0].value is None:
            feed_dict = {self.is_training:False} if self.batch_norm else {}
            feed_dict.update({self.nu_ph:x_nu,self.c_ph:c})
            return sess.run(r,feed_dict)
        else:
            batch_size = self.nu_ph.shape[0].value
            ii = 0
            iters = int(np.ceil(x_nu.shape[0]/batch_size))
            rlt = np.zeros([x_nu.shape[0],self.c_ph.shape[-1].value])
            for _ in range(iters):
                start = ii
                feed_dict,ii = self.update_feed_dict(x_nu,x_de,ii,batch_size,c)
                end = ii if ii < x_nu.shape[0] and ii > start else x_nu.shape[0]
                rlt[start:end] = sess.run(r,feed_dict)
            return rlt

        



    def log_ratio(self,sess,x_nu,x_de,c,nu_r=None,de_r=None,coef=None,concat_c=True,*args,**kargs):
        if concat_c:
            x_nu, x_de = self.concat_condition(x_nu,x_de,c)
        
        nu_r = self.nu_r if nu_r is None else nu_r
        nu_r  = nu_r * self.c_ph

        idf_gf = Ratio_fGAN.get_idf_gf(self.divergence)
        log_r = tf.log(idf_gf(nu_r))*self.c_ph

        feed_dict={self.nu_ph:x_nu,self.c_ph:c}
        if self.batch_norm:
            feed_dict.update({self.is_training:False})

        if self.nu_ph.shape[0].value is None:
            feed_dict = {self.is_training:False} if self.batch_norm else {}
            feed_dict.update({self.nu_ph:x_nu,self.c_ph:c})
            return sess.run(log_r,feed_dict)
        else:
            batch_size = self.nu_ph.shape[0].value
            ii = 0
            iters = int(np.ceil(x_nu.shape[0]/batch_size))
            rlt = np.zeros([x_nu.shape[0],self.c_ph.shape[-1].value])
            for _ in range(iters):
                start = ii
                feed_dict,ii = self.update_feed_dict(x_nu,x_de,ii,batch_size,c)
                end = ii if ii < x_nu.shape[0] and ii > start else x_nu.shape[0]
                rlt[start:end] = sess.run(log_r,feed_dict)
            return rlt

        
        

    def get_constr(self):
        
        if self.divergence == 'Pearson':
            constr = tf.square(self.cond_mean(0.5*self.de_r+1.) - 1.)
        
        elif self.divergence == 'Hellinger':
            constr = tf.square(self.cond_mean(tf.exp(2.*self.de_r)) - 1.)
        
        else:
            constr = tf.square(self.cond_mean(tf.exp(self.de_r)) - 1.)
                
        return constr

    def set_loss(self):

        act_fn,conj_f = fGAN.get_act_conj_fn(self.divergence)
        nu_r = self.nu_r * self.c_ph
        de_r = self.de_r * self.c_ph
        loss = -tf.reduce_mean(act_fn(nu_r),axis=0) + tf.reduce_mean(conj_f(act_fn(de_r)),axis=0)
        loss += self.get_constr()*self.constr

        c_num = tf.reduce_sum(self.c_ph,axis=0)
        c_prop = c_num /tf.reduce_sum(c_num)
        loss = loss * c_prop
       
        # average over all conditions
        loss = tf.reduce_sum(loss)
        self.ll_loss = loss # only for validation
        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())

        
        return loss + self.lambda_reg * reg_loss 

