from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import scipy as sp
import six
import tensorflow as tf
from abc import ABC, abstractmethod
from utils.train_util import config_optimizer, get_vars_by_scope, normal_logpdf
from utils.model_util import *
from .estimators import Estimator, KL_Loglinear_Estimator, Chi_Loglinear_Estimator,f_Estimator
from base_models.gans import GAN
from scipy.stats import norm


class Continual_Estimator(ABC):
    def __init__(self,net_shape,nu_ph,de_ph,prev_nu_ph,prev_de_ph,coef=None,conv=False,\
                    batch_norm=False,ac_fn=tf.nn.relu,reg=None,batch_train=False,\
                    cl_constr=True,div_type='KL',lambda_reg=1.,lambda_constr=1.,\
                    bayes=False,local_constr=0.,*args,**kargs):

        self.estimator = self.build_estimator(net_shape,nu_ph,de_ph,coef,conv,batch_norm,ac_fn,\
                                    reg,batch_train,div_type,lambda_reg,bayes,local_constr,*args,**kargs)
        self.div_type = div_type
        self.prev_nu_ph = prev_nu_ph if prev_nu_ph is not None else nu_ph
        self.prev_de_ph = prev_de_ph if prev_de_ph is not None else de_ph

        self.cl_constr = cl_constr
        self.lambda_constr = lambda_constr
        self.bayes = bayes
        
        if bayes:
            # initialize prior parameters
            self.prior_mean = 0.
            self.prior_std = 10.
      
        return


    @abstractmethod
    def build_estimator(self,net_shape,nu_ph,de_ph,coef=None,conv=False,batch_norm=False,\
                        ac_fn=tf.nn.relu,reg=None,batch_train=False,div_type='KL',\
                        lambda_reg=1.,*args,**kargs):
        pass

    
    @abstractmethod
    def save_prev_estimator(self,*args,**kargs):
        pass

    @abstractmethod
    def update_estimator(self,*args,**kargs):
        pass


class Continual_LogLinear_Estimator(Continual_Estimator):
    
    def build_estimator(self,net_shape,nu_ph,de_ph,coef=None,conv=False,batch_norm=False,\
                        ac_fn=tf.nn.relu,reg=None,batch_train=False,div_type='KL',lambda_reg=1.,\
                        bayes=False,local_constr=0.,*args,**kargs):
        
        if batch_norm:
            raise NotImplementedError('Batch norm not supported!')
        if div_type == 'KL':
            return KL_Loglinear_Estimator(net_shape,nu_ph,de_ph,coef,conv,batch_norm,ac_fn,reg,\
                                            batch_train=batch_train,lambda_reg=lambda_reg,bayes=bayes,constr=local_constr)
        
        elif div_type == 'Chi':
            return Chi_Loglinear_Estimator(net_shape,nu_ph,de_ph,coef,conv,batch_norm,ac_fn,reg,\
                                            batch_train=batch_train,lambda_reg=lambda_reg,bayes=bayes,constr=local_constr)

        else:
            raise NotImplementedError
    
    def save_prev_estimator(self,sess):

        self.prev_W = sess.run(self.estimator.W)
        self.prev_B = sess.run(self.estimator.B)
        print('check params',[np.sum(np.isnan(w)) for w in self.prev_W],[np.sum(np.isnan(b)) for b in self.prev_B])
        conv_L = len(self.estimator.net_shape[0]) if self.estimator.conv else 0
        print('conv_L',conv_L)
        prev_nu_H = GAN.restore_d_net(self.prev_nu_ph,self.prev_W,self.prev_B,conv_L,ac_fn=self.estimator.ac_fn,batch_norm=False)
        prev_de_H = GAN.restore_d_net(self.prev_de_ph,self.prev_W,self.prev_B,conv_L,ac_fn=self.estimator.ac_fn,batch_norm=False)

        return prev_nu_H[-1],prev_de_H[-1]


    def update_estimator(self,sess):
        
        self.prev_nu_r,self.prev_de_r = self.save_prev_estimator(sess)
        self.estimator.nu_r = self.estimator.nu_H[-1] - self.prev_nu_r
        self.estimator.de_r = self.estimator.de_H[-1] - self.prev_de_r
        self.update_train(self.estimator)

        return

    def get_cl_constr(self):
        estimator = self.estimator
        if self.div_type == 'KL':
            return tf.square(tf.div(tf.reduce_mean(tf.exp(estimator.de_r))*tf.reduce_mean(tf.exp(self.prev_nu_r)),\
                                tf.reduce_mean(tf.exp(estimator.de_H[-1]))) - 1.)
        elif self.div_type == 'Chi':
            return tf.square(tf.reduce_mean(tf.exp(estimator.de_r)) - 1.)
    
    def update_train(self,estimator,*args,**kargs):
        estimator.ll_loss = estimator.set_loss()
        if self.cl_constr:
            estimator.loss = estimator.ll_loss + self.lambda_constr * self.get_cl_constr()
                               
        else:
            estimator.loss = estimator.ll_loss

        if self.bayes:
            rt = self.current_log_ratio()
            estimator.loss -= tf.reduce_mean(normal_logpdf(rt,loc=self.prior_mean,scale=self.prior_std))

        var_list = get_vars_by_scope(estimator.scope)
        grads = tf.gradients(estimator.loss, var_list)
        estimator.grads_and_vars = list(zip(grads, var_list))

        estimator.train = estimator.opt[0].apply_gradients(estimator.grads_and_vars, global_step=estimator.opt[1])

        return


    def original_ratio(self,sess,x,x_de,*args,**kargs):
        
        r = self.estimator.ratio(sess,x,x_de,\
                                    nu_r=self.estimator.nu_H[-1],\
                                    de_r=self.estimator.de_H[-1])        
        
        return r


    def original_log_ratio(self,sess,x,x_de,*args,**kargs):
        
        r = self.estimator.log_ratio(sess,x,x_de,\
                                    nu_r=self.estimator.nu_H[-1],\
                                    de_r=self.estimator.de_H[-1])
        return r

    
    def learning(self,sess,nu_samples,de_samples,test_nu_samples=None,test_de_samples=None,\
                    batch_size=64,epoch=50,print_e=1,nu_dist=None,de_dist=None,early_stop=False,\
                    tol=0.,update_feed_dict=None,min_epoch=50,prev_nu_samples=None,prev_de_samples=None,\
                    t_prev_nu_samples=None,t_prev_de_samples=None,*args,**kargs):
        self.prev_map = {} 
        if prev_nu_samples is not None:                      
            self.prev_map[id(nu_samples)] = prev_nu_samples
            self.prev_map[id(de_samples)] = prev_de_samples
            if test_nu_samples is not None:
                self.prev_map[id(test_nu_samples)] = t_prev_nu_samples
                self.prev_map[id(test_de_samples)] = t_prev_de_samples

        return self.estimator.learning(sess,nu_samples,de_samples,test_nu_samples,test_de_samples,batch_size,epoch,print_e,\
                                    nu_dist,de_dist,early_stop,tol,self.update_feed_dict,min_epoch,*args,**kargs)


    
    def update_feed_dict(self,nu_samples,de_samples,ii,batch_size,is_training=True,*args,**kargs):
        feed_dict = {self.estimator.is_training:is_training} if self.estimator.batch_norm else {}
        ii_bk = ii

        nu_batch,_,ii = get_next_batch(nu_samples,batch_size,ii,repeat=True)
        de_batch,_,__ = get_next_batch(de_samples,batch_size,ii_bk,repeat=True)

        if len(self.prev_map)>0:
            prev_nu_batch,_,__ = get_next_batch(self.prev_map[id(nu_samples)],batch_size,ii_bk,repeat=True)
            prev_de_batch,_,__ = get_next_batch(self.prev_map[id(de_samples)],batch_size,ii_bk,repeat=True)
            feed_dict.update({self.estimator.nu_ph:nu_batch,self.estimator.de_ph:de_batch,\
                                self.prev_nu_ph:prev_nu_batch,self.prev_de_ph:prev_de_batch})
        else:

            feed_dict.update({self.estimator.nu_ph:nu_batch,self.estimator.de_ph:de_batch})

        return feed_dict,ii
    
    '''
    def current_log_ratio(self):
        return self.estimator.nu_r - tf.log(tf.reduce_mean(tf.exp(self.estimator.de_r)))


    def prev_ratio(self,sess,x,x_de,*args,**kargs):
        r = tf.div(tf.exp(self.prev_nu_r),tf.reduce_mean(tf.exp(self.prev_de_r)))

        feed_dict={self.estimator.nu_ph:x,self.estimator.de_ph:x_de}

        return sess.run(r,feed_dict)
    '''


class Continual_f_Estimator(Continual_LogLinear_Estimator):

    def build_estimator(self,net_shape,nu_ph,de_ph,coef=None,conv=False,batch_norm=False,\
                    ac_fn=tf.nn.relu,reg=None,batch_train=False,div_type='KL',lambda_reg=1.,\
                    bayes=False,local_constr=0.01,*args,**kargs):
        
        if batch_norm:
                raise NotImplementedError('Batch norm not supported!')
        
        return f_Estimator(net_shape,nu_ph,de_ph,coef,conv,batch_norm,ac_fn=ac_fn,reg=reg,\
                        lambda_reg=lambda_reg,bayes=bayes,constr=local_constr,divergence=div_type)

    
    def update_estimator(self,sess):
        self.prev_nu_r,self.prev_de_r = self.save_prev_estimator(sess)

        if self.div_type == 'KL':
            self.estimator.nu_r = self.estimator.nu_H[-1] - self.prev_nu_r + 1.
            self.estimator.de_r = self.estimator.de_H[-1] - self.prev_de_r + 1.
        elif self.div_type == 'Pearson': 
            nr = tf.clip_by_value((self.estimator.nu_H[-1]+2.)/(self.prev_nu_r+2.),-1e30,1e30)
            dr = tf.clip_by_value((self.estimator.de_H[-1]+2.)/(self.prev_de_r+2.),-1e30,1e30)
            self.estimator.nu_r = 2.*(nr - 1.)
            self.estimator.de_r = 2.*(dr - 1.)
            #self.estimator.nu_r = tf.clip_by_value(self.estimator.nu_r,-1e15,1e15)
            #self.estimator.de_r = tf.clip_by_value(self.estimator.de_r,-1e15,1e15)
        else:
            self.estimator.nu_r = self.estimator.nu_H[-1] - self.prev_nu_r 
            self.estimator.de_r = self.estimator.de_H[-1] - self.prev_de_r 
                   
        self.update_train(self.estimator)

        return

    
    def get_cl_constr(self):

        estimator = self.estimator

        if self.div_type == 'Pearson':
            constr = tf.square(tf.reduce_mean(tf.div(estimator.de_H[-1]+2., self.prev_de_r+2.)) - 1.)
        
        elif self.div_type == 'Hellinger':
            constr = tf.square(tf.reduce_mean(tf.exp(2.*estimator.de_r)) - 1.)
        
        else:
            constr = tf.square(tf.reduce_mean(tf.exp(estimator.de_r)) - 1.)
                
        return constr

