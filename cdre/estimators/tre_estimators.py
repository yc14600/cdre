from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf
from utils.train_util import get_next_batch,concat_cond_data
from base_models.gans import GAN,fGAN
from base_models.ratio_fgan import Ratio_fGAN
from .estimators import LogLinear_Estimator,f_Estimator
from .cond_estimators import Cond_Loglinear_Estimator

class TRE_Loglinear_Estimator(Cond_Loglinear_Estimator):

    def __init__(self,net_shape,nu_ph,de_ph,c_ph,coef=None,conv=False,batch_norm=False,ac_fn=tf.nn.relu,\
                reg=None,scope='ratio',batch_train=False, lambda_reg=1.,bayes=False,constr=0.,\
                c_dim=10, *args,**kargs):
                
        super(TRE_Loglinear_Estimator,self).__init__(net_shape,nu_ph,de_ph,c_ph,coef,conv,batch_norm,ac_fn,reg,scope,\
                batch_train,lambda_reg,bayes,constr)
        self.max_c = c_dim
        
        return

    def log_ratio(self,sess,x_nu,x_de,nu_r=None,de_r=None,coef=None,*args,**kargs):

        nu_r = self.nu_r if nu_r is None else nu_r
        de_r = self.de_r if de_r is None else de_r
        coef = self.coef if coef is None else coef

        de_mean = tf.reduce_mean(de_r,axis=0)
        log_r = nu_r - tf.log(de_mean)         
        
        if self.nu_ph.shape[0].value is None:
            feed_dict = {self.is_training:False} if self.batch_norm else {}
            feed_dict.update({self.nu_ph:x_nu,self.de_ph:x_de})
            rt = sess.run(log_r,feed_dict)

        else:
            batch_size = self.nu_ph.shape[0].value
            ii = 0
            iters = int(np.ceil(x_nu.shape[0]/batch_size))
            rt = np.zeros([x_nu.shape[0],self.c_ph.shape[-1].value])

            for _ in range(iters):
                start = ii
                feed_dict,ii = self.update_feed_dict(x_nu,x_de,ii,batch_size)
                end = ii if ii < x_nu.shape[0] and ii > start else x_nu.shape[0]

                rt[start:end] = sess.run(log_r,feed_dict)[:end-start]
        
        rt[np.isnan(rt)] = 0.
        return rt

    def original_log_ratio(self,sess,x,x_de,t,nu_r=None,de_r=None,coef=None,*args,**kargs):
            
        log_r = self.log_ratio(sess,x,x_de,nu_r,de_r)
        sum_log_r = np.sum(log_r[:,:t+1],axis=1)                                                       
        return sum_log_r

    def original_ratio(self,sess,x,x_de,t,nu_r=None,de_r=None,coef=None,*args,**kargs):
        logr = self.original_log_ratio(sess,x,x_de,t,nu_r,de_r,coef,*args,**kargs)
        return np.exp(logr)


    def learning(self,sess,nu_samples,de_samples,samples_c,test_nu_samples=None,test_de_samples=None,\
                test_samples_c=None,batch_size=64,epoch=50,print_e=1,nu_dist=None,de_dist=None,early_stop=False,\
                tol=0.,update_feed_dict=None,min_epoch=50,*args,**kargs):

        return super(TRE_Loglinear_Estimator,self).learning(sess,nu_samples,de_samples,samples_c,test_nu_samples,test_de_samples,\
                test_samples_c,batch_size,epoch,print_e,nu_dist,de_dist,early_stop,\
                tol,update_feed_dict,min_epoch,concat_c=False,*args,**kargs)


class TRE_KL_Loglinear_Estimator(TRE_Loglinear_Estimator):
    
    def set_loss(self):
        nu_r = self.nu_r * self.c_ph
        de_r = self.de_r * self.c_ph

        c_num = tf.reduce_sum(self.c_ph,axis=0)
        c_prop = c_num /tf.reduce_sum(c_num)

        nu_mean = self.cond_mean(nu_r) 
        de_mean = self.exp_de_mean(de_r)
        lloss = -nu_mean + tf.log(de_mean)
        
        self.ll_loss = tf.reduce_sum(lloss*c_prop) # for early stop
        loss = lloss * c_prop
       
        # average over all conditions
        loss = tf.reduce_sum(loss)

        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())

        return loss + self.lambda_reg * reg_loss 


class TRE_f_Estimator(TRE_Loglinear_Estimator,f_Estimator):

    def __init__(self,net_shape,nu_ph,de_ph,c_ph,coef=None,conv=False,batch_norm=False,ac_fn=tf.nn.relu,\
                reg=None,scope='ratio',batch_train=False, lambda_reg=1.,bayes=False,constr=0.01,\
                divergence='KL', c_dim=10, *args,**kargs):
        self.c_ph = c_ph
        self.c_dim = c_dim
        self.max_c = c_dim
        f_Estimator.__init__(self,net_shape,nu_ph,de_ph,coef,conv,batch_norm,ac_fn,reg,scope,\
                batch_train,lambda_reg,bayes,constr,divergence)
        
        return


    def log_ratio(self,sess,x_nu,x_de,nu_r=None,de_r=None,coef=None,*args,**kargs):
        
        nu_r = self.nu_r if nu_r is None else nu_r 

        idf_gf = Ratio_fGAN.get_idf_gf(self.divergence)
        log_r = tf.log(idf_gf(nu_r))


        if self.nu_ph.shape[0].value is None:
            feed_dict = {self.is_training:False} if self.batch_norm else {}
            feed_dict.update({self.nu_ph:x_nu})
            rlt = sess.run(log_r,feed_dict)
        else:
            batch_size = self.nu_ph.shape[0].value
            ii = 0
            iters = int(np.ceil(x_nu.shape[0]/batch_size))
            rlt = np.zeros([x_nu.shape[0],self.c_ph.shape[-1].value])
            for _ in range(iters):
                start = ii
                feed_dict,ii = self.update_feed_dict(x_nu,x_de,ii,batch_size)
                end = ii if ii < x_nu.shape[0] and ii > start else x_nu.shape[0]
                rlt[start:end] = sess.run(log_r,feed_dict)
        return rlt


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

