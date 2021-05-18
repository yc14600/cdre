from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import scipy as sp
import pandas as pd
import six
import tensorflow as tf
from abc import ABC, abstractmethod
from utils.train_util import config_optimizer, get_next_batch, get_vars_by_scope, shuffle_data
from base_models.gans import GAN,fGAN
from base_models.ratio_fgan import Ratio_fGAN

class Estimator(ABC):
    def __init__(self,net_shape,nu_ph,de_ph,coef=None,conv=False,batch_norm=False,ac_fn=tf.nn.relu,reg=None,\
                scope='ratio',batch_train=False,lambda_reg=1.,bayes=False,constr=0.,*args,**kargs):
        self.net_shape = net_shape
        self.nu_ph = nu_ph
        self.de_ph = de_ph
        self.coef = coef
        self.batch_norm = batch_norm
        self.conv = conv
        self.ac_fn = ac_fn
        self.reg = reg
        self.scope = scope
        self.batch_train = batch_train
        self.lambda_reg = lambda_reg
        self.bayes = bayes
        self.constr = constr 

        self.is_training = tf.placeholder(dtype=tf.bool,shape=[],name='estimator_batchnorm') if batch_norm else None

        self.define_estimator_vars(scope,net_shape,conv,ac_fn,batch_norm,reg)
        self.loss = self.set_loss()
        
        return


    def define_estimator_vars(self,scope,net_shape,conv,ac_fn,batch_norm,reg):
        
        with tf.variable_scope(scope):
            self.W, self.B, self.nu_H = self.define_base_fc(self.nu_ph,net_shape,conv=conv,ac_fn=ac_fn,batch_norm=batch_norm,training=self.is_training,reuse=False,reg=reg)
            _, __, self.de_H = self.define_base_fc(self.de_ph,net_shape,conv=conv,ac_fn=ac_fn,batch_norm=batch_norm,training=self.is_training,reuse=True,reg=reg)
            self.nu_H[-1] = tf.clip_by_value(self.nu_H[-1],-60.,60.)
            self.de_H[-1] = tf.clip_by_value(self.de_H[-1],-60.,60.)
            self.nu_r, self.de_r = self.nu_H[-1],self.de_H[-1]

        return


    def define_base_fc(self,x,net_shape,conv=False,ac_fn=tf.nn.relu,batch_norm=False,training=None,reuse=False,reg=None):
        return GAN.define_d_net(x,net_shape,conv=conv,ac_fn=ac_fn,batch_norm=batch_norm,training=training,reuse=reuse,reg=reg)   


    @abstractmethod
    def ratio(self,x,*args,**kargs):
        pass

    @abstractmethod
    def set_loss(self,*args,**kargs):
        pass

    @abstractmethod
    def learning(self,*args,**kargs):
        pass


class LogLinear_Estimator(Estimator):
    
    def log_ratio(self,sess,x_nu,x_de,nu_r=None,de_r=None,coef=None,correction=1.,*args,**kargs):
        
        nu_r = self.nu_r if nu_r is None else nu_r
        de_r = self.de_r if de_r is None else de_r
        coef = self.coef if coef is None else coef

        if coef is None:
            log_r = nu_r - tf.log(tf.reduce_mean(tf.exp(de_r))) - tf.log(correction)
   
        else:
            log_r = tf.matmul(nu_r,coef) - tf.log(tf.reduce_mean(tf.exp(tf.matmul(de_r,coef))))

        feed_dict={self.nu_ph:x_nu,self.de_ph:x_de}
        if self.batch_norm:
            feed_dict.update({self.is_training:False})

        return sess.run(log_r,feed_dict)
        

    def ratio(self,sess,x_nu,x_de,nu_r=None,de_r=None,coef=None,correction=1.,*args,**kargs):
        log_r = self.log_ratio(sess,x_nu,x_de,nu_r,de_r,coef,correction=correction)
        return np.exp(log_r)
    

    def config_train(self,learning_rate=0.001,decay=None,opt=None,clip=None,*args,**kargs):
        
        var_list = get_vars_by_scope(self.scope)
        
        grads = tf.gradients(self.loss, var_list)
        if clip is not None:
            for g in grads:
                g = tf.clip_by_value(g,clip[0],clip[1])
        self.grads = grads    
        self.grads_and_vars = list(zip(grads, var_list))

        if opt is None:
            opt = config_optimizer(learning_rate,'estimator_step','adam',decay=decay,scope=self.scope)
        
        self.opt = opt
        self.train = opt[0].apply_gradients(self.grads_and_vars, global_step=opt[1])

        return

    def update_feed_dict(self,nu_samples,de_samples,ii,batch_size,is_training=True,*args,**kargs):
        feed_dict = {self.is_training:is_training} if self.batch_norm else {}
        ii_bk = ii
        if isinstance(nu_samples,np.ndarray):
            nu_batch,_,ii = get_next_batch(nu_samples,batch_size,ii,repeat=True)
            de_batch,_,__ = get_next_batch(de_samples,batch_size,ii_bk,repeat=True)
        else:
            nu_batch = nu_samples.draw_samples(batch_size) # cope with data generator
            de_batch = de_samples.draw_samples(batch_size)

        feed_dict.update({self.nu_ph:nu_batch,self.de_ph:de_batch})

        return feed_dict,ii


    def learning(self,sess,nu_samples,de_samples,test_nu_samples=None,test_de_samples=None,\
                    batch_size=64,epoch=50,print_e=1,nu_dist=None,de_dist=None,early_stop=False,\
                    tol=0.,update_feed_dict=None,min_epoch=100,*args,**kargs):
        min_loss = -20.
        if update_feed_dict is None:
            update_feed_dict = self.update_feed_dict
        
        if isinstance(nu_samples,np.ndarray):
            N = max(nu_samples.shape[0],de_samples.shape[0])  
            num_iters = int(np.ceil(N/batch_size))
        else: # unlimited number of samples
            num_iters = 25 # just for convienient

        current_nu_mean,current_de_mean = 0., 0.
        loss, tloss, avg_err = 0., 0., 0.
        losses,tlosses,true_errs = [],[],[]
        farg = args[0] if len(args)>0 else None
        tfarg = args[1] if len(args)>1 else None
        for e in range(epoch):
            avg_tloss, avg_loss = 0., 0.
            idx = np.random.choice(len(nu_samples),len(nu_samples),replace=False)
            nu_samples = nu_samples[idx]
            #print(type(nu_samples[0].shape))
            idx = np.random.choice(len(de_samples),len(de_samples),replace=False)
            de_samples = de_samples[idx]
            ii = 0
            for i in range(num_iters):

                feed_dict,ii = update_feed_dict(nu_samples,de_samples,ii,batch_size,farg,**kargs)

                if self.batch_train:
                    #feed_dict.update({self.batch_num : (i+1.)})
                    feed_dict.update({self.prev_nu_mean : current_nu_mean}) 
                    feed_dict.update({self.prev_de_mean : current_de_mean})
            
                    _,loss,current_nu_mean,current_de_mean = sess.run([self.train,self.loss,self.current_nu_mean,self.current_de_mean],feed_dict=feed_dict)
                
                else:
                    _,loss,lloss = sess.run([self.train,self.loss,self.ll_loss],feed_dict=feed_dict)
    
                avg_loss+=loss
                if np.isnan(loss):
                    raise TypeError(e,i,'loss is nan')
                    
            if test_nu_samples is not None and test_de_samples is not None:

                ni = int(np.ceil(test_de_samples.shape[0]/batch_size))
                tloss,tlloss = 0.,0.
                for _ in range(ni):
                    t_feed_dict,_ = update_feed_dict(test_nu_samples,test_de_samples,_,batch_size,tfarg,**kargs)
                    tl = sess.run(self.loss,feed_dict=t_feed_dict)
                    tloss+=tl
                tloss/=ni   
                if np.isnan(tloss):
                    print('tloss is nan',e,i)


            losses.append(loss)
            tlosses.append(tloss)        

            if e%print_e==0:

                if test_nu_samples is not None and nu_dist is not None and de_dist is not None:
                    estimated_ratio = self.log_ratio(sess,test_de_samples,test_de_samples).reshape(-1)
                    nu_lp = nu_dist.log_prob(test_de_samples)
                    de_lp = de_dist.log_prob(test_de_samples)
                    true_ratio = nu_lp - de_lp

                    avg_err = np.mean(np.abs(true_ratio-estimated_ratio))                
                    true_errs.append(avg_err)

                print('epoch',e,'loss',loss,'valid loss',tloss,'avg log err',avg_err)
        

            if early_stop:
                if lloss < min_loss:
                    print('early stop satisfied',losses[-1])
                    break

                if test_nu_samples is not None:
                    if (e+1 > min_epoch and tlosses[-2] - tlosses[-1] < tol and tlosses[-3] - tlosses[-2] < tol) or tlloss < min_loss:
                        print('early stop satisfied by vloss',tlosses[-1])
                        break

                elif (e+1 >  min_epoch and losses[-2] - losses[-1] < tol and losses[-3] - losses[-2] < tol):
                    print('early stop satisfied by loss',losses[-1])
                    break                

        if self.bayes:
            rts = self.log_ratio(sess,nu_samples,de_samples)
            self.map_mean = np.mean(rts)
            self.map_std = np.std(rts)

        return losses,tlosses,true_errs

class KL_Loglinear_Estimator(LogLinear_Estimator):
    
    def set_loss(self,*args,**kargs):

        nu_mean = tf.reduce_mean(self.nu_r)
        if self.coef is None:
            loss = -nu_mean + tf.log(tf.reduce_mean(tf.exp(self.de_r))) #\
                    #+ tf.square(tf.reduce_mean(tf.reduce_mean(tf.exp(self.de_r)) * tf.exp(-1.* self.nu_r)) - 1.)
        else:
            loss = -tf.matmul(nu_mean,self.coef) + tf.log(tf.reduce_mean(tf.exp(tf.matmul(self.de_r,self.coef))))
    
        self.ll_loss = loss
        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())

        return loss + self.lambda_reg * reg_loss 


class Chi_Loglinear_Estimator(LogLinear_Estimator):
    
    def set_loss(self, *args, **kargs):
        d_r = tf.exp(self.de_r)#/tf.reduce_mean(tf.exp(self.de_r))
        n_r = tf.exp(self.nu_r)#/tf.reduce_mean(tf.exp(self.de_r))
        de_mean = 0.5 * tf.reduce_mean(tf.square(d_r))
        nu_mean = tf.reduce_mean(n_r)
        
        loss = de_mean - nu_mean 

        
        loss += tf.square(tf.reduce_mean(tf.exp(self.de_r)) - 1.) * self.constr

        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())

        return loss + self.lambda_reg * reg_loss
    
    def log_ratio(self,sess,x,x_de=None,nu_r=None,de_r=None,coef=None,*args,**kargs):
            
        nu_r = self.nu_r if nu_r is None else nu_r
                
        log_r = nu_r
   
        feed_dict={self.nu_ph:x}
        if self.batch_norm:
            feed_dict.update({self.is_training:False})

        return sess.run(log_r,feed_dict)

    
    def ratio(self,sess,x,x_de=None,nu_r=None,de_r=None,coef=None,*args,**kargs):
        log_r = self.log_ratio(sess,x,nu_r=nu_r)
        return np.exp(log_r)


class f_Estimator(LogLinear_Estimator):
    def __init__(self,net_shape,nu_ph,de_ph,coef=None,conv=False,batch_norm=False,ac_fn=tf.nn.relu,\
                    reg=None,scope='ratio',batch_train=False, lambda_reg=1.,bayes=False,constr=0.01,\
                    divergence='KL',*args,**kargs):
        
        self.divergence = divergence
        super(f_Estimator,self).__init__(net_shape,nu_ph,de_ph,coef,conv,batch_norm,ac_fn,\
                                        reg,scope,batch_train, lambda_reg,bayes,constr)

        return 


    def get_constr(self):
    
        if self.divergence == 'Pearson':
            constr = tf.square(tf.reduce_mean(0.5*self.de_r+1.) - 1.)
        
        elif self.divergence == 'Hellinger':
            constr = tf.square(tf.reduce_mean(tf.exp(2.*self.de_r)) - 1.)
        
        else:
            constr = tf.square(tf.reduce_mean(tf.exp(self.de_r)) - 1.)
                
        return constr

    def set_loss(self):
        act_fn,conj_f = fGAN.get_act_conj_fn(self.divergence)
        loss = -tf.reduce_mean(act_fn(self.nu_r)) + tf.reduce_mean(conj_f(act_fn(self.de_r)))
        self.ll_loss = loss 
        loss += self.get_constr()*self.constr
        return loss


    def ratio(self,sess,x,x_de=None,nu_r=None,de_r=None,*args,**kargs):
        nu_r = self.nu_r if nu_r is None else nu_r
        de_r = self.de_r if de_r is None else de_r

        idf_gf = Ratio_fGAN.get_idf_gf(self.divergence)
        r = idf_gf(nu_r)

        feed_dict={self.nu_ph:x,self.de_ph:x_de}
        if self.batch_norm:
            feed_dict.update({self.is_training:False})
        
        return sess.run(r,feed_dict)


    def log_ratio(self,sess,x,x_de=None,nu_r=None,de_r=None,*args,**kargs):
        r = self.ratio(sess,x,x_de,nu_r,de_r)
        return np.log(r)



