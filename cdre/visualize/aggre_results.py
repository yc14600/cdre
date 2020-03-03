import numpy as np
import pandas as pd
import os 
import six
import seaborn as sn
import matplotlib.pyplot as plt

from utils.test_util import calc_divgenerce
from utils.data_util import save_samples, extract_data, load_inception_net

import os
path = os.getcwd()
import sys
sys.path.append(path+'/../../')
#import prd.prd_score as prd

import matplotlib as mtp
mtp.rcParams['pdf.fonttype'] = 42
mtp.rcParams['ps.fonttype'] = 42



def aggreate_score(path,stype='fid',file_id=[0]):
    """aggreate scores with multiple runs
    
    Arguments:
        path {str} -- the path to results of multiple runs
    
    Keyword Arguments:
        stype {str} -- can be one of ['fid','kid','kl','prd','prd_seg','prd_half'] (default: {'fid'})
        file_id {list} -- only used when ftype is prd_seg or prd_half to specify file sequences (default: {[0]})
    
    Returns:
        [type] -- [description]
    """
    sd = os.listdir(path)
    data = []
    for s in sd:
        print(s)
        if '.' in s:
            continue
        #print(path+s+'/'+stype+'.csv')
        if stype in ['prd_seg','prd_half']:
            for fi in file_id:
                #print('fi',fi)
                ds = np.loadtxt(path+s+'/'+stype+str(fi)+'.csv',delimiter=',')
                data.append(ds)
                #print(len(data))
        else:
            ds = np.loadtxt(path+s+'/'+stype+'.csv',delimiter=',')
        #print(ds.shape)
                    
        if stype == 'kid':
            if len(ds.shape) > 1:
                data.append(ds[:,0])
            else:
                data.append(ds[0])
        else:
            #print('load score',ds)
            data.append(ds)

    n = len(data)
    #print(n)
    if 'prd' not in stype:
        data = np.vstack(data)
        data = data.reshape(n,data.shape[-1])
        
    return data


def aggregate_paths(paths,stype='fid',file_id=[0]):
    """aggregate the same type of scores in different test cases with multiple runs
    
    Arguments:
        paths {str list} -- paths where score files located
    
    Keyword Arguments:
        stype {str} -- can be one of ['fid','kid','kl','prd','prd_seg','prd_half'] (default: {'fid'})
        file_id {list} -- only used when ftype is prd_seg or prd_half to specify file sequences (default: {[0]})
    
    Returns:
        list -- aggreated scores
    """
    data = []
    for p in paths:       
        d = aggreate_score(p,stype,file_id)
        data.append(d)
    return data


def read_scores(paths,div_type=None):
    """read results of FID, KID or f-divergence
    
    Arguments:
        paths {str list} -- paths to result files
    
    Keyword Arguments:
        div_type {type of f-divergence} -- can be one of ['FID','KID','KL','rv_KL','Jensen_Shannon','Helligner'] (default: {None})
    
    Returns:
        scores {list}
    """
    scores=[]
    for p in paths:
        if div_type is None:
            s = np.loadtxt(os.path.join(p,'score.csv'),delimiter=',')
        else:
            s = np.loadtxt(os.path.join(p,div_type+'.csv'),delimiter=',')
        
        scores.append(s)
    return scores

def gen_fdiv_scores(paths,divs=['KL','rv_KL','Jensen_Shannon','Hellinger']):
    """generate score files of f-divergences by estimated sample ratios in multiple runs.
    
    Arguments:
        paths {str list} -- paths to ratio files
    
    Keyword Arguments:
        divs {list} -- supported types of f-divergence (default: {['KL','rv_KL','Jensen_Shannon','Hellinger']})
    """
    for div_type in divs:
        for pt in paths:
            sd = os.listdir(pt)
            spth = []
            for s in sd:
                #print(s)
                s = os.path.join(pt,s)
                spth.append(s)
                r = np.loadtxt(s+'/sample_ratios.csv',delimiter=',')
                div = calc_divgenerce(div_type,[r])
                np.savetxt(X=div,fname=os.path.join(s,div_type+'.csv'))



def compare_static_scores(score_paths):
    """compare different scores in different test cases with multiple runs, 
    single task of static learning
    
    Arguments:
        score_paths {dict} -- dictionary containing score types as keys and paths to score files as values
    
    Returns:
        score means, score values {tuple of arrays}
    """

    compare_score_means = []
    compare_score_stds = []

    for stype, paths in six.iteritems(score_paths):

        data = aggregate_paths(paths,stype=stype)

        score_means, score_stds = [], []
        for d in data:
            score_mean = np.mean(d)
            score_std = np.std(d)
            score_means.append(score_mean)
            score_stds.append(score_std)

        compare_score_means.append(score_means)
        compare_score_stds.append(score_stds)

    compare_score_means = np.vstack(compare_score_means)
    compare_score_stds = np.vstack(compare_score_stds)

    return compare_score_means, compare_score_stds


def plot_static_scores(score_paths,xticks,legends,save_path,width = 0.3):
    """[summary]
    
    Arguments:
        score_paths {dict} -- dictionary containing score types as keys and paths to score files as values
        xticks {list} -- list of ticks of x-axis
        legends {list} -- list of legend labels
        save_path {str} -- path to save figure
    
    Keyword Arguments:
        width {float} -- bar width (default: {0.3})
    """

    compare_score_means, compare_score_stds = compare_static_scores(score_paths)
    
    x_ax = np.arange(len(compare_score_means))
    K = compare_score_means.shape[1]
    for k in range(K):
        plt.bar(x_ax+width*k,compare_score_means[:,k],yerr=compare_score_stds[:,k],width=width)

    plt.xticks(x_ax + (K-1)*width/2, xticks)
    plt.legend(legends)
    plt.savefig(save_path)


def plot_prd_scores(paths,save_path,legends,stype='prd'):

    if stype == 'prd_segs':
        num_segs = 4
    elif stype == 'prd_half':
        num_segs = 2
    elif stype == 'prd':
        num_segs = 1

    file_id = np.arange(num_segs)
    data = aggregate_paths(paths,stype=stype,file_id=file_id)

    if  'prd' == stype:
        prd_data = [(data[i][0][0],data[i][0][1]) for i in range(len(data))]
        prd.plot(prd_data, labels=legends, out_path=save_path)

    elif stype in ['prd_seg','prd_half']:
        colors = ['r','k','g']
        lstyle = ['-','-.','--',':']
        fig = plt.figure(figsize=(4.6, 4.6))
        for m in range(len(data)):
            prd_data = [(data[m][i][0],data[m][i][1]) for i in range(len(data[0]))]
            for k,(prec,rec) in enumerate(prd_data):
                plt.plot(rec,prec,c=colors[m],alpha=0.75,linestyle=lstyle[k])
                
        plt.legend(legends,fontsize=12)

        plt.xlabel('Recall',fontsize=15)
        plt.ylabel('Precision',fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path,bbox_inches='tight')



