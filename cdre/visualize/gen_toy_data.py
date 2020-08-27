
import numpy as np 
import argparse

from utils.data_util import *
from utils.test_util import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
parser.add_argument('--ns_p_rg', default=[0.01,0.5], type=str2flist,help='noise portion range')
parser.add_argument('--p_step', default=0.01, type=float, help='increasing step of noise portion')
parser.add_argument('--ns_dim_rg', default=[1,50], type=str2ilist, help='noise dimension range')
parser.add_argument('--d_step', default=1, type=int, help='increasing step of noise dimension')
parser.add_argument('--spath', default='./', type=str, help='save path')

args = parser.parse_args()

from tensorflow.examples.tutorials.mnist import input_data

if args.dataset== 'mnist':
    data_dir = '/home/yu/gits/data/mnist/'  
elif args.dataset=='fashion':
    data_dir = '/home/yu/gits/data/fashion/'

data = input_data.read_data_sets(data_dir,one_hot=False) 
data = data.train.images

gen_noise_samples_by_range(data, args.ns_p_rg, args.ns_dim_rg, p_step=args.p_step, dim_step=args.d_step, save_path=args.spath)

