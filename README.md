# Continual Density Ratio Estimation

This repository contains the source code for paper: [Continual Density Ratio Esitmation in an Online Setting](https://arxiv.org/abs/2103.05276).

## Installation
 The source code is based on Python 3.7. The required packages are listed in the requirement.txt. The last 4 packages can be found here: [utils](https://github.com/yc14600/utils), [base_models](https://github.com/yc14600/base_models),[cl_models](https://github.com/yc14600/cl_models), [hsvi](https://github.com/yc14600/hsvi). 

 ## Usage

 #### 1.Single pair of original and dynamic distributions
   
 To test a single pair of original and dynamic distributions, please use *cl_ratio_test.py*. One usage example for the simulated case of 64-D Gaussian distributions in the paper is as below:

 `python cl_ratio_test.py --T 10 --result_path ./results/ --d_dim 64 --delta_mean 0.02 --delta_std 0.02 --sample_size 50000 --test_sample_size 10000 --epoch 500 --batch_size 2000 --festimator False --divergence KL --continual_ratio True --learning_rate 0.00001 --lambda_constr 10. --seed 0`


#### 2.Multiple pairs of original and dynamic distributions

To test multiple pairs of original and dynamic distributions, please use *cond_cl_ratio_test.py*. One usage example for the simulated case of 64-D Gaussian distributions in the paper is as below:

`python cond_cl_ratio_test.py --T 10 --result_path ./results/ --d_dim 64 --delta_par 0.01 --sample_size 50000 --test_sample_size 10000 --epoch 5000 --batch_size 2000 --festimator False --divergence KL --continual_ratio True --learning_rate 0.00001 --lambda_constr 100. --increase_constr True --multihead True --seed 0`


#### 3. Standard KLIEP

To test standard KLIEP, just use cl_ratio_test.py or cond_cl_ratio_test.py as demonstrated above and set '--continual_ratio False'.

#### 4. Experiments with stock data

The dataset consists of one-day transactions of the Microsoft stock which can be downloaded from [here](https://lobsterdata.com/info/DataSamples.php) and is sample data for free. We only used the one level data. To pre-process the stock data and train regression model on it, please use the notebook *experiments_stock.ipynb* in *stock* folder. 

To run CDRE on the stock data, please use *cl_ratio_test.py* as well. For example, to reproduce the results of our experiment with restart:

`python cl_ratio_test.py --T 9 --d_dim 19 --dataset stock --datapath ./stock/MSFT.npy --result_path ./results/stock/MSFT/  --sample_size 5000 --test_sample_size 1000 --epoch 1000 --batch_size 1000 --festimator False --divergence KL --continual_ratio True --learning_rate 0.00002 --lambda_constr 1. --restart True --restart_th .5 --seed 0` 

To test the case without restart, just set '--restart False'. 


#### 5. Testing FID, KID, and PRD

We also provide code to test FID, KID, and PRD in *score_test.py*. To test PRD, please install the [PRD package](https://github.com/msmsajjadi/precision-recall-distributions) first. The following is an example to test FID with features extracted by a classifier: 

`python score_test.py -dataset FASHION -result_path './results/' -dpath '../vis_results/fGAN/fashion/JS/' -test_model_type JS -extract_feature True -z_dim 64 -feature_type classifier -model_type single -conv True -sample_size 1000 -score_type fid  -seed 0`