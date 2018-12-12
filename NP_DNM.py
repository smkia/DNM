#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:01:27 2018

@author: seykia
"""
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
import nibabel as nib
from scipy.io import savemat
from util.utilities import read_phenomics_data, prepare_data, apply_dropout_test
from util.utilities import np_loss, extreme_value_prob, extreme_value_prob_fit

############################ Parsing inputs ################################### 
parser = argparse.ArgumentParser(description='Neural Processes (NP) for Deep Normative Modeling')
parser.add_argument('--batchnum', type=int, default=10, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--r_dim', type=int, default=25, metavar='N',
                    help='dimension of r, the hidden representation of the context points')
parser.add_argument('--z_dim', type=int, default=10, metavar='N',
                    help='dimension of z, the global latent variable')
parser.add_argument('--M', type=int, default=10, metavar='N',
                    help='number of fixed-effect estimations')
parser.add_argument('--run_num', type=int, default=10, metavar='N',
                    help='number of runs')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

####################### Loading training and test data ########################
main_dir = 'Path to Phenomics Data/ds000030_R1.0.5/'
save_path = 'Path to Save Results'
ex_img = nib.load(main_dir + '/derivatives/task/sub-10159/taskswitch.feat/example_func.nii.gz')  
phenotypes = ['barratt']
task = 'stopsignal' 
cope = 1

# "bis_2npimp", "bis_1atten", "bis_1cogcom", "bis_1sc", "bis_2attimp", "bis_1coginst", "bis_2motimp", "bis_1pers", "bis_1mot", "bis_factor1_ci", "bis_factor2_bi"
factor_idx = [2, 20, 4, 0, 40, 42, 11, 49, 46, 27, 50]

control_fmri_data, control_phenotype_data = read_phenomics_data(main_dir, diagnosis='CONTROL', task_name = task , cope_num = cope, phenotypes = phenotypes)
SCHZ_fmri_data, SCHZ_phenotype_data = read_phenomics_data(main_dir, diagnosis = 'SCHZ', task_name = task , cope_num = cope, phenotypes = phenotypes)
ADHD_fmri_data, ADHD_phenotype_data = read_phenomics_data(main_dir, diagnosis = 'ADHD', task_name = task , cope_num = cope, phenotypes = phenotypes)
BIPL_fmri_data, BIPL_phenotype_data = read_phenomics_data(main_dir, diagnosis = 'BIPOLAR', task_name = task , cope_num = cope, phenotypes = phenotypes)
original_image_size = list(control_fmri_data.shape)

phen_idx = factor_idx
control_phenotype_data = control_phenotype_data[:,phen_idx]
SCHZ_phenotype_data = SCHZ_phenotype_data[:,phen_idx]
ADHD_phenotype_data = ADHD_phenotype_data[:,phen_idx]
BIPL_phenotype_data = BIPL_phenotype_data[:,phen_idx]

control_phenotype_data[control_phenotype_data<-100] = np.nan
SCHZ_phenotype_data[SCHZ_phenotype_data<-100] = np.nan
ADHD_phenotype_data[ADHD_phenotype_data<-100] = np.nan
BIPL_phenotype_data[BIPL_phenotype_data<-100] = np.nan

X_imputer = Imputer()
control_phenotype_data = X_imputer.fit_transform(control_phenotype_data)
SCHZ_phenotype_data = X_imputer.transform(SCHZ_phenotype_data)
ADHD_phenotype_data = X_imputer.transform(ADHD_phenotype_data)
BIPL_phenotype_data = X_imputer.transform(BIPL_phenotype_data)
control_phenotype_data =torch.tensor(control_phenotype_data)
SCHZ_phenotype_data =torch.tensor(SCHZ_phenotype_data)
ADHD_phenotype_data =torch.tensor(ADHD_phenotype_data)
BIPL_phenotype_data =torch.tensor(BIPL_phenotype_data)

# Cropping
x_from = 8; x_to = 57; y_from = 8; y_to = 69; z_from = 1; z_to = 41;
control_fmri_data = torch.tensor(control_fmri_data[:,x_from:x_to,y_from:y_to,z_from:z_to])
SCHZ_fmri_data = torch.tensor(SCHZ_fmri_data[:,x_from:x_to,y_from:y_to,z_from:z_to])
ADHD_fmri_data = torch.tensor(ADHD_fmri_data[:,x_from:x_to,y_from:y_to,z_from:z_to])
BIPL_fmri_data = torch.tensor(BIPL_fmri_data[:,x_from:x_to,y_from:y_to,z_from:z_to])
Y_shape = control_fmri_data.shape[1:]
voxel_num = np.prod(control_fmri_data.shape[1:])
    
train_num = 75
factor = args.M
sampling = 'bootstrap'

##################################### Model ###################################
args.r_conv_dim = 100
lrlu_neg_slope = 0.01
dp_level = 0.1

class NP(nn.Module):
    def __init__(self, args):
        super(NP, self).__init__()
        self.r_conv_dim = args.r_conv_dim
        self.r_dim = args.r_dim
        self.z_dim = args.z_dim
        self.factor=factor
        self.x_dim = x_context.shape[2]
        self.context_sample_num = x_context.shape[0]

        self.h_fmri_1 = nn.Conv3d(in_channels=factor, out_channels=25, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True) # in:(1,49,61,40) out:(5,47,59,38)
        self.h_fmri_1_bn = nn.BatchNorm3d(25)
        self.h_fmri_2 = nn.AvgPool3d(kernel_size=3, stride=2, padding=0, ceil_mode=False) 
        self.h_fmri_3 = nn.Conv3d(in_channels=25, out_channels=15, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True) # out: (7,21,27,16)
        self.h_fmri_3_bn = nn.BatchNorm3d(15)
        self.h_fmri_4 = nn.AvgPool3d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
        self.h_fmri_5 = nn.Conv3d(in_channels=15, out_channels=10, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True) # out: (10,8,11,5)
        self.h_fmri_5_bn = nn.BatchNorm3d(10)
        self.h_fmri_6 = nn.AvgPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.h_fmri_7_dp = nn.Dropout(p=dp_level)
        self.h_fmri_7 = nn.Linear(400, self.r_conv_dim)
        
        self.h_1_dp = nn.Dropout(p=dp_level)
        self.h_1 = nn.Linear(self.r_conv_dim + self.x_dim , 100)
        self.h_2_dp = nn.Dropout(p=dp_level)
        self.h_2 = nn.Linear(100, self.r_dim)

        self.r_to_z_mean_dp = nn.Dropout(p=dp_level)
        self.r_to_z_mean = nn.Linear(self.r_dim, self.z_dim)
        self.r_to_z_logvar_dp = nn.Dropout(p=dp_level)
        self.r_to_z_logvar = nn.Linear(self.r_dim, self.z_dim)

        self.g_1_dp = nn.Dropout(p=dp_level)
        self.g_1 = nn.Linear(self.z_dim + self.x_dim, 100)
        self.g_2_dp = nn.Dropout(p=dp_level)
        self.g_2 = nn.Linear(100, 400)
        self.g_3 = nn.Upsample(scale_factor=2)
        self.g_4 = nn.ConvTranspose3d(in_channels=10, out_channels=15, kernel_size=3, stride=1, padding=0, output_padding=(0,0,0), groups=1, bias=True, dilation=1) # out: (10,12,6) 
        self.g_4_bn = nn.BatchNorm3d(15)
        self.g_5 = nn.Upsample(scale_factor=2)
        self.g_6 = nn.ConvTranspose3d(in_channels=15, out_channels=25, kernel_size=3, stride=1, padding=0, output_padding=(0,0,0), groups=1, bias=True, dilation=1) # out: (22,26,14)
        self.g_6_bn = nn.BatchNorm3d(25)
        self.g_7 = nn.Upsample(scale_factor=(2.14,2.27,2.72)) 
        self.g_8 = nn.ConvTranspose3d(in_channels=25, out_channels=1, kernel_size=3, stride=1, padding=(0,0,0), output_padding= (0,0,0), groups=1, bias=True, dilation=1) # out: (49,61,40) 

    def h(self, x, fmri):
        fmri = F.leaky_relu(self.h_fmri_1_bn(self.h_fmri_1(fmri)),lrlu_neg_slope)
        fmri = self.h_fmri_2(fmri)
        fmri = F.leaky_relu(self.h_fmri_3_bn(self.h_fmri_3(fmri)),lrlu_neg_slope)
        fmri = self.h_fmri_4(fmri)
        fmri = F.leaky_relu(self.h_fmri_5_bn(self.h_fmri_5(fmri)),lrlu_neg_slope)
        fmri= self.h_fmri_6(fmri)
        fmri = F.leaky_relu(self.h_fmri_7(self.h_fmri_7_dp(fmri.view(fmri.shape[0],400))),lrlu_neg_slope)
        x_y = torch.cat((fmri,torch.mean(x,dim=1)),1)
        x_y = F.leaky_relu(self.h_1(self.h_1_dp(x_y)),lrlu_neg_slope)
        x_y = F.leaky_relu(self.h_2(self.h_2_dp(x_y)),lrlu_neg_slope)
        return x_y
    
    def xy_to_z_params(self, x, fmri):
        r = self.h(x, fmri)
        mu = self.r_to_z_mean(self.r_to_z_mean_dp(r))
        logvar = self.r_to_z_logvar(self.r_to_z_logvar_dp(r))
        return mu, logvar
      
    def reparameterise(self, z):
        mu, logvar = z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sample = eps.mul(std).add_(mu)
        return z_sample

    def g(self, z_sample, x_target):
        z_x = torch.cat([z_sample, torch.mean(x_target,dim=1)], dim=1)
        z_x = F.leaky_relu(self.g_1(self.g_1_dp(z_x)),lrlu_neg_slope)
        z_x = F.leaky_relu(self.g_2(self.g_2_dp(z_x)),lrlu_neg_slope)
        z_x = z_x.view(x_target.shape[0],10,4,5,2)
        z_x = self.g_3(z_x)
        z_x = F.leaky_relu(self.g_4_bn(self.g_4(z_x)),lrlu_neg_slope)
        z_x = self.g_5(z_x)
        z_x = F.leaky_relu(self.g_6_bn(self.g_6(z_x)),lrlu_neg_slope)
        z_x = self.g_7(z_x)
        y_hat = F.sigmoid(self.g_8(z_x))
        return y_hat

    def forward(self, x_context, y_context, x_all=None, y_all=None, n = None):
        y_sigma = None
        z_context = self.xy_to_z_params(x_context, y_context)
        if self.training:
            z_all = self.xy_to_z_params(x_all, y_all)
            z_sample = self.reparameterise(z_all)
            y_hat = self.g(z_sample, x_all)
        else:  
            z_all = z_context
            temp = torch.zeros([n,y_context.shape[0],1,y_context.shape[2],y_context.shape[3],y_context.shape[4]])
            for i in range(n):
                z_sample = self.reparameterise(z_all)
                temp[i,:] = self.g(z_sample, x_context)
            y_hat = torch.mean(temp, dim=0)
            y_sigma = torch.std(temp, dim=0)
        return y_hat, z_all, z_context, y_sigma
    
########################### Training and Evaluation ###########################
        
epochs = [int(args.epochs/4),int(args.epochs/2),int(args.epochs/5),int(args.epochs-args.epochs/4-args.epochs/2-args.epochs/5)]
EVD_params = []
abnormal_probs = []
auc_all = np.zeros([args.run_num,])
auc_SCHZ = np.zeros([args.run_num,])
auc_ADHD = np.zeros([args.run_num,])
auc_BIPL = np.zeros([args.run_num,])
for r in range(args.run_num):
    x_context, y_context, x_all, y_all, x_context_test, y_context_test, x_test, y_test, labels, diagnosis_labels, scaler = \
            prepare_data(control_fmri_data, control_phenotype_data, SCHZ_fmri_data, SCHZ_phenotype_data,
                         ADHD_fmri_data, ADHD_phenotype_data, BIPL_fmri_data, BIPL_phenotype_data, train_num, factor, sampling=sampling)
    x_test = x_test.view((x_test.shape[0],1,x_test.shape[1]))
    model = NP(args).to(device)
    k = 1
    mini_batch_num = args.batchnum
    batch_size = int(x_context.shape[0]/mini_batch_num)
    model.train()
    for e in range(len(epochs)): 
        optimizer = optim.Adam(model.parameters(), lr=10**(-e-2))
        for j in range(epochs[e]):
            train_loss = 0
            rand_idx = np.random.permutation(x_context.shape[0])
            for i in range(mini_batch_num):
                optimizer.zero_grad()
                idx = rand_idx[i*batch_size:(i+1)*batch_size]
                y_hat, z_all, z_context, dummy = model(x_context[idx,:,:], y_context[idx,:,:,:,:], x_all[idx,:,:], y_all[idx,:,:,:,:])
                loss = np_loss(y_hat, y_all[idx,:,:,:,:], z_all, z_context)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print('Run: %d, Epoch: %d, Loss:%f' %(r+1, k, train_loss))
            k += 1
        
    model.eval()
    model.apply(apply_dropout_test)
    with torch.no_grad():
        y_hat, z_all, z_context, y_sigma = model(x_context_test, y_context_test, n = 100)
        test_loss = np_loss(y_hat, y_test, z_all, z_context).item()
         
    NPM = (y_test - y_hat) / (y_sigma)
    mask = np.zeros_like(control_fmri_data[0,:,:,:])
    mask[control_fmri_data[0,:,:,:].numpy()!=0]=1
    mask = torch.tensor(mask)
    NPM = NPM.mul(mask)
    NPM = np.nan_to_num(NPM.numpy())
    NPM = NPM.squeeze()
    
    temp=NPM.reshape([NPM.shape[0],NPM.shape[1]*NPM.shape[2]*NPM.shape[3]])
    EVD_params.append(extreme_value_prob_fit(temp, 0.01))
    abnormal_probs.append(extreme_value_prob(EVD_params[r], temp, 0.01))
    auc_all[r] = roc_auc_score(labels, abnormal_probs[r])
    auc_SCHZ[r] = roc_auc_score(labels[(diagnosis_labels==0) | (diagnosis_labels==1),], abnormal_probs[r][(diagnosis_labels==0) | (diagnosis_labels==1),])
    auc_ADHD[r] = roc_auc_score(labels[(diagnosis_labels==0) | (diagnosis_labels==2),], abnormal_probs[r][(diagnosis_labels==0) | (diagnosis_labels==2),])
    auc_BIPL[r] = roc_auc_score(labels[(diagnosis_labels==0) | (diagnosis_labels==3),], abnormal_probs[r][(diagnosis_labels==0) | (diagnosis_labels==3),])    
    
    original_image_size[0] = NPM.shape[0]
    temp = np.zeros(original_image_size, dtype=np.float32)
    temp[:,x_from:x_to,y_from:y_to,z_from:z_to] = NPM
    image = nib.Nifti1Image(np.transpose(temp, [1,2,3,0]), ex_img.affine, ex_img.header)
    nib.save(image, save_path + 'NPMs_M' + str(factor) + '_run' + str(r) +'.nii.gz')
    torch.save(model.state_dict(), save_path + 'model_M' + str(factor) + '_run' + str(r) + '.pt')
    savemat(save_path + 'Results_M' + str(factor) + '.mat',{'EVD_params': EVD_params, 'abnormal_probs': abnormal_probs,
                                                              'auc_all': auc_all, 'auc_SCHZ': auc_SCHZ,
                                                              'auc_ADHD':auc_ADHD, 'auc_BIPL':auc_BIPL,
                                                              'diagnosis_labels':diagnosis_labels,'x_test':x_test.squeeze().numpy()})