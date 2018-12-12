# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.nn import functional as F
from util import fileio
from scipy.stats import genextreme

def extreme_value_prob_fit(NPM, perc):
    n = NPM.shape[0]
    t = NPM.shape[1]
    n_perc = int(round(t * perc))
    m = np.zeros(n)
    for i in range(n):
        temp =  np.abs(NPM[i, :])
        temp = np.sort(temp)
        temp = temp[t - n_perc:]
        temp = temp[0:int(np.floor(0.90*temp.shape[0]))]
        m[i] = np.mean(temp)
    params = genextreme.fit(m)
    return params
    
def extreme_value_prob(params, NPM, perc):
    n = NPM.shape[0]
    t = NPM.shape[1]
    n_perc = int(round(t * perc))
    m = np.zeros(n)
    for i in range(n):
        temp =  np.abs(NPM[i, :])
        temp = np.sort(temp)
        temp = temp[t - n_perc:]
        temp = temp[0:int(np.floor(0.90*temp.shape[0]))]
        m[i] = np.mean(temp)
    if params[0] <= 0:  # if the shape is right tailed for extreme values
        probs = genextreme.cdf(m,*params)
    elif params[0] > 0: # if the shape is left tailed for extreme values
        probs = 1 - genextreme.cdf(m,*params)
    return probs

def ravel_2D(a):
    s = a.shape
    return np.reshape(a,[s[0], np.prod(s[1:])]) 

def unravel_2D(a, s):
    return np.reshape(a,s)    

def read_phenomics_data(main_dir, diagnosis = 'CONTROL', task_name = 'stopsignal', cope_num = 1, phenotypes = ['stopsignal'], masked = False, scanner=None):
    subjects_info = pd.read_csv(main_dir + 'participants.tsv', delimiter='\t')
    if scanner==None:
        subj_list =  subjects_info.loc[lambda subjects_info: subjects_info.diagnosis==diagnosis,'participant_id']
    else:
        a = np.asarray(subjects_info.diagnosis==diagnosis) * np.asarray(subjects_info.ScannerSerialNumber==scanner)
        subj_list =  subjects_info.loc[lambda subjects_info: a,'participant_id']
    phenotype_info = list()
    for i in range(len(phenotypes)):
        temp = pd.read_csv(main_dir + 'phenotype/' + phenotypes[i] + '.tsv', delimiter='\t')
        if i == 0:      
            temp = temp.iloc[:,1:]
        else:
            temp = temp.iloc[:,2:]
        phenotype_info.append(temp)
    phenotype_info = pd.concat(phenotype_info,axis=1)
    fmri_data = list()
    phenotype_data = list()
    for i in range(len(subj_list)):
        address = main_dir + 'derivatives/task/' +  subj_list.iloc[i] + '/' + task_name + '.feat/stats/' + 'cope' + str(cope_num) + '.nii.gz'
        if os.path.isfile(address):
            volume = fileio.load_nifti(address, vol = True)
            if masked:
                volmask = fileio.create_mask(volume, mask = main_dir + '/derivatives/task_group/' + task_name + '/mask.nii.gz')
                fmri_data.append(fileio.vol2vec(volume, volmask))
            else:
                volmask = fileio.create_mask(volume, mask = main_dir + '/derivatives/task_group/' + task_name + '/mask.nii.gz')
                volume = volume * volmask
                fmri_data.append(volume)
            temp = phenotype_info.loc[lambda phenotype_info: phenotype_info.participant_id==subj_list.iloc[i],:]
            temp = temp.apply(pd.to_numeric, errors='coerce')
            phenotype_data.append(np.float32(np.asarray(temp.iloc[0,1:])))
    return np.asarray(fmri_data), np.asarray(phenotype_data)

def prepare_data(control_fmri_data, control_phenotype_data, SCHZ_fmri_data, SCHZ_phenotype_data, \
                 ADHD_fmri_data, ADHD_phenotype_data, BIPL_fmri_data, BIPL_phenotype_data, train_num, factor=5, sampling='bootstrap'):
    CTRL_num = control_phenotype_data.shape[0]
    SCHZ_num = SCHZ_phenotype_data.shape[0]
    ADHD_num = ADHD_phenotype_data.shape[0]
    BIPL_num = BIPL_phenotype_data.shape[0]
    x_context = torch.zeros([train_num+15, factor, control_phenotype_data.shape[1]])
    y_context = torch.zeros([train_num+15, factor, control_fmri_data.shape[1], control_fmri_data.shape[2], control_fmri_data.shape[3]])
    x_all = torch.zeros([train_num+15, factor, control_phenotype_data.shape[1]])
    y_all = torch.zeros([train_num+15, factor, control_fmri_data.shape[1], control_fmri_data.shape[2], control_fmri_data.shape[3]])
    
    rand_idx = np.random.permutation(CTRL_num)
    train_idx_ctrl = rand_idx[0:train_num]
    test_idx_ctrl = np.setdiff1d(np.array(range(CTRL_num)),train_idx_ctrl)
    rand_idx = np.random.permutation(SCHZ_num)
    train_idx_SCHZ = rand_idx[0:5]
    test_idx_SCHZ = np.setdiff1d(np.array(range(SCHZ_num)),train_idx_SCHZ)
    rand_idx = np.random.permutation(ADHD_num)
    train_idx_ADHD = rand_idx[0:5]
    test_idx_ADHD = np.setdiff1d(np.array(range(ADHD_num)),train_idx_ADHD)
    rand_idx = np.random.permutation(BIPL_num)
    train_idx_BIPL = rand_idx[0:5]
    test_idx_BIPL = np.setdiff1d(np.array(range(BIPL_num)),train_idx_BIPL)

    x_context_train = torch.cat((control_phenotype_data[train_idx_ctrl,:],
                                      SCHZ_phenotype_data[train_idx_SCHZ,:], ADHD_phenotype_data[train_idx_ADHD,:], BIPL_phenotype_data[train_idx_BIPL,:]))
    means = x_context_train.mean(dim = 0, keepdim = True)
    stds = x_context_train.std(dim = 0, keepdim = True)
    x_context_train = (x_context_train - means) / stds
    x_context_train[x_context_train != x_context_train] = 0
    x_context_train[x_context_train == float("-Inf")] = 0
    x_context_train[x_context_train == float("Inf")] = 0
    
    x_context_test = torch.cat((control_phenotype_data[test_idx_ctrl,:], 
                                SCHZ_phenotype_data[test_idx_SCHZ,:], ADHD_phenotype_data[test_idx_ADHD,:], BIPL_phenotype_data[test_idx_BIPL,:]),0)
    x_context_test = (x_context_test - means) / stds
    x_context_test[x_context_test != x_context_test] = 0
    x_context_test[x_context_test == float("-Inf")] = 0
    x_context_test[x_context_test == float("Inf")] = 0
    
    x_test = x_context_test
    x_context_test = x_context_test.unsqueeze(1).expand(-1,factor,-1)
    
    y_context_train = torch.cat((control_fmri_data[train_idx_ctrl,:,:,:],
                                 SCHZ_fmri_data[train_idx_SCHZ,:,:,:], ADHD_fmri_data[train_idx_ADHD,:,:,:], BIPL_fmri_data[train_idx_BIPL,:,:,:]),0)
    y_test = torch.cat((control_fmri_data[test_idx_ctrl,:,:,:], SCHZ_fmri_data[test_idx_SCHZ,:,:,:], 
                        ADHD_fmri_data[test_idx_ADHD,:,:,:], BIPL_fmri_data[test_idx_BIPL,:,:,:]),0)
    y_context_test = torch.zeros([y_test.shape[0], factor, y_test.shape[1], y_test.shape[2], y_test.shape[3]])
    
    scaler = QuantileTransformer()
    scaler.fit(ravel_2D(np.concatenate((control_fmri_data, SCHZ_fmri_data, ADHD_fmri_data, BIPL_fmri_data),0)))
    
    for i in range(factor):
        if sampling == 'noise':
            x_context[:,i,:] = x_context_train + torch.randn(x_context_train.shape) * 0.01
            x_context_test[:,i,:] = x_context_test[:,i,:] + torch.randn([x_context_test.shape[0],x_context_test.shape[2]]) * 0.01
        elif sampling == 'bootstrap':
            x_context[:,i,:] = x_context_train[:,:]
        idx = np.random.randint(0,x_context_train.shape[0], x_context_train.shape[0])
        for j in range(y_context_train.shape[1]):
            for k in range(y_context_train.shape[2]):
                for l in range(y_context_train.shape[3]):
                    reg = LinearRegression()
                    if sampling == 'noise':
                        reg.fit(x_context[:,i,:].numpy(),y_context_train[:,j,k,l].numpy())
                    elif sampling == 'bootstrap':
                        reg.fit(x_context[idx,i,:].numpy(),y_context_train[idx,j,k,l].numpy())
                        
                    y_context[:,i,j,k,l] = torch.tensor(reg.predict(x_context[:,i,:].numpy()))    
                    y_context_test[:,i,j,k,l] = torch.tensor(reg.predict(x_context_test[:,i,:].numpy()))
        y_context[:,i,:,:,:] = torch.tensor(unravel_2D(scaler.transform(ravel_2D(y_context[:,i,:,:,:])),y_context[:,i,:,:,:].shape))
        y_context_test[:,i,:,:,:] = torch.tensor(unravel_2D(scaler.transform(ravel_2D(y_context_test[:,i,:,:,:])),y_context_test[:,i,:,:,:].shape))
        print(i)
    x_all = x_context_train.unsqueeze(1).expand(-1,factor,-1)
    y_all = torch.tensor(unravel_2D(scaler.transform(ravel_2D(y_context_train)),y_context_train.shape),dtype=torch.float32).unsqueeze(1).expand(-1,factor,-1,-1,-1)
    y_test = torch.tensor(unravel_2D(scaler.transform(ravel_2D(y_test)),y_test.shape),dtype=torch.float32)
    y_test = y_test.view((y_test.shape[0],1,y_test.shape[1],y_test.shape[2],y_test.shape[3]))
   
    labels = np.zeros(y_test.shape[0])
    labels[len(test_idx_ctrl):] = 1
    diagnosis_labels = np.zeros(y_test.shape[0])
    diagnosis_labels[len(test_idx_ctrl):len(test_idx_ctrl)+len(test_idx_SCHZ)] = 1
    diagnosis_labels[len(test_idx_ctrl)+len(test_idx_SCHZ):len(test_idx_ctrl)+len(test_idx_SCHZ)+len(test_idx_ADHD)] = 2
    diagnosis_labels[len(test_idx_ctrl)+len(test_idx_SCHZ)+len(test_idx_ADHD):len(test_idx_ctrl)+len(test_idx_SCHZ)+len(test_idx_ADHD)+len(test_idx_BIPL)] = 3
    return x_context, y_context, x_all, y_all, x_context_test, y_context_test, x_test, y_test, labels, diagnosis_labels, scaler

def apply_dropout_test(m):
    if type(m) == nn.Dropout:
        m.train()

def kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    var_p = torch.exp(logvar_p)
    kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / (var_p) \
             - 1.0 \
             + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum()
    return kl_div

def np_loss(y_hat, y, z_all, z_context):
    BCE = F.binary_cross_entropy(torch.squeeze(y_hat), torch.mean(y,dim=1), reduction="sum")
    KLD = kl_div_gaussians(z_all[0], z_all[1], z_context[0], z_context[1])
    return BCE + KLD
