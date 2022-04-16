import sys
sys.path.append('../')
import os
import math
import torch
import importlib
import numpy as np
import pandas as pd
from torch import nn, optim
from sklearn.metrics import mean_squared_error
from time import gmtime, strftime

def compute_mean_random_features_anc(t_founders,anc,hparams,random_features):
    '''
    Objective:
        - Compute the mean of the Random Features of the Real data with a network.
    Input:
        - t_founders: SNP+ANC
        - anc: Particular ancestry
        - random_features: Linear network with a ReLU activation
        - hparams: Hyper-parameters and parameters
    Output: 
        - precompute_mean_RF: Contain the mean values of features of the real data of a particular ancestry
    '''  
    idx = t_founders[:][1] == anc.item()
    # Call network with real data
    ff_anc = random_features(t_founders[idx.nonzero().squeeze(1)][0].to(hparams['device']),False)
    
    # Compute mean of each snp (column)
    return torch.mean(ff_anc,axis=0,dtype=torch.float32)  



def compute_mean_snp(t_founders,anc,hparams):
    '''
    Objective:
        - Compute mean of each SNP of a particular ancestry
    Input:
        - t_founders: SNP+ANC
        - anc: Particular ancestry
        - hparams: Hyper-parameters and parameters 
    Output: 
        - precomputed_mean: Contain the mean values of the SNPs of a particular ancestry
    '''  
    # Compute the mean of each snp (columns)
    idx = t_founders[:][1] == anc.item()
    return torch.mean(t_founders[idx.nonzero().squeeze(1)][0],axis=0,dtype=torch.float32)   


def inicialize_precomuted_info(hparams,labels_unique):
    '''
    Objective:
        - Inicialize tensors where the values are going to be precomputed
    Input:
        - hparams: Hyper-parameters and parameters
        - labels_unique: Number of ancestries
    Output: 
        - precompute_mean_RF: Contain the mean values of Random Features of the SNPs of a particular ancestry
        - precomputed_mean: Contain the mean values of the SNPs of a particular ancestry
    '''  
    #### INICIALIZE
        
    # Random Features
    if hparams['loss_generator'] == "random_features":
        precompute_mean_RF = torch.zeros(hparams['Output_size_random_features'])
        
    # Mean
    else:
        precomputed_mean = torch.zeros(hparams['num_inputs'])
    
    if hparams['loss_generator'] == "random_features":
        return precompute_mean_RF,0
    else:
        return 0,precomputed_mean
    
    
def precomuted_info(t_founders,random_features,hparams):
    '''
    Objective:
        - Compute the mean of Random Features of a particular ancestry 
        or
        - Compute mean of each SNP of a particular ancestry 
    Input:
        - t_founders: SNP+ANC
        - random_features: Linear network with a ReLU or cosinus activation
        - hparams: Hyper-parameters and parameters
    Output: 
        - precompute_mean_RF: Contain the mean values of Random Features of the SNPs of a particular ancestry
        - precomputed_mean: Contain the mean values of the SNPs of a particular ancestry
    '''  
    labels_unique = t_founders[:][1].unique()
    
    #### INICIALIZE
    precompute_mean_RF,precomputed_mean = inicialize_precomuted_info(hparams,labels_unique)
            
    # Compute for ach ancestry
    for anc in labels_unique:

        # Compute random fourier features        
        if hparams['loss_generator'] == "random_features":
            precompute_mean_RF = compute_mean_random_features_anc(t_founders,anc,hparams,random_features) 
        
        # Compute mean of snp of each anc
        else:
            precomputed_mean = compute_mean_snp(t_founders,anc,hparams)            
                
    # Return tensors depending on the loss of the generator    
    
    if hparams['loss_generator'] == "random_features":
        return 0, precompute_mean_RF
    else:
        return precomputed_mean, 0