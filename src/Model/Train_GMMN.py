import os
import time
import math
import json
import hjson
import torch
import allel 
import csv
import importlib
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import chain
from itertools import cycle
from torch import nn, optim
import matplotlib.pyplot as plt
from IPython.display import display
from torch.utils.data import TensorDataset
from sklearn.metrics import mean_squared_error
from time import gmtime, strftime
from datetime import datetime

import Model.MMN_architecture
import Evaluation.Evaluation
import Precompute_per_anc

device = torch.device("cuda")

@torch.no_grad()
def evaluate(generator, z_val, t_founders,random_features,hparams,epoch):
    '''
    Objective:
        - Evaluate quantitatively and qualitatively the samples generated from the generator.
        - Qualitatively: PCA/Isomap/UMAP
        - Quantitatively: Discriminator, called in the function GMMN_per_anc
    Input:
        - generator: Trained generator
        - z_val: Noise that enters the generator and we want to be converted into samples
        - t_founders: Snp and ancestries. SNP: t_founders[:][0], ANC: t_founders[:][1]
        - random_features: : Linear layer + Relu 
        - hparams: Hyper-parameters and parameters
        - epoch: Number of epoch
    Output: 
        - fake_samples: Samples generated by the generator
        - labels_fake: Anc of the fake samples
    '''  

    ancestry_names = {'EUR':0.0, 'AFR':1.0, 'EAS':3.0,'SAS':5.0}
    
    generator.eval()
    
    # Generate fake samples
    fake_samples = generator(z_val).cpu()
    # generate the corresponding labels to compute a dimentionality reduction tecnique
    labels_fake = torch.ones(len(fake_samples)) * t_founders[1][1]

    # t_founders[:][0] : snp, t_founders[:][1]: labels
    # Perform PCA, LDA , UMAP or isomap
    if hparams['Save_plots_dimreduction']:
        Evaluation.Evaluation.perform_dim_reduction_plots(t_founders[:][0],t_founders[:][1],hparams,ancestry_names,fake_samples,labels_fake)

    return fake_samples, labels_fake

def compute_random_features(t_founders,precompute_mean_RF,random_features,fake_samples,hparams,epoch,i):
    '''
    Objective:
        - Compute Random Features of generated data
        - Compute the mean of the features of each snp
        - If hparams['restart_features'] is True, the RF network weights are restarted every 500 epochs. Then, the Random Features are computed for Real and Generated data.
    Input:
        - t_founders: SNP + ANC
        - precompute_mean_RF: Contain a mean of the features of the Real Data
        - random_features: Linear layer + Relu 
        - fake_samples: Samples generated by the generator 
        - criterion_MSE: Definition of mean square error
        - hparams: Hyper-parameters and parameters
        - epoch: Number of epoch
        - i: Number of batch
    Output: 
        - ff_fake_mean: Mean of the RF of generated data
        - precompute_mean_RF: Mean of the RF of the real data. When the weights of the network are restarted the features of real data are recomputed.
    '''     
    # Call the network
    # If epoch % 500 == 0, reset weights and recompute real features
    if hparams['restart_features']:
        if epoch % 100 == 0 and i == 0:
            ff_fake_val = random_features(fake_samples, True) 
            _, precompute_mean_RF  = Precompute_per_anc.precomuted_info(t_founders,random_features,hparams)
        else:
            ff_fake_val = random_features(fake_samples, False)
    else:
        ff_fake_val = random_features(fake_samples, False)
    
    # Compute mean of each snp (column)
    ff_fake_mean = torch.mean(ff_fake_val,axis=0,dtype=torch.float32)  
          
    return ff_fake_mean,precompute_mean_RF     
    
    
    
def optimize_generator(z,hparams,generator,t_founders,precompute_mean_RF,precomputed_mean,criterion_MSE,epoch,optimizer_g,scheduler,random_features,i): 
    '''
    Objective:
        - Optimize the generator.
        - Compute Random Features of the generated data, if necesary
        - Compute mean of the real data or Random Features
        - The generator loss is computed : 
                 1. MSE of the differences of snp between real and fake samples
                 2. MSE of the differences of snp between real and fake random features
    Input:
        - z: noise 
        - Hparams : hyperparameters
        - Generator: Generates samples from noise
        - t_founders: SNP + ANC
        - precomputed_mean: Mean of the SNP of a particular ancestry
        - precompute_mean_RF: Mean of the Random Features of a particular ancestry
        - criterion_MSE: Mean square error loss
        - epoch: Number of epoch
        - optimizer_g
        - scheduler: Learning rate schedule
        - random_features: Compute kind of variance of the snp with a neural network of each ancestry and return the mean of snips. This emulates random fourier features.
        - i: Number of batch
        
    Output: 
        - loss_g: Generator loss:'freq_matching_loss'/'random_features'
        - fake_samples: Generated samples 
    '''   
    
    # Enter noise to the generator  
    fake_samples = generator(z)  
    
    #hparams['num_generated_data'] = len(t_founders[:][1])
    
    # Compute random features and the mean of the features
    if hparams['loss_generator'] == "random_features":
        mean_snp_fakesamples,precompute_mean_RF = compute_random_features(t_founders,precompute_mean_RF,random_features,fake_samples,hparams,epoch,i)
        mean_snp_realsamples = precompute_mean_RF.float().cuda()
            
    # Compute the mean of the data
    else:
        # Compute the mean of each SNPs (columns)
        mean_snp_fakesamples = torch.mean(fake_samples,axis=0,dtype=torch.float32)
        mean_snp_realsamples = precomputed_mean.cuda()
   
    # Compute loss 
    loss_g = criterion_MSE(mean_snp_fakesamples,mean_snp_realsamples).float()
     
    if hparams['loss_generator'] == "freq_matching_loss":
        loss_g = 100*loss_g    


    # Backpropagate
    if hparams['loss_generator'] == "random_features":
        loss_g.backward(retain_graph=True)
    else:
        loss_g.backward()

    # Update weights
    optimizer_g.step() 

    # learning rate scheduler
    if hparams['lr_scheduler']: 
        scheduler.step()
    
    return (loss_g,fake_samples,precompute_mean_RF)


def train_batch(t_founders, generator, optimizer_g,scheduler,random_features,hparams,criterion_MSE,precompute_mean_RF,precomputed_mean,epoch,i): 
    '''
    Objective:
        - Train the MMN (generator)
        - Generate noise and optimize the generator network calling the function optimize_generator
    Input:
        - t_founders: SNP+ANC
        - Generator: Generates samples from noise
        - Optimizer_g: Optimizer of the generator
        - scheduler: Learning rate schedule
        - random_features: Compute Random Features of each ancestry with a neural network and return the mean of SNPs.  
        - Hparams: Hyper-parameters and parameters
        - criterion_MSE: Mean square error loss
        - precomputed_mean: Mean of the SNP of a particular ancestry
        - precompute_mean_RF: Mean of the Random Features of a particular ancestry
        - epoch
    Output: 
        - loss_g: Loss generator:'freq_matching_loss'/'random_features'
    '''   

    generator.train()

    ####################
    # OPTIMIZE GENERATOR
    ####################

    # Reset gradients
    optimizer_g.zero_grad()
    
    # Generate fake samples (noise)
    z = torch.randn(hparams['batch_size'], hparams['noise_size'], device=device)
    
    # Optimize generator
    loss_g,fake_samples,precompute_mean_RF = optimize_generator(z,hparams,generator,t_founders,precompute_mean_RF,precomputed_mean,criterion_MSE,epoch, optimizer_g,scheduler,random_features,i)

    return loss_g.item(),precompute_mean_RF





def train_epochs(generator, optimizer_g,scheduler,random_features,hparams,criterion_MSE,t_founders,precomputed_mean,precompute_mean_RF):
    '''
    Objective:
        - Train the MMN (generator) and compute the generator losses
        - The generator loss is computed : 
                 1. MSE of the differences of snp between real and fake samples
                 2. MSE of the differences of snp between real and fake random features
        - Call the function train_batch for each epoch and batch and evaluate the generated samples every 1000 epochs
    Input:
        - generator: Network that generates samples from noise
        - optimizer_g: Optimizer of the generator
        - scheduler: learning rate schedule
        - random_features: Compute Random Features of each ancestry with a neural network and return the mean of SNPs.    
        - hparams: Hyper-parameters and parameters
        - criterion_MSE: Mean square error loss
        - t_founders: SNP+ANC
        - precomputed_mean: Mean of the SNP of a particular ancestry
        - precompute_mean_RF: Mean of the Random Features of a particular ancestry
    Output: 
        - generator_losses: Loss generator
        - z_val: Noise 
        - epoch: Number of epoch
    '''       
    # Init list to save the evolution of the losses
    generator_losses = []
        
    # Count number of batches
    num_batches = round(len(t_founders[:][1])/hparams['batch_size'],0)+1
        
    # Generate noise
    z_val = torch.randn(hparams['num_generated_data'], hparams['noise_size'], device=device)    
    
    # Train the MMN with epochs and batches
    for epoch in range(hparams['num_epochs']):
        for i in range(int(num_batches)):
            loss_g,precompute_mean_RF = train_batch(t_founders, generator,optimizer_g,scheduler,random_features,hparams,criterion_MSE,precompute_mean_RF,precomputed_mean,epoch,i)        
        
        generator_losses.append(loss_g)       
        
        # Every 1000 epochs evaluate the generated samples
        if epoch % 1000 == 0:
            print("epoch: {}/{} batch: {}/{} G_loss: {}".format(epoch+1,hparams['num_epochs'],i+1,num_batches,loss_g))
            evaluate(generator, z_val, t_founders,random_features,hparams,epoch)   
       
    return generator_losses, z_val,epoch


def inicialize(hparams):
    '''
    Objective:
        - Inicialize the generator, optimizer_g and criterion
        - There are 2 types of optimizers: Adam or SGD
    Input:
        - Hparams : hyperparameters
    Output: 
        - Generator: Generates samples from noise
        - Optimizer_g: Optimizer of the generator
        - schedule: learning rate schedule
        - random_features: Compute Random Features of each ancestry with a neural network and return the mean of SNPs.
        - criterion_MSE: Mean square error loss
    '''
    # Inicialize generator 
    generator = Model.MMN_architecture.Generator(hparams).to(hparams['device']) 

    # Inicializate random_features network
    random_features = Model.MMN_architecture.Random_Features(hparams).to(hparams['device']) 
    
    # Inicialize optimizers
    if hparams['optimizer'] == "Adam":
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=hparams['learning_rate_g'], betas=hparams['betas'])
    else:
        optimizer_g = torch.optim.SGD(generator.parameters(), lr=hparams['learning_rate_g'],momentum=hparams['momentum'])
    
    # Define learning rate schedule  
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer_g, start_factor=0.5, total_iters=4)
    
    # Define criterion
    criterion_MSE = nn.MSELoss()        
 
    return generator,optimizer_g,scheduler,random_features,criterion_MSE

