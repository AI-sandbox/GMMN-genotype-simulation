import sys
import os
import math
import json
import hjson
import torch
import allel 
import importlib
import numpy as np
import pandas as pd
from torch import nn, optim
from IPython.display import display
from torch.utils.data import TensorDataset

import Evaluation.Evaluation
import Evaluation.Discriminator
import Precompute_per_anc
import Model.Train_GMMN
import utils

device = torch.device("cuda")


def calls(t_founders,hparams):
    '''
    Objective:
        - Inicialize the MMN, train the MMN, plot the losses and evaluate it.
    Input:
        - t_founders: SNP and ancestries. SNP: t_founders[:][0], ANC: t_founders[:][1] 
        - hparams: Hyper-parameters
    Output: 
        - fake_samples: Generated samples by the generator
        - fake_labels: Labels of the generated samples
    '''   
    
    # Inicialize
    generator,optimizer_g,scheduler,random_features,criterion_MSE = Model.Train_GMMN.inicialize(hparams)
    
    # Precompute per ancestry
    # Precompute mean of real data or Random Features
    precomputed_mean, precompute_mean_RF = Precompute_per_anc.precomuted_info(t_founders,random_features,hparams)
    
    # Train MMN
    generator_losses, z_val,epoch = Model.Train_GMMN.train_epochs(generator, optimizer_g,scheduler, random_features,hparams,criterion_MSE,t_founders,precomputed_mean,precompute_mean_RF)
    
    # Plot loss
    if hparams['Save_plots_loss']:
        Evaluation.Evaluation.plot_loss(generator_losses,hparams)
    
    # Evaluate
    fake_samples,fake_labels = Model.Train_GMMN.evaluate(generator, z_val, t_founders,random_features,hparams,epoch)

    return fake_samples,fake_labels



def GMMN_per_ancestry(founders,hparams,path):
    ''' 
    Objective:
        - Creates the dataloader with SNP and anc. The dataloader contain one anc each time. The GMMN is trained as much times as the number of anc. 
        - Trains the GMMN one time for ancestry calling the script 'calls'
        - To evaluate the data: 
        - Dimencionality reduction tecniques (PCA/UMAP/Isomap) of all the generated data with real samples
        - Accuracy of Classifiers (Classify between real and generated samples) 
    Input:
        - founders: Snp and ancestries. SNP: t_founders[:][0], ANC: t_founders[:][1]
        - hparams: Hyperparameters
        - path: path of the folder
    Output: 
        - none
    ''' 
    # Total number of ancestries
    labels_unique = founders[:][1].unique()
    total_labels = []
    j = 0
    # Traing for each ancestry
    for anc in labels_unique:
        index = []
        # Select indexes of each anc
        for i in range(founders[:][1].shape[0]):
            if founders[i][1] == anc:
                index.append(i)  
        # Take the snp and labels of a particular ancestry 
        t_founders_anc = founders[index][0]
        labels_anc = founders[index][1]
        dataset = TensorDataset(t_founders_anc, labels_anc)

        # Change the name of the file to save the different figures
        hparams['Name_file'] = 'fig_' + str(j)
        j += 1
        # Train the MMN and return the generated samples
        fake_samples_return,_ = calls(dataset,hparams)

        # Concat the generated samples from each ancestry and the same with labels
        if anc == labels_unique[0]:
            fake_samples = fake_samples_return.detach().clone()
            labels_fake_anc = torch.ones(len(fake_samples_return)) * anc
            total_labels = labels_fake_anc.detach().clone()
        else:
            fake_samples = torch.cat((fake_samples,fake_samples_return),dim=0)
            labels_fake_anc = torch.ones(len(fake_samples_return)) * anc
            total_labels = torch.cat((total_labels,labels_fake_anc),dim=0)
        
    hparams['Name_file'] = 'fig_' + str(j)
    
    # Train discriminator classifiers: Evaluation metric (Accuracy)
    if hparams['Save_discriminator']:
        Evaluation.Discriminator.discriminator(path,founders[:][0],fake_samples,founders[:][1],total_labels,hparams)            
    
    # Save Generated data and labels
    if hparams['Save_data']:
        utils.save_data(fake_samples,total_labels,hparams)   
     
    # Compute PCA of the founders (Real samples) with generated data of each ancestry
    ancestry_names = {'EUR':0, 'AFR':1, 'EAS':3,'SAS':5}   
    total_labels = pd.Series(total_labels)
    # Perform PCA / UMAP / isomap
    if hparams['Save_plots_dimreduction']:
        Evaluation.Evaluation.perform_dim_reduction_plots(founders[:][0],founders[:][1],hparams,ancestry_names,fake_samples,total_labels) 

        
    
def main():
    ''' 
    Objective:
        - Read config file, where there are the hyperparameters of the model
        - Read the data
        - Define some more parameters 
    Input:
        - None
    Output: 
        - none
    '''   
    import time
    start = time.time()    

    # Read hyper-parameters and parameters, create directory to store data and results, and check if data path exists
    hparams,path,path_founders,path_anc = utils.configure()
    
    # Read data
    # t_founders: Snp and ancestries. SNP: t_founders[:][0], ANC: t_founders[:][1]
    t_founders,labels,founders  = utils.read_function(path_founders,path_anc)

    # Add fields to hyperparameters
    hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparams['noise_size'] = t_founders.shape[1] # Number of snp
    hparams['num_inputs'] = t_founders.shape[1] # Number of snp
    hparams['Output_size_random_features'] = int(t_founders.shape[1]*hparams['Output_size_random_features'])
    hparams['Name_file'] = 'fig_0'

    # Write the config of parameters used in the execution
    #name_file = hparams['PATH']+ '/' + '_config.txt'
    #utils.save_config(name_file,hparams) 

    # Train the MMN one time for ancestry
    GMMN_per_ancestry(founders,hparams,path)
   
    end = time.time()
    time = end - start
 
    # Save time of each execution
    import csv
    name_file = hparams['PATH']+ '/' + 'time'
    with open(name_file, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Time'])
        writer.writerow({'Time':round(time,4)}) 


if __name__ == '__main__':
    main()