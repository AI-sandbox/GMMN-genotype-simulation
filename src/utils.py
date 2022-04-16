import sys
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
from torch import nn, optim
from time import gmtime, strftime
from datetime import datetime
from torch.utils.data import TensorDataset

def read_function(path_founders,path_anc): 
    '''
    Objective:
        - Read the Numpy files. 
        - Each individual is splitted in matern and patern. Therefore, each individual correspond to two rows on the dataset. 
        - The input files are numpys.
    Input:
        - path_founders: Path of the SNPs
        - path_anc: Path of the ancestries
    Output: 
        - t_founders: Tensor containing the SNPs of the founders. Duplicate samples (founders) for having in one the maternal SNP and in the other the paternal SNP 
        - labels: Ancestries of the founders
        - dataset: Contain snp and ancestries
                      
    '''
    # Read founders
    founders = np.load(path_founders)
    
    # Read ancestries
    anc = np.load(path_anc)
    
    # Convert numpy array into tensor   
    t_founders = torch.from_numpy(founders)
    labels = torch.from_numpy(anc)
    
    # Compute dataset
    dataset = TensorDataset(t_founders, labels)
    return (t_founders,labels,dataset)



def save_config(name_file,hparams):
    ''' 
    Objective:
        - Save file with the hyper-parameters and parameters
    Input:
        - hparams: Hyper-parameters and parameters
        - name_file: Name of the file
    Output: 
        - None
    '''   
    with open(name_file, 'w') as f:
        f.write(json.dumps(hparams))   

 
        
def save_data(fake_samples,total_labels,hparams):
    ''' 
    Objective:
        - Save generated data and labels in numpy format
    Input:
        - fake_samples: Generated data
        - total_labels: Ancestries of the generated data
        - hparams: Hyper-parameters and parameters
    Output: 
        - None
    '''  
    fake_samples_numpy = fake_samples.numpy() #convert to Numpy array
    f = hparams['PATH'] + '/' + 'samples_' + hparams['Save_data_name']
    #save to file
    np.save(f, fake_samples_numpy)
    
    fake_labels_numpy = total_labels.numpy() #convert to Numpy array
    f_1 = hparams['PATH'] + '/' + 'fake_labels_' + hparams['Save_data_name']
    #save to file  
    np.save(f_1, fake_labels_numpy)

        
def create_directory(name):
    ''' 
    Objective:
        - Create directory
    Input:
        - name: Name of the directory to be created
    Output: 
        - None
    ''' 
    try:
        os.mkdir(name)
    except OSError:
        print ("Creation of the directory %s failed" % name)
    else:
        print ("Successfully created the directory %s " % name)    
        
        
def configure():
    ''' 
    Objective:
        - Read hyper-parameters and parameters, create directory to store data and results, and check if data path exists
    Input:
        - None
    Output: 
        - hparams: Hyper-parameters and parameters
        - path: Path of the directory created
        - path_founders: Path of the data
        - path_anc: Path of the labels
    '''

    # Read hyper-parameters and parameters from the model
    with open('./Model/config.txt') as json_file:
        hparams = hjson.load(json_file)

    # Read hyper-parameters and parameters from the user
    with open('./config_users.txt') as json_file:
        hparams_1 = hjson.load(json_file)
        
    # Join the two json (hparams and hparams_1) in hparams
    hparams.update(hparams_1)
    
    # Add some hyper-parameters to the list
    if hparams['loss_generator'] == 'random_features':
        hparams['learning_rate_g'] = hparams['learning_rate_RF']
        hparams['hidden_size'] = hparams['hidden_size_RF']
    else:
        hparams['learning_rate_g'] = hparams['learning_rate_mean']
        hparams['hidden_size'] = hparams['hidden_size_mean']
    """
    with open('_config.txt') as json_file:
        hparams = hjson.load(json_file)
    """
    
    # Compute start time
    date_time = strftime("%a %d %b %Y %H:%M:%S", gmtime())
    
    # define the name of the directory to be created
    if hparams['PATH'] == '':
        hparams['PATH'] = "./Results" + date_time
    path = hparams['PATH']
    
    # Create directory 
    create_directory(hparams['PATH'])
    
    # Check if data path exist
    path_founders = hparams['path_founders']
    path_anc = hparams['path_anc']
    assert os.path.isfile(path_founders), '{} not found. Is it a file?'.format(path_founders)
    assert os.path.isfile(path_anc), '{} not found. Is it a file?'.format(path_anc)
    return hparams,path,path_founders,path_anc