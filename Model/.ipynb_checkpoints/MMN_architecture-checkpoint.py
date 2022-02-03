import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Function
import time
import importlib
import itertools  
import math
from IPython.display import display
device = torch.device("cuda")
 


class Generator(torch.nn.Module):
    '''
    Objective:
        - Generate samples from noise
        - If hparams['Quantizer'] is True, a binary quantizer is used (Default TRUE), if not a sigmoid and the samples are manually binarized.
    Input:
        - Noise
        - hparams: hyperparameters and definition of parameters
    Output: 
        - Generated samples
    ''' 
    def __init__(self,p):
        self.hparams = p
        super().__init__()
        
        if self.hparams['Quantizer']:
            ## Architecture_1
            self.arch1 = nn.Sequential(
                nn.Linear(self.hparams['noise_size'], self.hparams['hidden_size']),
                nn.ReLU(),
                nn.BatchNorm1d(self.hparams['hidden_size']),
                nn.Linear(self.hparams['hidden_size'], self.hparams['num_inputs']),
                BinaryQuantizeWrapper()

            )        
        else:
            ## Architecture_1
            self.arch1 = nn.Sequential(
                nn.Linear(self.hparams['noise_size'], self.hparams['hidden_size']),
                nn.ReLU(),
                nn.BatchNorm1d(self.hparams['hidden_size']),
                nn.Linear(self.hparams['hidden_size'], self.hparams['num_inputs']),
                nn.Sigmoid()

            )         
    def forward(self, x):

        y = self.arch1(x.float())
        return y
    
    
class BinaryQuantize(Function): 
    '''
    Objective:
        - Convert sigmoid output between 0-1 into 0 or 1, depending of a threshold
    ''' 
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        x = (out + 1)/2
        return x
    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

class BinaryQuantizeWrapper(nn.Module):
    '''
    Objective:
        - Call BinaryQuantize class
    Parameters k and t: Transform the sigmoid into a hard threshold (1 if x>0.5 else 0)
    ''' 
    def __init__(self):
        super(BinaryQuantizeWrapper, self).__init__()
        self.k = nn.Parameter(torch.tensor([10]).float(), requires_grad=False)
        self.t = nn.Parameter(torch.tensor([0.1]).float(), requires_grad=False)
    def forward(self, x):
        return BinaryQuantize().apply(x, self.k, self.t)
    

class Random_Features(torch.nn.Module):
    '''
    Objective:
        - Compute Random features from real and generated data.
        - They are computed with a linear layer + Relu.
        - If reset_weight is True (hparams['restart_features']), every 500 epochs the weights are reset (Default TRUE).
    Input:
        - Real or fake samples (snp)
        - hparams: hyperparameters and definition of parameters
    Output: 
        - Random Features
    ''' 
    def __init__(self,p):
        self.hparams = p
        super().__init__()
        
        self.arch1 = nn.Sequential(
            nn.Linear(self.hparams['num_inputs'], self.hparams['Output_size_random_features']),
            nn.ReLU()
        )        
                
    def forward(self, x, reset_weight):

        if reset_weight:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))  

        y = self.arch1(x.float())
        return y

    
