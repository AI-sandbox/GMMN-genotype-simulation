import sys
sys.path.append('../')
import os
import time
import math
import torch
import allel 
import importlib
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import chain
from itertools import cycle
from torch import nn, optim
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset
from sklearn.metrics import mean_squared_error
from umap.parametric_umap import ParametricUMAP
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.manifold import Isomap
import matplotlib.lines as mlines
from datetime import datetime


def PCA_real_data(t_founders,labels):
    '''
    Objective:
        - Train PCA, and compute the two principal components with real data.
    Input:
        - t_founders: Tensor containing the snp of the founders
        - labels: Ancestries of the founders
    Output:
        - pca: Model
        - results_df_real: Dataset with the two principal components of PCA and the labels of the ancestry 
    '''
    
    ## REAL DATA
    pca = PCA(n_components=2)

    # Fit the PCA to the real data, and then transform the data into its principal components
    principal_components_real = pca.fit_transform(t_founders)

    pcs_df_real = pd.DataFrame(data = principal_components_real, columns = ['principal component 1', 'principal component 2'])
    ## Concatenate the labels as the last dimension of the principal components to create the results dataframe
    results_df_real = pd.concat([pcs_df_real, labels], axis=1)
    results_df_real.columns = ['principal component 1', 'principal component 2', 'labels']  
    return pca,results_df_real

def UMAP_real_data(t_founders,labels):    
    '''
    Objective:
        - - Train UMAP, and compute the two principal components with real data.
    Input:
        - t_founders: Tensor containing the snp of the founders
        - labels: Ancestries of the founders
    Output:
        - pca: Model
        - results_df_real: Dataset with the two principal components of UMAP and the labels of the ancestry 
    '''
    ## Initialize the Embedder for parametric UMAP
    embedder = ParametricUMAP(n_epochs = 50, verbose=True, random_state=42)
    
    ## REAL DATA
    # Fit the Parametric UMAP to the real data, and then transform the data into its principal components
    embedding_real = embedder.fit_transform(t_founders)
    embedding_df_real = pd.DataFrame(data = embedding_real, columns = ['Embedding 1', 'Embedding 2'])
    ## Concatenate the labels as the last dimension of the principal components to create the results dataframe
    results_df_real = pd.concat([embedding_df_real, labels], axis=1)
    results_df_real.columns = ['Embedding 1', 'Embedding 2', 'labels']
    return embedder,results_df_real


def Isomap_real_data(t_founders,labels):
    '''
    Objective:
        - - Train Isomap, and compute the two principal components with real data.
    Input:
        - t_founders: Tensor containing the snp of the founders
        - labels: Ancestries of the founders
    Output:
        - pca: Model
        - results_df_real: Dataset with the two principal components of isomap and the labels of the ancestry 
    '''
    ## REAL DATA
    
    isomap = Isomap(n_components=2) 
    # Fit the PCA to the real data, and then transform the data into its principal components
    principal_components_real_1 = isomap.fit_transform(t_founders)

    pcs_df_real_1 = pd.DataFrame(data = principal_components_real_1, columns = ['principal component 1', 'principal component 2'])
    ## Concatenate the labels as the last dimension of the principal components to create the results dataframe
    results_df_real_1 = pd.concat([pcs_df_real_1, labels], axis=1)
    results_df_real_1.columns = ['principal component 1', 'principal component 2', 'labels']  
    return isomap,results_df_real_1



def PCA_fake_data(fake_samples,labels_fake,pca):
    '''
    Objective:
        - Compute two principal components of PCA with fake data from PCA trainned with the real data.
    Input:
        - fake_samples: Tensor containing the snp of the generated data
        - labels_fake: Ancestries of the generated data
        - pca: Model trainned
    Output:
        - results_df_fake: Dataset with the two principal components of PCA and the labels of the ancestry 
    '''
    ## FAKE DATA
    # Transform the fake data into the PCA compoments
    principal_components_fake = pca.transform(fake_samples)
    pcs_df_fake = pd.DataFrame(data = principal_components_fake, columns = ['principal component 1', 'principal component 2'])    
    ## Annote the fake labels differently, so as to be able to identify them
    labels_fake_identified = []
    for l in labels_fake:
        ## For some reason the labels sometimes are Tensors and sometimes are ints; getting ready for that
        if type(l) == torch.Tensor:
            l = l.item()
        labels_fake_identified.append(l+10)
    labels_fake_identified = pd.Series(labels_fake_identified)
    ## Concatenate the labels as the last dimension of the principal components to create the results dataframe
    results_df_fake = pd.concat([pcs_df_fake, labels_fake_identified], axis=1)
    results_df_fake.columns = ['principal component 1', 'principal component 2', 'labels']
    return results_df_fake


def UMAP_fake_data(fake_samples,labels_fake,embedder):
    '''
    Objective:
        - Compute two principal components of UMAP with fake data from PCA trainned with the real data.
    Input:
        - fake_samples: Tensor containing the snp of the generated data
        - labels_fake: Ancestries of the generated data
        - embedder: Model trainned
    Output:
        - results_df_fake: Dataset with the two principal components of UMAP and the labels of the ancestry 
    '''
    ## FAKE DATA
    # Transform the fake data into the Parametric UMAP embeddings
    embedding_fake = embedder.transform(fake_samples)
    embedding_df_fake = pd.DataFrame(data = embedding_fake, columns = ['Embedding 1', 'Embedding 2'])
    
    ## If Type_GMMN == "Gan" we do not have label information. 
    ## Otherwise, annote the fake labels differently, so as to be able to identify them
    labels_fake_identified = []
    for l in labels_fake:
        ## For some reason the labels sometimes are Tensors and sometimes are ints; getting ready for that
        if type(l) == torch.Tensor:
            l = l.item()
        labels_fake_identified.append(l+10)
    labels_fake_identified = pd.Series(labels_fake_identified)
    ## Concatenate the labels as the last dimension of the embeddings to create the results dataframe
    results_df_fake = pd.concat([embedding_df_fake, labels_fake_identified], axis=1)
    results_df_fake.columns = ['Embedding 1', 'Embedding 2', 'labels']    
    return results_df_fake
    

def Isomap_fake_data(fake_samples,labels_fake,isomap):
    '''
    Objective:
        - Compute two principal components of Isomap with fake data from PCA trainned with the real data.
    Input:
        - fake_samples: Tensor containing the snp of the generated data
        - labels_fake: Ancestries of the generated data
        - isomap: Model trainned
    Output:
        - results_df_fake: Dataset with the two principal components of Isomap and the labels of the ancestry 
    '''
    ## FAKE DATA
    # Transform the fake data into the PCA compoments
    principal_components_fake = isomap.transform(fake_samples)
    pcs_df_fake = pd.DataFrame(data = principal_components_fake, columns = ['principal component 1', 'principal component 2'])    
    ## Annote the fake labels differently, so as to be able to identify them
    labels_fake_identified = []
    for l in labels_fake:
        ## For some reason the labels sometimes are Tensors and sometimes are ints; getting ready for that
        if type(l) == torch.Tensor:
            l = l.item()
        labels_fake_identified.append(l+10)
    labels_fake_identified = pd.Series(labels_fake_identified)
    ## Concatenate the labels as the last dimension of the principal components to create the results dataframe
    results_df_fake = pd.concat([pcs_df_fake, labels_fake_identified], axis=1)
    results_df_fake.columns = ['principal component 1', 'principal component 2', 'labels']
    return results_df_fake


def perform_dim_reduction_plots(t_founders,labels,hparams,ancestry_names,fake_samples,labels_fake):
    '''
    Objective:
        - Compute PCA / UMAP / Isomap with the real and generated data
    Input:
        - t_founders: Tensor containing the snp of the founders
        - labels: Ancestries of the founders
        - hparams: Hyper-parameters and parameters
        - ancestry_names: Dictionary of the ancestries. Maping numbers to ancestries. (#ancestry_names = {'EUR':0, 'AFR':1, 'AMR':2, 'EAS':3, 'OCE':4,'SAS':5,'WAS':6})
        - fake_samples: Generated samples
        - labels_fake: Labels of the generated samples. If we are training a conditional gan or GMMN_for_ancestry, we know it. If not is empty
    Output:
        - none
    '''
    
    # If there is no quantizer, convert generated samples to binary output (0 or 1)
    if hparams['Quantizer'] == 0:
        fake_samples[fake_samples >= 0.5] = 1
        fake_samples[fake_samples < 0.5] = 0
        
    labels = pd.Series(labels)
    labels_fake = pd.Series(labels_fake)
    # Used to change the shape of the real data and the generated
    shape_t = [0] * len(t_founders)
    if len(labels_fake)!=0:
        shape = [1] * len(labels_fake)
    ## Create dictionary for labels; the real labels will be a points, the fake labels will be a cross
    markers = {}
    ## Save all unique labels appeared both in real and in fake data, to set the legend
    all_labels = []
    for ancestry in set(labels):
        ## For some reason the labels sometimes are Tensors and sometimes are ints; getting ready for that
        if type(ancestry) == torch.Tensor:
            ancestry = ancestry.item()
        markers[ancestry] = "."
        all_labels.append(ancestry)
    for ancestry in set(labels_fake):
        ## For some reason the labels sometimes are Tensors and sometimes are ints; getting ready for that
        if type(ancestry) == torch.Tensor:
            ancestry = ancestry.item()
        markers[ancestry+10] = "X"
        all_labels.append(ancestry)
    #print('markers',markers)
    ## Create specific a palette so that each color represents real and fake samples of a same ancestry
    palette_set2 = sns.color_palette("Set2", 8)
    actual_palette = {0:palette_set2[0], 1:palette_set2[1], 2:palette_set2[2], 3:palette_set2[3], 4:palette_set2[4], 5:palette_set2[5], 6:palette_set2[6], 10:palette_set2[0], 11:palette_set2[1], 12:palette_set2[2], 13:palette_set2[3], 14:palette_set2[4], 15:palette_set2[5], 16:palette_set2[6]}

    ## REAL DATA
    if 'PCA' in hparams['Evaluation']:
        pca,results_df_real_PCA = PCA_real_data(t_founders,labels)   

    if 'UMAP' in hparams['Evaluation']:
        emb,results_df_real_UMAP = UMAP_real_data(t_founders,labels)
        
    if 'Isomap' in hparams['Evaluation']:
        isomap,results_df_real_Isomap = Isomap_real_data(t_founders,labels)    

    for method in hparams['Evaluation']: 
        
        ## FAKE DATA
        if method == 'PCA':
            results_df_fake_PCA = PCA_fake_data(fake_samples,labels_fake,pca)
        elif method == 'UMAP':
            results_df_fake_UMAP = UMAP_fake_data(fake_samples,labels_fake,emb)
        else:
            results_df_fake_Isomap = Isomap_fake_data(fake_samples,labels_fake,isomap)

        if method == 'PCA':
            x_name='principal component 1'
            y_name='principal component 2'
            results_df_real = results_df_real_PCA
            results_df_fake = results_df_fake_PCA
            title = 'PCA'
        elif method == 'UMAP':
            x_name='Embedding 1'
            y_name='Embedding 2'   
            results_df_real = results_df_real_UMAP
            results_df_fake = results_df_fake_UMAP
            title = 'UMAP'
        else:
            x_name='principal component 1'
            y_name='principal component 2'
            results_df_real = results_df_real_Isomap
            results_df_fake = results_df_fake_Isomap 
            title = 'Isomap'
            
        # PLOT 
        plt.figure()
        f = sns.scatterplot(x=x_name, y=y_name, hue='labels', data=results_df_real, linewidth=0.1, palette=actual_palette, style='labels', markers=markers, edgecolor='black', s=50, alpha=0.3)
        f = sns.scatterplot(x=x_name, y=y_name, hue='labels', data=results_df_fake, linewidth=0.1, palette=actual_palette, style='labels', markers=markers, edgecolor='black', s=70)
        plt.xlabel(x_name, fontsize = 14)
        plt.ylabel(y_name, fontsize = 14)
        
        plt.title(title + ': snp {},Real {},Fake {} '.format(hparams['num_inputs'],t_founders.shape[0],fake_samples.shape[0]), fontsize = 12)
        
        
        handles, labels  =  f.get_legend_handles_labels()
        ## All possible ancestries are 'EUR':0, 'AFR':1, 'AMR':2, 'EAS':3, 'OCE':4,'SAS':5,'WAS':6
        handles = []
        if 0 in all_labels:
            EUR = mlines.Line2D([], [], color=actual_palette[0], marker='_', linestyle='None', mew=2,
                              markersize=15, label='EUR')
            handles.append(EUR)
        if 1 in all_labels:
            AFR = mlines.Line2D([], [], color=actual_palette[1], marker='_', linestyle='None', mew=2,
                              markersize=15, label='AFR')
            handles.append(AFR)
        if 2 in all_labels:
            AMR = mlines.Line2D([], [], color=actual_palette[2], marker='_', linestyle='None', mew=2,
                              markersize=15, label='AMR')
            handles.append(AMR)
        if 3 in all_labels:
            EAS = mlines.Line2D([], [], color=actual_palette[3], marker='_', linestyle='None', mew=2,
                              markersize=15, label='EAS')
            handles.append(EAS)
        if 4 in all_labels:
            OCE = mlines.Line2D([], [], color=actual_palette[4], marker='_', linestyle='None', mew=2,
                              markersize=15, label='OCE')
            handles.append(OCE)
        if 5 in all_labels:
            SAS = mlines.Line2D([], [], color=actual_palette[5], marker='_', linestyle='None', mew=2,
                              markersize=15, label='SAS')
            handles.append(SAS)
        if 6 in all_labels:
            WAS = mlines.Line2D([], [], color=actual_palette[6], marker='_', linestyle='None', mew=2,
                              markersize=15, label='WAS')
            handles.append(WAS)

        plt.legend(handles=handles, loc='upper right', fontsize = 14)
        plt.grid()
        plt.show()

        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 

        plt.rcParams['font.family'] = 'Serif' 
        
        if hparams['Save_plots_dimreduction']:
            if method == 'PCA':
                name_file = hparams['PATH']+ '/' + hparams['Name_file'] + '_PCA.png'
                plt.savefig(name_file,dpi=300)
                plt.close() 
            elif method == 'Isomap':
                name_file = hparams['PATH']+ '/' + hparams['Name_file'] + '_Isomap.png'
                plt.savefig(name_file, dpi=300)
                #plt.savefig(name_file)
                plt.close() 
            else:
                name_file = hparams['PATH']+ '/' + hparams['Name_file'] + '_UMAP.png'
                plt.savefig(name_file,dpi=300)
                plt.close() 

        
def plot_loss(generator_losses,hparams):
    '''
    Objective:
        - Plot the generator loss
    Input:
        - generator_losses: Loss generator
        - hparams: Hyper-parameters and parameters
    Output: 
        - Plot of the losses
    '''
    # Plot genarator loss
    plt.close()
    #plt.figure(figsize=(10, 8))
    plt.figure
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Loss', fontsize = 12)
    plt.plot(generator_losses, label='generator')
    plt.title('Generator loss', fontsize = 12)
    plt.legend()
    plt.show()
    
    if hparams['Save_plots_loss']:
        name_file = hparams['PATH'] + '/' + hparams['Name_file'] + '_loss.png'
        plt.savefig(name_file)  
    plt.close()    

    