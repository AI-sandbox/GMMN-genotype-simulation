# GMMN-genotype-simulation

Generative moment matching networks (GMMN) are used to generate new synthetic data. 
This repository includes a python implementation of a GMMN to generate genotypes of a particular ancestry. 
The network can be trained with real data or random features. The architecture of the GMMN using random features is the following:

![alt text](https://github.com/AI-sandbox/GMMN-genotype-simulation/blob/main/doc-fig/gmmn.png)

## Instalation

To install the software, go to the desired folder and enter in the command line interface:
     
     git clone https://github.com/AI-sandbox/GMMN-genotype-simulation.git
     cd GMMN-genotype-simulation

The dependencies are listed in requirements.txt. They can be installed with pip:

    pip install -r requirements.txt  

## Usage

### Parameters

The GMMN can be trained with real data or random features, and the network weights of the random features can be restarted.
By default, the GMMN is trained with random features with a restart. This can be changed in the config_users.txt with the following parameters:

     "loss_generator" : "Random_features" or "freq_matching_loss" (Real data)
     "restart_features": 0 (No restart) or 1 (restart)

The number of generated samples and the input and output path of the data is also in config_users.txt
The parameters and hyperparameters of the model are defined in config.txt.

### Execution

To execute the program enter in the command-line interface:
     
     cd src
     python3 MMN.py


## Input

The input data are two NumPy files:

- Data of all the ancestries: Each row is the SNPs of an individual.
- Labels of the ancestries: Each row is the ancestry of an individual.

The input paths are in config_users.txt:

    "path_founders": 'data_all_anc.npy',
    "path_anc": 'labels_ancestries.npy',
    
### Data

The data used to test the models can be found in the following link: https://figshare.com/articles/dataset/Founders_and_ancestries_ch22_5K_features_/19714480

The founders data contain 5K SNPs of the individuals, and the Ancestries data the ancestries of the individuals.
In the dataset, there are 7 ancestries, but just 4 were used to test the models (Europeans, Africans, East Asians, and South Asians).

In the following links can be found the same data with 10K and 1K SNPs (features): https://figshare.com/articles/dataset/Founders_and_ancestries_ch22_10K_features_/19709461,  https://figshare.com/articles/dataset/Founders_and_ancestries_ch22_1K_features_/19714519


## Output

The default output is the generated data. However, this can be changed in the config_users.txt.

The name of the folder that is created for saving all the outputs is defined with the parameter PATH. If this field is empty the name of the folder is Results + date and the hour of the day is executed. 
The variable save_data can be 1 (save data) or 0 (don't save data), and the variable Save_data_name is the name of the files + samples or labels because the software saves two files, one for the samples and another for the labels.

The evaluation metrics and plots can also be saved.

    ### Name of the folder, where it will be saved all the data
    'PATH': '',
    ### Save data (0/1) and name of the file
    'Save_data': 1,
    'Save_data_name': 'data', 
    #### Evaluation: PCA/UMAP/Isomap
    'Evaluation': ['PCA','Isomap','UMAP']
    ### Save plots (0/1) and evaluation metrics (0/1)
    'Save_plots_loss': 0,    
    'Save_plots_dimreduction': 1,  
    'Save_discriminator': 0, # Accuracies of the classifiers
    


## License

**NOTICE**: This software is available for use free of charge for academic research use only. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as "academic research" must contact the [Stanford Office of Technology Licensing](https://otl.stanford.edu/) for a separate license. This applies to this repository directly and any other repository that includes source, executables, or git commands that pull/clone this repository as part of its function. Such repositories, whether ours or others, must include this notice. Academic users may fork this repository and modify and improve to suit their research needs, but also inherit these terms and must include a licensing notice to that effect.


## Cite

When using this software, please cite the following paper (currently pre-print):

```
@article {Perera2022.04.14.488350,
    author = {Perera, Maria and Montserrat, Daniel Mas and Barrab{\'e}s, M{\'\i}riam and Geleta, Margarita and Gir{\'o}-i-Nieto, Xavier and Ioannidis, Alexander G.},
    title = {Generative Moment Matching Networks for Genotype Simulation},
    elocation-id = {2022.04.14.488350},
    year = {2022},
    doi = {10.1101/2022.04.14.488350},
    publisher = {Cold Spring Harbor Laboratory},
    abstract = {The generation of synthetic genomic sequences using neural networks has potential to ameliorate privacy and data sharing concerns and to mitigate potential bias within datasets due to under-representation of some population groups. However, there is not a consensus on which architectures, training procedures, and evaluation metrics should be used when simulating single nucleotide polymorphism (SNP) sequences with neural networks. In this paper, we explore the use of Generative Moment Matching Networks (GMMNs) for SNP simulation, we present some architectural and procedural changes to properly train the networks, and we introduce an evaluation scheme to qualitatively and quantitatively assess the quality of the simulated sequences.Competing Interest StatementThe authors have declared no competing interest.},
    URL = {https://www.biorxiv.org/content/early/2022/04/14/2022.04.14.488350},
    eprint = {https://www.biorxiv.org/content/early/2022/04/14/2022.04.14.488350.full.pdf},
    journal = {bioRxiv}
}
```
