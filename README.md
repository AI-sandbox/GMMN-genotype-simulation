# GMMN-genotype-simulation

Generative moment matching networks (GMMN) are used to generate new synthetic data. We use a GMMN to generate genotypes of a particular ancestry. The network can be trained with the real data or with Random Features computed with a network.

## Usage

The GMMN can be trainned with Real Data or Random Features, and the network weights of the Random Features can be restarted.
By default the GMMN is trained with Random Features with restart. This can be changed in the config_users.txt with the following parameters:

     1. "loss_generator" : "Random_features" or "freq_matching_loss"
     2. "restart_features": 0 (No restart) or 1 (restart)



## Input



## Output