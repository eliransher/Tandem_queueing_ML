## Welcome to the Tandem queueing analyzer

### This is a GitHub repository which accompanies the paper "Computing the steady-state probabilities of a tandem queueing system, an ML approach."
### Link to archive: chrome-extension://kdpelmjpfafjppnhbloffcjpeomlnpah/https://arxiv.org/pdf/2411.07599

The paper contains three Neural networks. 
NN1: predicts the first inter-departure moments out of a GI/GI/1 queue
NN2: predicts the steady-state probabilities of the number of customers in the systems of a G/GI/1 queue
NN3: predicts the first inter-departure moments out of a G/GI/1 queue

## NN1
### *Input of NN1*: The first five inter-arrival and service time moments of a GI/GI/1 queue must be log values.
### *Output of NN1*: first five inter-departure moments and eight inter-departure auto-correlation values.
The auto-correlation values are according to Table 1 in the paper.


## NN2
### *Input of NN2*: The first five inter-arrival, eight inter-departure auto-correlation values, and service time moments of a G/GI/1 queue. All moments must be in log values. 
### *Output of NN2*: The number of customers in the system steady-state probabilities. 
The auto-correlation values are according to Table 1 in the paper.

## NN3
### *Input of NN3*:  The first five inter-arrival, eight inter-departure auto-correlation values, and service time moments of a G/GI/1 queue. All moments must be in log values. 
### *Output of NN3*: first five inter-departure moments and eight inter-departure auto-correlation values.
The auto-correlation values are according to Table 1 in the paper.

### The file main.py loads all NNs and print an example of all NNs.
### NN modes are in the folder models
### data is the folder data.
