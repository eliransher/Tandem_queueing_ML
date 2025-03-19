import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import torch
from math import factorial
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pickle as pkl


class my_Dataset_moms_testset2(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms=5, num_ser_moms=5,
                 num_depart_corrs=5):
        self.data_paths = data_paths
        self.max_lag = max_lag
        self.max_power_1 = max_power_1
        self.max_power_2 = max_power_2
        self.num_arrival_moms = num_arrival_moms
        self.num_ser_moms = num_ser_moms
        self.num_depart_corrs = num_depart_corrs
        self.df = df

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x, y = pkl.load(open(self.data_paths[index], 'rb'))

        y = y[:, :self.num_depart_corrs]

        x1 = x[:, :self.num_arrival_moms]
        x2 = x[:, 5: 5 + self.num_ser_moms]
        x = np.concatenate((x1, x2), axis=1)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, y


class Net_depart_0_moms(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 70)
        self.fc3 = nn.Linear(70, 50)
        self.fc4 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Net_depart_0_corrs(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 70)
        self.fc3 = nn.Linear(70, 50)
        self.fc4 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Net_steady_1(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 70)
        self.fc3 = nn.Linear(70, 200)
        self.fc4 = nn.Linear(200, 350)
        self.fc5 = nn.Linear(350, 600)
        self.fc6 = nn.Linear(600, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class Net_depart_1_moms(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 70)
        self.fc3 = nn.Linear(70, 50)
        self.fc4 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Net_depart_1_corrs(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 70)
        self.fc3 = nn.Linear(70, 50)
        self.fc4 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



model_path_depart_0_moms = r'..\models\depart_0_moms'
model_path_depart_0_cors = r'..\models\depart_0_corrs'

model_path_depart_1_moms = r'..\models\depart_1_moms'
model_path_depart_1_cors = r'..\models\depart_1_cors'

model_path_steady_1 = r'..\models\steady_1'

models_depart_0_moms =  os.listdir(model_path_depart_0_moms)
models_depart_0_cors =  os.listdir(model_path_depart_0_cors)

models_depart_1_moms =  os.listdir(model_path_depart_1_moms)
models_depart_1_cors =  os.listdir(model_path_depart_1_cors)

models_steady_1 =  os.listdir(model_path_steady_1)

full_path_depart_0_moms = os.path.join(model_path_depart_0_moms, models_depart_0_moms[0])
full_path_depart_0_cors = os.path.join(model_path_depart_0_cors, models_depart_0_cors[0])

full_path_depart_1_moms = os.path.join(model_path_depart_1_moms, models_depart_1_moms[0])
full_path_depart_1_cors = os.path.join(model_path_depart_1_cors, models_depart_1_cors[0])

full_path_steady_1 = os.path.join(model_path_steady_1, models_steady_1[0])


# print(full_path_depart_0_moms, full_path_depart_0_cors)
# print(full_path_depart_1_moms, full_path_depart_1_cors)

# print(full_path_steady_1)

input_size_depart_0_moms = 10
output_size_depart_0_moms = 5
net_depart_0_moms = Net_depart_0_moms(input_size_depart_0_moms, output_size_depart_0_moms).to(device)
net_depart_0_moms.load_state_dict(torch.load(full_path_depart_0_moms))
# print(net_depart_0_moms)


input_size_depart_0_corrs = 10
output_size_depart_0_corrs = 8
net_depart_0_cors = Net_depart_0_corrs(input_size_depart_0_corrs, output_size_depart_0_corrs).to(device)
net_depart_0_cors.load_state_dict(torch.load(full_path_depart_0_cors))

# print(net_depart_0_cors)

input_size_steady_1 = 18
output_size_steady_1 = 1499
net_steady_1 = Net_steady_1(input_size_steady_1, output_size_steady_1).to(device)
net_steady_1.load_state_dict(torch.load(full_path_steady_1))

# print(net_steady_1)

input_size_depart_1_moms = 18
output_size_depart_1_moms = 5
net_depart_1_moms = Net_depart_1_moms(input_size_depart_1_moms, output_size_depart_1_moms).to(device)
net_depart_1_moms.load_state_dict(torch.load(full_path_depart_1_moms))

# print(net_depart_1_moms)

input_size_depart_1_corrs = 18
output_size_depart_1_corrs = 8
net_depart_1_corrs = Net_depart_1_corrs(input_size_depart_1_corrs, output_size_depart_1_corrs).to(device)
net_depart_1_corrs.load_state_dict(torch.load(full_path_depart_1_cors))

# print(net_depart_1_corrs)

example_ind = -1


##########################################
################# NN1  ###################
##########################################

# inp_pkl_depart_0 = r'../data/depart_0/depart_0_testset2'
inp_pkl_depart_0 = r'../data/depart_0_examples'
files_depart_0 = os.listdir(inp_pkl_depart_0)
# print(files_depart_0[example_ind])


input_depart_0, out_depart_0 = pkl.load(open(os.path.join(inp_pkl_depart_0,files_depart_0[example_ind]), 'rb'))
input_depart_0 = torch.tensor(input_depart_0)
input_depart_0 = input_depart_0.reshape(1,-1)
out_depart_0 = torch.tensor(out_depart_0)


input_depart_0 = input_depart_0.float()
input_depart_0 = input_depart_0.to(device)

moms_depart_0 = net_depart_0_moms(input_depart_0)
moms_depart_0.shape

corrs_depart_0 = net_depart_0_cors(input_depart_0)
corrs_depart_0.shape

##########################################
################# NN2  ###################
##########################################
# inp_pkl_steady_1 = r'../data/steady_1/steady_1_testset2'
inp_pkl_steady_1 = r'../data/steady_1_examples'

files_steady_1 = os.listdir(inp_pkl_steady_1)
# print(files_steady_1[example_ind])

num_arrival_moms = 5
num_ser_moms = 5
input_steady_1, out_steady_1 = pkl.load(open(os.path.join(inp_pkl_steady_1, files_steady_1[example_ind]), 'rb'))
cor_inds =[10, 11, 15, 16, 35, 36, 40, 41]


input_steady_1 = torch.tensor(input_steady_1)
input_steady_1 = input_steady_1.reshape(1,-1)
out_steady_1 = torch.tensor(out_steady_1)

m = nn.Softmax(dim=1)
input_steady_1 = input_steady_1.float()
input_steady_1 = input_steady_1.to(device)
x1 = input_steady_1[:, :num_arrival_moms]
x2 = input_steady_1[:,cor_inds]
x3 = input_steady_1[:,-num_ser_moms:]
input_steady_1 = torch.concatenate((x1,x2,x3), axis = 1)
probs_steady_1 = net_steady_1(input_steady_1)
normalizing_const = torch.exp(input_steady_1[0,-5])
probs_steady_1 = m(probs_steady_1)
probs_steady_1  = probs_steady_1*normalizing_const

probs_steady_1 = probs_steady_1.to('cpu')
probs_steady_1 = torch.concatenate((torch.tensor([[1-normalizing_const]]),probs_steady_1[0:1,:]), axis = 1)
probs_steady_1.shape


##########################################
################# NN3  ###################
##########################################

# inp_pkl_depart_1 = r'../data/depart_1/depart_1_testset2'
inp_pkl_depart_1 = r'../data/depart_1_examples'

files_depart_1 = os.listdir(inp_pkl_depart_1)
# print(files_depart_1[example_ind])
input_depart_1, out_depart_1 = pkl.load(open(os.path.join(inp_pkl_depart_1, files_depart_1[example_ind]), 'rb'))

input_depart_1 = torch.tensor(input_depart_1)
input_depart_1 = input_depart_1.reshape(1,-1)
x1 = input_depart_1[:, :num_arrival_moms]
x2 = input_depart_1[:,cor_inds]
x3 = input_depart_1[:,-num_ser_moms: ]
input_depart_1 = torch.concatenate((x1,x2,x3), axis = 1)


out_depart_1 = torch.tensor(out_depart_1)

input_depart_1 = input_depart_1.float()
input_depart_1 = input_depart_1.to(device)

moms_depart_1 = net_depart_1_moms(input_depart_1)
moms_depart_1.shape

corrs_depart_1 = net_depart_1_corrs(input_depart_1)
corrs_depart_1.shape

###################################
#### prints #######################
###################################




## Printing NN2 results
with torch.no_grad():
    fig, (ax1) = plt.subplots(1, 1, figsize=(11, 3.5))
    width = 0.25
    num_probs_presenet = 20
    max_probs = num_probs_presenet
    rects1 = ax1.bar(1.5*width+np.arange(max_probs), probs_steady_1[0,:num_probs_presenet].cpu(), width, label='NN')
    rects2 = ax1.bar(np.arange(max_probs) , out_steady_1[:num_probs_presenet].cpu(), width, label='Label')
    plt.rcParams['font.size'] = '20'

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel('PMF', fontsize=21)
    ax1.set_xlabel('Number of customers', fontsize=20)
    ax1.set_title( 'Presenting L distribution' , fontsize=21, fontweight="bold")
    ax1.set_xticks(np.linspace(0,num_probs_presenet,num_probs_presenet+1).astype(int))
    ax1.set_xticklabels(np.linspace(0,num_probs_presenet,num_probs_presenet+1).astype(int), fontsize=19)
    ax1.legend(fontsize=22)
    plt.title('Steady-state probabilites' , fontsize=22)
    plt.show()

print('stop')


###############################
####### Print results #########
###############################
print('The predicted 5 moments from a GI/GI/1 queue' )
print(torch.exp(moms_depart_0))
print('The true 5 moments from a GI/GI/1 queue' )
print( torch.exp(out_depart_0[:5]))
print('####################################################')

print('The predicted 5 moments from a G/GI/1 queue' )
print(torch.exp(moms_depart_1))
print('The true 5 moments from a GI/GI/1 queue' )
print(torch.exp(out_depart_1[:5]))
print('####################################################')

print('The predicted auto-correlations from a GI/GI/1 queue' )
print(corrs_depart_0)
print('The true auto-correlations a GI/GI/1 queue' )
print( out_depart_0[cor_inds])
print('####################################################')

print('The predicted auto-correlations from a G/GI/1 queue' )
print(corrs_depart_1)
print('The true auto-correlations a G/GI/1 queue' )
print( out_depart_1[cor_inds])





