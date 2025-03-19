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


print(full_path_depart_0_moms, full_path_depart_0_cors)
print(full_path_depart_1_moms, full_path_depart_1_cors)

print(full_path_steady_1)

input_size_depart_0_moms = 10
output_size_depart_0_moms = 5
net_depart_0_moms = Net_depart_0_moms(input_size_depart_0_moms, output_size_depart_0_moms).to(device)
net_depart_0_moms.load_state_dict(torch.load(full_path_depart_0_moms))
print(net_depart_0_moms)


input_size_depart_0_corrs = 10
output_size_depart_0_corrs = 8
net_depart_0_cors = Net_depart_0_corrs(input_size_depart_0_corrs, output_size_depart_0_corrs).to(device)
net_depart_0_cors.load_state_dict(torch.load(full_path_depart_0_cors))

print(net_depart_0_cors)

input_size_steady_1 = 18
output_size_steady_1 = 1499
net_steady_1 = Net_steady_1(input_size_steady_1, output_size_steady_1).to(device)
net_steady_1.load_state_dict(torch.load(full_path_steady_1))

print(net_steady_1)

input_size_depart_1_moms = 18
output_size_depart_1_moms = 5
net_depart_1_mom = Net_depart_1_moms(input_size_depart_1_moms, output_size_depart_1_moms).to(device)
net_depart_1_mom.load_state_dict(torch.load(full_path_depart_1_moms))

print(net_depart_1_mom)

input_size_depart_1_corrs = 18
output_size_depart_1_corrs = 8
net_depart_1_corrs = Net_depart_1_corrs(input_size_depart_1_corrs, output_size_depart_1_corrs).to(device)
net_depart_1_corrs.load_state_dict(torch.load(full_path_depart_1_cors))

print(net_depart_1_corrs)

inp_pkl_depart_0 =
input_depart_0 = pkl.load(open(, 'rb'))

input_depart_0 = input_depart_0.float()
input_depart_0 = input_depart_0.to(device)

moms_depart_0 = net_depart_0_moms(input_depart_0)
moms_depart_0.shape

corrs_depart_0 = net_depart_0_cors(input_depart_0)
corrs_depart_0.shape
