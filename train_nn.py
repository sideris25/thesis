import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import os

current_path = os.getcwd()
os.chdir(current_path+'/nn_dataset')

X = np.load('dataset_X.npy')
Y = np.load('dataset_Y.npy')


#check cuda


#80-20 train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

#NN architecture
input_size = X_train.shape[1]
output_size = Y_train.shape[1]  

class InversionNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(InversionNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)
    
