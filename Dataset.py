from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)
    
def read_data(y_type='float'):
    X_train, y_train = [], []
    for idx in range(4):
        for num in range(1):
            file_name = 'raw_data/goal' + str(idx) + '-' + str(num) + '.csv'
            file = open(file_name, 'r')
            data = np.loadtxt(file, dtype='float', delimiter=', ')
            states_action = []
            goal = []
            # reading raw data
            for row in data:
                states_action.append(row[:7])
                g = [0] * 4
                g[int(row[7])] = 1
                goal.append(g)
            # sample and assemble data
            for i in range(0, len(goal)- 20):
                X_train.append(np.array(states_action[i:i+20]))
                y_train.append(goal[i])
        # print('read file: {}'.format(num) 
    X, y = np.array(X_train), np.array(y_train)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long() if y_type == 'long' else torch.from_numpy(y).float()
    return CustomDataset(X, y)