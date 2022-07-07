import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)


def read_data(self, path):
        file = open(path, 'r')
        data = np.loadtxt(file, dtype='float', delimiter=',')
        state = []
        action = []
        target_state = []
        goal = []
        # reading raw data
        for row in data:
            state.append(np.array(row[:6]))
            action.append(int(row[6]))
            target_state.append(int(row[7]))
            goal.append(row[8])
        X_train, y_train = [], []
        # sample and assemble data
        for i in range(0, len(row)- self.sample_fre * self.sample_fre):
            X_train.append([state[i], action[i], target_state[i]])
            y_train.append(goal[i])
        return np.asarray(X_train), np.asarray(y_train)

