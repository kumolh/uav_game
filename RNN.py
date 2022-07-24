import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from Dataset import read_data

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, input_tensor):
        h0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size)
        h1, (_, _)= self.lstm(input_tensor, (h0, c0))  
        # h1: batch_size, seq_len, hidden_size
        output = self.fc(h1)[:, -1, :]
        output = self.softmax(output) # batch_size, output_size
        return output

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)



# with torch.no_grad():
#     pass
if __name__ == '__main__':
    rand = torch.rand(1, 20, 7)
    # 5 is the number of sample, 20: sequence length, 7: feature dimension
    rnn = RNN(7, 64, 1, 4)
    # out = rnn(rand)[:, -1, :] # (5, 20, 4) 5: num of sample/ batch size; 20, each operation output; 4: possibility distributions of 4 outcomes
    # print(out)
    train_data = read_data('float') #CustomDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=0.003)   
    for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
        for X, y in train_loader:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            # print(X.size())
            rnn.zero_grad()
            # Step 2. Get our inputs ready for the network, that is, turn them into Tensors of word indices.

            # Step 3. Run our forward pass.
            y_predict = rnn(X)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(y_predict, y)
            loss.backward()
            optimizer.step()
    X_test = train_data[0][0]
    X_test = X_test[None, :] # model needs a dummy dimension(as batch size)
    y_pre = rnn(X_test)
    print(y_pre)