import torch.optim as optim
import torch
from torch import nn, Tensor
# import torch.nn.functional as F
from torch.nn import Transformer
import numpy as np
from torch.utils.data import DataLoader
from Dataset import read_data

class TransformerModel(nn.Module):
    def __init__(self, batch_size):
        super(TransformerModel, self).__init__()
        self.batch_size = batch_size
        self.model = Transformer(d_model=7, nhead=7, dim_feedforward=512, batch_first=True)
        self.fc = nn.Linear(7, 4)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input):# input: batch_size, sequence length, feature dimension
        batch_size, _, _ = input.size()
        target = torch.zeros((batch_size, 1, 7)) 
        h1 = self.model(input, target) #  batch_size, sequence length, feature dimension
        h2 = self.fc(h1) # from input feature dimension to output feature dimension
        h2 = torch.squeeze(h2, 1) # remove sequence dimension, since we only need one output
        output = self.softmax(h2) 
        return output #torch.squeeze(output)


# with torch.no_grad():
#     pass
if __name__ == '__main__':
    rand = torch.rand((4, 20, 7))
    # 4: batch size, 20: sequence length, 7: feature dimension
    transformer = TransformerModel(4)
    # out = rnn(rand)[:, -1, :] # (5, 20, 4) 5: num of sample/ batch size; 20, each operation output; 4: possibility distributions of 4 outcomes
    out = transformer(rand)
    print(out)
    # X_train, y_train = read_data()
    # print(np.shape(X_train))
    # X_train = torch.from_numpy(X_train).float()
    # y_train = torch.from_numpy(y_train).float()
    train_data = read_data() #CustomDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    # out = transformer(X_train)[:, -1, :] #X_train[0]
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(transformer.parameters(), lr=0.003)   
    for epoch in range(1):  #
        for X, y in train_loader:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            # print(X.size())
            transformer.zero_grad()
            # Step 2. Get inputs ready for the network, that is, turn them into Tensors of word indices.

            # Step 3. Run forward pass.
            y_predict = transformer(X)

            # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = loss_function(y_predict, y)
            loss.backward()
            optimizer.step()
    X_test = train_data[0][0]
    X_test = X_test[None, :] # model needs a dummy dimension(as batch size)
    y_pre = transformer(X_test)
    print(y_pre)