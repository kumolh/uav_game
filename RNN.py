import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from Dataset import read_data
from torchinfo import summary

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
    rnn = RNN(7, 64, 3, 5)
    # print(rnn)
    print(summary(rnn))
    # out = rnn(rand)[:, -1, :] # (5, 20, 4) 5: num of sample/ batch size; 20, each operation output; 4: possibility distributions of 4 outcomes
    # print(out)


    # train_data = read_data() #CustomDataset(X_train, y_train)
    # train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(rnn.parameters(), lr=0.003)   
    # num_epochs = 10
    # for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    #     correct = 0
    #     for X, y in train_loader:
    #         # Step 1. remove gradients.
    #         rnn.zero_grad()
    #         # Step 2. Run forward pass.
    #         y_predict = rnn(X)
    #         # Step 3. Compute the loss, gradients, and update the parameters by calling optimizer.step()
    #         loss = loss_function(y_predict, y)
    #         loss.backward()
    #         optimizer.step()
    #         pred_label = torch.argmax(y_predict, dim=0)
    #         label = torch.argmax(y, dim=0)
    #         correct += int((pred_label == label).float().sum())
    #     accuracy = 100 * correct / len(train_data)
    #     print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1, num_epochs, float(loss), accuracy))
    # torch.save(rnn, 'rnn_model.pt')
    # X_test = train_data[0][0]
    # X_test = X_test[None, :] # model needs a dummy dimension(as batch size)
    # y_pre = rnn(X_test)
    # print(y_pre)