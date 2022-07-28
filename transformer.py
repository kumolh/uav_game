import torch.optim as optim
import torch
from torch import nn, Tensor
# import torch.nn.functional as F
from torch.nn import Transformer
import numpy as np
from torch.utils.data import DataLoader
from Dataset import read_data

class TransformerModel(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, output_size):
        super(TransformerModel, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.model = Transformer(d_model=input_size, nhead=input_size, dim_feedforward=hidden_size, batch_first=True)
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input):# input: batch_size, sequence length, feature dimension
        batch_size, _, _ = input.size() # batch size is 1 when testing, can not asign a fixed value
        target = torch.zeros((batch_size, 1, self.input_size)) 
        # target = torch.zeros_like(input)
        h1 = self.model(input, target) #  batch_size, sequence length, feature dimension
        h2 = self.fc(h1) # from input feature dimension to output feature dimension
        h2 = torch.squeeze(h2, 1) # remove sequence dimension, since we only need one output
        output = self.softmax(h2) 
        return output #torch.squeeze(output)


# with torch.no_grad():
#     pass
if __name__ == '__main__':
    # 4: batch size, 20: sequence length, 7: feature dimension, 5: output dimension
    transformer = TransformerModel(4, 7, 512, 5)
    # X_train, y_train = read_data()
    # print(np.shape(X_train))
    # X_train = torch.from_numpy(X_train).float()
    # y_train = torch.from_numpy(y_train).float()
    train_data = read_data() #CustomDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    # out = transformer(X_train)[:, -1, :] #X_train[0]
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(transformer.parameters(), lr=0.003)   
    num_epochs = 1
    for epoch in range(num_epochs):  #
        correct = 0
        for i, (X, y) in enumerate(train_loader):
            # Step 1. Clear former gradients.
            transformer.zero_grad()
            # Step 2. Run forward pass.
            y_predict = transformer(X)
            # Step 3. Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = loss_function(y_predict, y)
            loss.backward()
            optimizer.step()
            pred_label = torch.argmax(y_predict, dim=0)
            label = torch.argmax(y, dim=0)
            correct += int((pred_label == label).float().sum())
        accuracy = 100 * correct / len(train_data)
        print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1, num_epochs, float(loss), accuracy))
    torch.save(transformer, 'transformer_model.pt')
    X_test = train_data[0][0]
    X_test = X_test[None, :] # model needs a dummy dimension(as batch size)
    y_pre = transformer(X_test)
    print(y_pre)