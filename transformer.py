import torch.optim as optim
import torch
from torch import nn, Tensor
# import torch.nn.functional as F
from torch.nn import Transformer
import numpy as np
from torch.utils.data import DataLoader
from Dataset import read_data
from torchinfo import summary
# import matplotlib as mpl
import matplotlib.pyplot as plt

class TransformerModel(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, output_size):
        super(TransformerModel, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.model = Transformer(d_model=input_size, nhead=input_size, num_encoder_layers= 3, dim_feedforward=hidden_size, batch_first=True)
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input):# input: batch_size, sequence length, feature dimension
        batch_size, _, _ = input.size() # batch size is 1 when testing, can not asign a fixed value
        # batch_size = self.batch_size
        target = torch.zeros((batch_size, 1, self.input_size)) 
        # target = torch.zeros_like(input)
        h1 = self.model(input, target) #  batch_size, sequence length, feature dimension
        h2 = self.fc(h1) # from input feature dimension to output feature dimension
        h2 = torch.squeeze(h2, 1) # remove sequence dimension, since we only need one output
        output = self.softmax(h2) 
        return output #torch.squeeze(output)


# with torch.no_grad():
#     pass
def draw(y, y_label):
    plt.figure()
    x = list(range(1, 1 + len(y)))
    # mpl.use('tkagg')
    plt.plot(x, y, label=y_label)
    plt.xlabel('iterations')
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 4: batch size, 20: sequence length, 7: feature dimension, 5: output dimension
    transformer = TransformerModel(4, 7, 512, 5)
    # transformer = torch.load('transformer_model.pt')
    print(summary(transformer))
    # 

    # X_train, y_train = read_data()
    # print(np.shape(X_train))
    # X_train = torch.from_numpy(X_train).float()
    # y_train = torch.from_numpy(y_train).float()


    full_dataset = read_data(2) #CustomDataset(X_train, y_train)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True)
    # out = transformer(X_train)[:, -1, :] #X_train[0]
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(transformer.parameters(), lr=0.003)   
    # history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)
    num_epochs = 3
    train_accuracy = []
    train_loss = []
    test_accuracy = []
    test_loss = []
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
        train_accuracy.append(100 * correct / len(train_data))
        train_loss.append(loss)
        
        correct = 0
        for i, (X, y) in enumerate(test_loader):
            # Step 1. Clear former gradients.
            transformer.zero_grad()
            # Step 2. Run forward pass.
            y_predict = transformer(X)
            # Step 3. Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = loss_function(y_predict, y)
            pred_label = torch.argmax(y_predict, dim=0)
            label = torch.argmax(y, dim=0)
            correct += int((pred_label == label).float().sum())
        test_accuracy.append(100 * correct / len(test_data))
        test_loss.append(loss)
        print("Epoch {}/{}, Train_Loss: {:.3f}, Train_Accuracy: {:.3f}, Test_Loss: {:.3f}, Test_Accuracy: {:.3f}"\
            .format(epoch+1, num_epochs, float(train_loss[-1]), train_accuracy[-1], float(test_loss[-1]), test_accuracy[-1]))
        ## plot ##
        ## num of parameters ##
    draw(train_accuracy, 'accuracy')
    torch.save(transformer, 'transformer_model_1.pt')

    # X_test = train_data[0][0]
    # X_test = X_test[None, :] # model needs a dummy dimension(as batch size)
    # y_pre = transformer(X_test)
    # print(y_pre)