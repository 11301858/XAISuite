#An example of how to use a skorch wrapper to integrate PyTorch models
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier

class NeuralNet(nn.Module):
    def __init__(self, num_units=10, dropout=0.1):
        super(NeuralNet, self).__init__()
        self.num_units = num_units
        self.linear_1 = nn.Linear(13, num_units)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_units, 10)
        self.linear_3 = nn.Linear(10, 3)

    def forward(self, x):

        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        x = F.relu(x)
        x = self.linear_3(x)
        x = F.softmax(x, dim=-1)

        return x

nn = NeuralNetClassifier(NeuralNet, max_epochs=10, lr=0.01, batch_size=12, optimizer=optim.RMSprop)
      
