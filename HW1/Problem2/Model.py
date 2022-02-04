import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,input_dim, hidden_dim, n_layers, output_dim, 
                 activation='ReLu'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.af = F.relu
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.af(self.fc1(x))
        out = self.fc2(out)
        return out
