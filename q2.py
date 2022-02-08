import pandas as pd
import os
import shutil
import pprint
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

os.makedirs('Results', exist_ok=True)

def iterators(N=100, train_samp=0.8, batch_size=64, num_workers=8):

    t = torch.FloatTensor(N).uniform_(0, 2*torch.pi)
    r = torch.randn(N)
    x = torch.zeros((2*N, 3))

    x[:N, 0] = r*torch.cos(t)
    x[:N, 1] = r*torch.sin(t)
    x[:N, 2] = 0

    x[N:, 0] = (r + 5)*torch.cos(t)
    x[N:, 1] = (r + 5)*torch.sin(t)
    x[:N, 2] = 1

    idx = np.random.permutation(np.arange(len(x)))
    x = x[idx]

    train_cutoff = int(train_samp*N)

    data_pts = x[:, :-1]
    labels = x[:, -1].unsqueeze(-1)
    valid_data = x[:N]

    train = data.TensorDataset(data_pts[:train_cutoff], labels[:train_cutoff])
    valid = data.TensorDataset(data_pts[train_cutoff:], labels[train_cutoff:])

    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    valid_loader = data.DataLoader(valid, batch_size=batch_size, shuffle=False,num_workers=num_workers)

    return train_loader, valid_loader, valid_data



class Model(nn.Module):
    def __init__(self,input_dim, hidden_dim, n_layers, output_dim, 
                 activation='ReLu'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.af = F.relu
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

    def forward(self, x):
        out = self.fc1(x)
        out = self.af(out)
        out = self.fc2(out)
        return out


device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_samples = 1000
train_samp = 0.8
batch_size = 32

max_epoch = 200
lr = 1e-3
print_step = 10

input_dim = 2 
hidden_dim = 30
n_layers = 1
output_dim = 1

train, valid, v_data = iterators(N_samples, train_samp=train_samp, batch_size=batch_size)
model = Model(input_dim, hidden_dim, n_layers, output_dim).to(device)
obj_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

fieldnames= [
    'epoch',
    'batch_loss',
    'batch_acc',
    'val_loss',
    'val_acc']

outfile = 'Results/logs.csv'
if os.path.exists(outfile):
    os.remove(outfile)

with open(outfile, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for epoch in range(1, max_epoch+1):
        model.train()
        train_loss = 0
        train_acc = 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            yhat = model(x)
            loss = obj_func(yhat, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            train_acc += ((torch.sigmoid(yhat)>0.5)*1 == y).float().mean()
        train_loss = train_loss/len(train)
        train_acc = train_acc/len(train)
        
        model.eval()
        with torch.no_grad():
            yhats = []
            ys = []
            for x, y in valid:
                x, y = x.to(device), y.to(device)
                yhat = model(x)
                yhats.append(yhat)
                ys.append(y)
            
            yhats = torch.cat(yhats, axis=0)
            ys = torch.cat(ys, axis=0)
            
            val_loss = obj_func(yhats, ys).item()
            val_acc = ((torch.sigmoid(yhats)>0.5)*1 == ys).float().mean()

        log = dict(
            epoch = epoch,
            batch_loss = train_loss.item(),
            batch_acc = train_acc.item(),
            val_loss = val_loss,
            val_acc = val_acc.item()
            )
        writer.writerow(log)
        if epoch % print_step == 0:
            pprint.pprint(log)
            print('\n')

model_weights = model.fc1.weight
model_bias = model.fc1.bias

label_0_idx = np.where(v_data[:,-1] == 0)
label_1_idx = np.where(v_data[:,-1] == 1)

plt.scatter(v_data[label_1_idx][:, 0], v_data[label_1_idx][:, 1], marker='^',label='Label=1')
plt.scatter(v_data[label_0_idx][:, 0], v_data[label_0_idx][:, 1], label='Label=0')

xx = np.linspace(-10, 10)

for i, weight in enumerate(model_weights):
    w0 = weight[0].item()
    w1 = weight[1].item()
    b = model_bias[i].item()
    plt.plot(xx, -(xx*w0 + b)/w1)

plt.legend(loc='lower left')
plt.ylim(-10, 10)
plt.xlim(-10, 10)
plt.savefig('Results/validation_data_lines.png')
plt.show()
plt.clf()
plt.close()


logs = pd.read_csv('Results/logs.csv')

fig, axs = plt.subplots(1,1)
axs.set_ylabel('Accuracy')
axs.plot(logs['epoch'], logs['batch_acc'], label='Train')
axs.plot(logs['epoch'], logs['val_acc'], label='Validation')
axs.set_xlabel('Epoch')
plt.legend()
plt.savefig('Results/Accuracies.png')
plt.show()
plt.clf()
plt.close()

