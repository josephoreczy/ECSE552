import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

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

    plt.scatter(x[:N, 0], x[:N, 1], label='Label=0')
    plt.scatter(x[N:, 0], x[N:, 1], marker='^',label='Label=1')
    plt.legend(loc='lower left')
    plt.savefig('Results/P2_data_preview.png')

    n_samples = x.shape[0]
    train_cutoff = int(train_samp*n_samples)

    data_pts = x[:, :-1]
    labels = x[:, -1].unsqueeze(-1)

    train = data.TensorDataset(data_pts[:train_cutoff], labels[:train_cutoff])
    valid = data.TensorDataset(data_pts[train_cutoff:], labels[train_cutoff:])

    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    valid_loader = data.DataLoader(valid, batch_size=batch_size, shuffle=False,num_workers=num_workers)

    return train_loader, valid_loader
