import os
import pprint
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tabulate
from data_iterator import iterators
from Model import Model
import csv
device = 'cuda' if torch.cuda.is_available() else 'cpu'



N = 100
train_samp = 0.8
batch_size = 64

train, valid = iterators(N, train_samp=train_samp, batch_size=batch_size)

input_dim = 2 
hidden_dim = 16
n_layers = 1
output_dim = 1
model = Model(input_dim, hidden_dim, n_layers, output_dim).to(device)


obj_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


fieldnames= [
    'epoch',
    'batch_loss',
    'batch_acc',
    'val_loss',
    'val_acc']

outfile = 'Results/logs.csv'

max_epoch = 100

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
        if epoch % 10 == 0:
            pprint.pprint(log)
            print('\n')
            # print(f"{epoch}\tbatch-loss: {train_loss:.4f}\tbatch-acc: {train_acc:.4f}\tval-loss: {val_loss:.4f}\t val-acc: {val_acc:.4f}")
