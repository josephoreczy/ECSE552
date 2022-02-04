import os
import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


logs = pd.read_csv('Results/logs.csv')

fig, axs = plt.subplots(1,1)
axs.set_ylabel('Accuracy')
axs.plot(logs['epoch'], logs['batch_acc'], label='Train')
axs.plot(logs['epoch'], logs['val_acc'], label='Validation')
axs.set_xlabel('Epoch')
plt.legend()
plt.show()
