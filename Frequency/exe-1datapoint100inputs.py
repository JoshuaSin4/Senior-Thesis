from source import SNN, SurrGradSpike
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torchvision



ini_time_for_now = datetime.now()
 
# printing initial_date
print ('initial_date:', str(ini_time_for_now))

dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

tau_mem = 10e-3
tau_syn = 5e-3

# Dictionary of Weight Parameters
wparams = {}
wparams['nb_steps']  = 100
wparams['nb_inputs']  = 28*28
wparams['nb_units']  = 28*28
wparams['nb_hidden']  = 100
wparams['nb_outputs'] = 10
wparams['batch_size'] = 256
wparams['time_step'] = 1e-3


# Input Data
freq = 5 # Firing Rate
prob = freq*wparams['time_step'] # Probability

# Here we load the Dataset
root = os.path.expanduser("~/data/datasets/torch/fashion-mnist")
train_dataset = torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=None, target_transform=None, download=True)

# Standardize data
# x_train = torch.tensor(train_dataset.train_data, device=device, dtype=dtype)
x_train = np.array(train_dataset.data, dtype=float)
x_train = x_train.reshape(x_train.shape[0],-1)/255
# x_test = torch.tensor(test_dataset.test_data, device=device, dtype=dtype)
x_test = np.array(test_dataset.data, dtype=float)
x_test = x_test.reshape(x_test.shape[0],-1)/255

# y_train = torch.tensor(train_dataset.train_labels, device=device, dtype=dtype)
# y_test  = torch.tensor(test_dataset.test_labels, device=device, dtype=dtype)
y_train = np.array(train_dataset.targets, dtype=np.int64)
y_test  = np.array(test_dataset.targets, dtype=np.int64)
spike_fn = SurrGradSpike.apply

# List of Items for Loop
nb_epochs = 30
axis_std_w1 = np.arange(0.006, 0.02, 0.001)
axis_std_w2 = np.arange(0.006, 0.02, 0.001)
grid_w1_w2 = np.meshgrid(axis_std_w1, axis_std_w2)

lr = 5e-4

# Main Loop

train_accuracy_matrix_w1_w2 = np.zeros((len(axis_std_w1),len(axis_std_w2)))
test_accuracy_matrix_w1_w2 = np.zeros((len(axis_std_w1),len(axis_std_w2)))
for (i,std_w1) in zip(range(len(axis_std_w1)),axis_std_w1):

    for (j,std_w2) in zip(range(len(axis_std_w2)), axis_std_w2):
        # Instantiating SNN model and Using Surrogate Gradients
        snn = SNN(spike_fn, tau_syn, tau_mem,device, dtype, **wparams)
        optimizer = snn.init_train(std_w1,std_w2, lr=lr,**wparams)
        log_softmax_fn = nn.LogSoftmax(dim=1) # The log softmax function across output units
        loss_fn = nn.NLLLoss() # The negative log likelihood loss function
        
        loss_hist = []
        for e in range(nb_epochs):
            local_loss = []
            for x_local, y_local in snn.sparse_data_generator(x_train, y_train, device, shuffle=True, **wparams):
                output,recs = snn.run_snn(x_local.to_dense(), device, dtype, **wparams)
                _,spks=recs
                m,_=torch.max(output,1)
                log_p_y = log_softmax_fn(m)
                
                # Here we set up our regularizer loss
                # The strength paramters here are merely a guess and there should be ample room for improvement by
                # tuning these paramters.
                reg_loss = 1e-7*torch.sum(spks) # L1 loss on total number of spikes
                reg_loss += 1e-8*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron
                
                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local) + reg_loss

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())
            mean_loss = np.mean(local_loss)
            print("Epoch %i: loss=%.5f"%(e+1,mean_loss))
            loss_hist.append(mean_loss)
            
                
            train_accuracy = snn.compute_classification_accuracy(x_train, y_train, device, dtype, shuffle=True, **wparams)
            test_accuracy = snn.compute_classification_accuracy(x_test, y_test, device, dtype, shuffle=True, **wparams)

            train_accuracy_matrix_w1_w2[i][j] = train_accuracy
            test_accuracy_matrix_w1_w2[i][j] = test_accuracy
            
        data = {}
        data['grid_w1_w2'] = grid_w1_w2
        data['train_accuracy_w1_w2'] = train_accuracy_matrix_w1_w2
        data['test_accuracy_w1_w2'] = test_accuracy_matrix_w1_w2
        np.savez("std_w1{}std_w2{}.npz".format(std_w1,std_w2),**data)

final_time_for_now = datetime.now()
 
# printing initial_date
print ('final_date:', str(final_time_for_now))
print('Time difference:', str(final_time_for_now - ini_time_for_now))