import torch
import torch.nn as nn
torch.set_num_threads(1)
from source import SNN, SurrGradSpike
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torchvision
from multiprocessing import Pool
from functools import partial
import itertools

ini_time_for_now = datetime.now()

print ('initial_date:', str(ini_time_for_now))         

dtype = torch.float

# Check whether a GPU is available
device = torch.device("cpu")     

tau_mem = 10e-3
tau_syn = 5e-3

# Dictionary of Weight Parameters
wparams = {}
wparams['nb_steps']  = 100
wparams['nb_inputs']  = 28*28
wparams['nb_units']  = 28*28
wparams['nb_hidden']  = 500
wparams['nb_outputs'] = 10
wparams['batch_size'] = 256
wparams['time_step'] = 2e-3


# Input Data
freq = 5 # Firing Rate
prob = freq*wparams['time_step'] # Probability
spike_fn = SurrGradSpike.apply


# Here we load the Dataset
root = os.path.expanduser("~/data/datasets/torch/mnist")
train_dataset = torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.MNIST(root, train=False, transform=None, target_transform=None, download=True)

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


# List of Items for Loop
nb_epochs = 5
axis_std_w1 = np.arange(0.005, 0.1, 0.015)
axis_std_w2 = np.arange(0.005, 0.1, 0.015)
grid_w1_w2 = np.meshgrid(axis_std_w1, axis_std_w2)

lr = 2e-4
scale = 1


def train(sample, std_w1, std_w2, spike_fn=spike_fn, tau_syn=tau_syn, tau_mem=tau_mem,device=device, dtype=dtype, **wparams):
    shuffle2=True
    wparams['sample'] = sample
    
    # Instantiating SNN model and Using Surrogate Gradients
    snn = SNN(spike_fn, tau_syn, tau_mem,device, dtype, **wparams)
    optimizer = snn.init_train(std_w1,std_w2, lr=lr,**wparams)
    log_softmax_fn = nn.LogSoftmax(dim=1) # The log softmax function across output units
    loss_fn = nn.NLLLoss() # The negative log likelihood loss function

    loss_hist = []
    average_frequency = []
    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local, average_rate_of_batch in snn.images2spike(x_train, y_train, shuffle=shuffle2, device=device, **wparams):
            output,recs = snn.run_snn(x_local, device, dtype, **wparams)
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
            average_frequency.append(average_rate_of_batch)

        mean_loss = np.mean(local_loss)
        print("Epoch %i: loss=%.5f"%(e+1,mean_loss))
        loss_hist.append(mean_loss)

    train_accuracy = snn.compute_classification_accuracy(x_train, y_train, device=device, shuffle=False, dtype=dtype,**wparams)
    test_accuracy = snn.compute_classification_accuracy(x_test, y_test, device=device, shuffle=False,  dtype=dtype, **wparams)


    return train_accuracy, test_accuracy, np.mean(average_frequency), loss_hist
# Main Loop


seed_list = np.arange(1000,1100, float(sys.argv[1]))

if __name__ == '__main__':
    with Pool(processes=60) as pool:
        map_train = partial(train, **wparams)
        product = itertools.product(seed_list, axis_std_w1,axis_std_w2)
        
        result= pool.starmap(map_train, product) 
        tuple_array = np.array(result)

        
        seed_array = np.array_split(tuple_array, len(seed_list)) #splitting array based on seed
        
        for i, seed in enumerate(seed_array):
            average_frequency_list = []
            train_accuracy_list = []
            test_accuracy_list = []
            loss_hist_list = []#looping over every splitted array based on seed
            for array_list in seed: # looping over every array in splitted array
                array = list(array_list) # converting 4 valued tuple into 4 valued lists
                train_acc = array[0]
                train_accuracy_list.append(train_acc)
                test_acc = array[1]
                test_accuracy_list.append(test_acc)
                average_frequency = np.mean(array[2])
                average_frequency_list.append(average_frequency)
                loss_hist = array[3]
                loss_hist_list.append(loss_hist)
                
            print(train_accuracy_list)
            print(test_accuracy_list)
            train_accuracy_matrix = np.array(train_accuracy_list).reshape((len(axis_std_w1),len(axis_std_w2)))
            test_accuracy_matrix = np.array(test_accuracy_list).reshape((len(axis_std_w1),len(axis_std_w2)))
                    
            data = {}
            data['grid_w1_w2'] = grid_w1_w2
            data['N'] = wparams['nb_hidden']
            data['train_accuracy_w1_w2'] = train_accuracy_matrix
            data['test_accuracy_w1_w2'] = test_accuracy_matrix
            data['average_frequency'] = average_frequency_list
            data['loss_hist'] = loss_hist
            np.savez("uniform-distribution-frequency-{}-seed-{}.npz".format(65, i),**data)

final_time_for_now = datetime.now()
print ('final_date:', str(final_time_for_now))
print('Time difference:', str(final_time_for_now - ini_time_for_now))