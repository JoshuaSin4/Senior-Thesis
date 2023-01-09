from source import SNN, SurrGradSpike
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys

ini_time_for_now = datetime.now()
 
# printing initial_date
print ('initial_date:', str(ini_time_for_now))

dtype = torch.float
device = torch.device("cpu")

time_step = 1e-3
tau_mem = 1e-3
tau_syn = 5e-3

# Dictionary of Weight Parameters
wparams = {}
wparams['nb_steps']  = 200
wparams['nb_inputs']  = 250
wparams['nb_hidden']  = 4
wparams['nb_outputs'] = 2
wparams['batch_size'] = 256


# Input Data
freq = 5 # Firing Rate
prob = freq*time_step # Probability

# Random Seed Sample 
np.random.seed(int(sys.argv[1]))
mask = torch.rand((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype)
x_data = torch.zeros((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype, requires_grad=False)
x_data[mask<prob] = 1.0 # Tensor filled wit spikes

np.random.seed(int(sys.argv[1]))
y_data = torch.tensor(1*(np.random.rand(wparams['batch_size'])<0.5), device=device, dtype = torch.long)
spike_fn = SurrGradSpike.apply

# List of Items for Loop
epoch = 1000
sample_list = np.arange(50,150, int(sys.argv[1]))
axis_std_w1 = np.arange(0, 1, 0.1)
axis_std_w2 = np.arange(0, 1, 0.1)
grid_w1_w2 = np.meshgrid(axis_std_w1, axis_std_w2)

# Main Loop
for sample in sample_list:

    wparams['sample'] = sample
    accuracy_matrix_w1_w2 = np.zeros((len(axis_std_w1),len(axis_std_w2)))
    for (i,std_w1) in zip(range(len(axis_std_w1)),axis_std_w1):

        for (j,std_w2) in zip(range(len(axis_std_w2)), axis_std_w2):
            # Instantiating SNN model and Using Surrogate Gradients
            snn = SNN(spike_fn,time_step, tau_syn, tau_mem,device, dtype, **wparams)
            optimizer = snn.init_train(std_w1,std_w2,**wparams)
            log_softmax_fn = nn.LogSoftmax(dim=1) # The log softmax function across output units
            loss_fn = nn.NLLLoss() # The negative log likelihood loss function

            loss_hist = []
            for e in range(epoch):
                output,_ = snn.run_snn(x_data, device, dtype, **wparams)
                m,_=torch.max(output,1)
                log_p_y = log_softmax_fn(m)
                loss_val = loss_fn(log_p_y, y_data)

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                loss_hist.append(loss_val.item())
                
            accuracy = snn.print_classification_accuracy(x_data, y_data, device, dtype, **wparams)
            accuracy_matrix_w1_w2[i][j] = accuracy
            
    data = {}
    data['grid_w1_w2'] = grid_w1_w2
    data['accuracy_w1_w2'] = accuracy_matrix_w1_w2
    np.savez("input{}sample{}.npz".format(wparams['nb_inputs'],sample),**data)

final_time_for_now = datetime.now()
 
# printing initial_date
print ('final_date:', str(final_time_for_now))
print('Time difference:', str(final_time_for_now - ini_time_for_now))