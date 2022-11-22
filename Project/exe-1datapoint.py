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
tau_mem = 10e-3
tau_syn = 5e-3

# Dictionary of Weight Parameters
wparams = {}
wparams['nb_steps']  = 200
wparams['nb_inputs']  = 100
wparams['nb_hidden']  = 4
wparams['nb_outputs'] = 2
wparams['batch_size'] = 256
wparams['alpha']   = float(np.exp(-time_step/tau_syn))
wparams['beta']    = float(np.exp(-time_step/tau_mem))    
wparams['sample'] = 42

# Input Data
freq = 5 # Firing Rate
prob = freq*time_step # Probability

# Random Seed Sample 42
np.random.seed(42)
mask = torch.rand((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype)
x_data = torch.zeros((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype, requires_grad=False)
x_data[mask<prob] = 1.0 # Tensor filled wit spikes
y_data = torch.tensor(1*(np.random.rand(wparams['batch_size'])<0.5), device=device, dtype = torch.long)



# List of Items for Loop
epoch = 1000
betas = np.arange(0, 1.0,0.05)
acc_list = []
std_w1_list = []
std_w2_list = []

# Main Loop
for beta in betas: 
    wparams['beta'] = beta
    # Instantiating SNN model and Using Surrogate Gradients
    snn = SNN(device, dtype, **wparams)
    spike_fn = SurrGradSpike.apply
    optimizer = snn.init_train(**wparams)
    log_softmax_fn = nn.LogSoftmax(dim=1) # The log softmax function across output units
    loss_fn = nn.NLLLoss() # The negative log likelihood loss function
    # Training the network
    for e in range(epoch):
        # run the network and get output
        output,_ = snn.run_snn(x_data, device, dtype, **wparams) 
        # compute the loss
        m,_=torch.max(output,1) # maximum of the potential
        log_p_y = log_softmax_fn(m) # softmax
        loss_val = loss_fn(log_p_y, y_data)

        # update the weights
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
    accuracy = snn.print_classification_accuracy(x_data, y_data, device, dtype, **wparams)
    acc_list.append(accuracy)
    std_w1_list.append(snn.weight_scale/np.sqrt(wparams['nb_inputs']))
    std_w2_list.append(snn.weight_scale/np.sqrt(wparams['nb_hidden']))

index = np.argmax(acc_list)
best_std_w1 = std_w1_list[index]
best_std_w2 = std_w2_list[index]

saving={}
sample_acc_list=[]
sample_std_w1_list=[]
sample_std_w2_list = []
saving['acc_list']=sample_acc_list
saving['std_w1_list']=sample_std_w1_list
saving['std_w2_list']=sample_std_w2_list

sample_std_w1_list.append(best_std_w1)
sample_std_w2_list.append(best_std_w2)

np.savez('result',**saving)

final_time_for_now = datetime.now()
 
# printing initial_date
print ('final_date:', str(final_time_for_now))
print('Time difference:', str(final_time_for_now - ini_time_for_now))