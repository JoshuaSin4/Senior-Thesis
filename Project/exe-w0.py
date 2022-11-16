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
wparams['nb_steps']  = 2
wparams['nb_inputs']  = 1
wparams['nb_hidden']  = 0
wparams['nb_outputs'] = 1
wparams['batch_size'] = 256
wparams['alpha']   = float(np.exp(-time_step/tau_syn))
wparams['beta']    = float(np.exp(-time_step/tau_mem))    
wparams['sample'] = 42

# Input Data
freq = 5 # Firing Rate
prob = freq*time_step # Probability

epoch = 1000
average_std_list = []
std_average_list = []
betas = np.arange(0,1,0.1)
nb_list = np.arange(80,401, 100)
std_mean_graph = []
std_std_graph = []

mask = torch.rand((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype)
x_data = torch.zeros((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype, requires_grad=False)
x_data[mask<prob] = 1.0 # Tensor filled wit spikes
y_data = torch.tensor(1*(np.random.rand(wparams['batch_size'])<0.5), device=device, dtype = torch.long)

snn = SNN(device, dtype, **wparams)

out_rec,_ = snn.run_snn(x_data, device, dtype, **wparams)

print(x_data.sum(axis=1))
final_time_for_now = datetime.now()
 
# printing initial_date
print ('final_date:', str(final_time_for_now))
print('Time difference:', str(final_time_for_now - ini_time_for_now))