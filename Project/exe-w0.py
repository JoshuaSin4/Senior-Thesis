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
wparams['nb_inputs']  = 1
wparams['nb_hidden']  = 1
wparams['nb_outputs'] = 0
wparams['batch_size'] = 256
wparams['alpha']   = float(np.exp(-time_step/tau_syn))
wparams['beta']    = float(np.exp(-time_step/tau_mem))    
wparams['sample'] = 10

# Input Data
freq = 5 # Firing Rate
prob = freq*time_step # Probability

mask = torch.rand((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype)
x_data = torch.zeros((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype, requires_grad=False)
x_data[mask<prob] = 1.0 # Tensor filled wit spikes
y_data = torch.tensor(1*(np.random.rand(wparams['batch_size'])<0.5), device=device, dtype = torch.long)

snn = SNN(device, dtype, **wparams)

# self.w1 = 0.1717 sample 42
# self.w1 = 0.1653 sample 1
# self.w1 = 0.1371 sample 1.8e19
# self.w1 = 0.1527 sample 1.79e19
# self.w1 = -0.1503 sample 10
snn.weight_scale = 0.25


optimizer = snn.init_train(**wparams)
out_rec,spk_rec = snn.run_snn1(x_data, device, dtype, **wparams)

print(spk_rec.sum(axis=1))

print(snn.w1)
final_time_for_now = datetime.now()
 
# printing initial_date
print ('final_date:', str(final_time_for_now))
print('Time difference:', str(final_time_for_now - ini_time_for_now))