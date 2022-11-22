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
wparams['batch_size'] = 1
wparams['alpha']   = float(np.exp(-time_step/tau_syn))
wparams['beta']    = float(np.exp(-time_step/tau_mem))    
wparams['sample'] = 1
# Input Data
freq = 5 # Firing Rate
prob = freq*time_step # Probability

mask = torch.rand((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype)
x_data = torch.zeros((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype, requires_grad=False)
x_data[0,10,0] = 1.0

snn = SNN(device, dtype, **wparams)

snn.w1 = snn.w1 + float(sys.argv[1])

#optimizer = snn.init_train(**wparams)
out_rec,spk_rec = snn.run_snn1(x_data, device, dtype, **wparams)

print("Spike Recording")
print(spk_rec.sum(axis=1))
print("Output Recording")
print(out_rec.sum(axis=1))

final_time_for_now = datetime.now()
 
# printing initial_date
print ('final_date:', str(final_time_for_now))
print('Time difference:', str(final_time_for_now - ini_time_for_now))