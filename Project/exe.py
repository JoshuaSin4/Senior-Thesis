from source import SNN, SurrGradSpike
import numpy as np
import torch
import matplotlib.pyplot as plt


dtype = torch.float
device = torch.device("cpu")

time_step = 1e-3
tau_mem = 10e-3
tau_syn = 5e-3

wparams = {}
wparams['nb_steps']  = 200
wparams['nb_inputs']  = 100
wparams['nb_hidden']  = 4
wparams['nb_outputs'] = 2
wparams['batch_size'] = 256
wparams['alpha']   = float(np.exp(-time_step/tau_syn))
wparams['beta']    = float(np.exp(-time_step/tau_mem))


# Before the Surrogate Gradient 
snn100 = SNN(device, dtype, **wparams)

freq = 5 # Firing Rate
prob = freq*time_step # Probability
mask = torch.rand((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype)
x_data = torch.zeros((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype, requires_grad=False)
x_data[mask<prob] = 1.0 # Tensor filled wit spikes

y_data = torch.tensor(1*(np.random.rand(wparams['batch_size'])<0.5), device=device, dtype = torch.long)

loss_hist_true_grad = snn100.optimize_loss_function(x_data, y_data, device, dtype, **wparams)

# After the Surrogate Gradient 
snn100.spike_fn = SurrGradSpike.apply

loss_hist = snn100.optimize_loss_function(x_data, y_data, device, dtype, **wparams)

# Plotting 
fig, ax = plt.subplots(dpi=150)
ax.plot(loss_hist_true_grad, label="True gradient")
ax.plot(loss_hist, label="Surrogate gradient")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.show()