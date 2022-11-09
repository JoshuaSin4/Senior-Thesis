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


freq = 5 # Firing Rate
prob = freq*time_step # Probability
mask = torch.rand((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype)
x_data = torch.zeros((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype, requires_grad=False)
x_data[mask<prob] = 1.0 # Tensor filled wit spikes

y_data = torch.tensor(1*(np.random.rand(wparams['batch_size'])<0.5), device=device, dtype = torch.long)

betas = np.arange(0,1,0.01)
accuracy_list = []
std_list = []

for i in betas:
    wparams['beta'] = i
    snn100 = SNN(device, dtype, **wparams)
    snn100.spike_fn = SurrGradSpike.apply
    snn100.optimize_loss_function(x_data, y_data, device, dtype, **wparams)
    accuracy = snn100.print_classification_accuracy(x_data, y_data, device, dtype, **wparams)
    std=(snn100.weight_scale)/np.sqrt(wparams['nb_inputs'])
    print(snn100.weight_scale)
    accuracy_list.append(accuracy)
    std_list.append(std)

fig, ax = plt.subplots()
ax.plot(np.array(std_list), np.array(accuracy_list) , marker ='o')
ax.set_xlabel("Standard Deviation")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs. Standard Deviation")
plt.show()