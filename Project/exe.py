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

epoch = 1000
average_std_list = []
std_average_list = []
betas = np.arange(0, 1,0.2)
sample_list = int(sys.argv[1])
sampling = np.arange(0, 10*sample_list, 10)
nb_list = np.arange(80,401, 100)
std_mean_graph = []
std_std_graph = []

# Main Loop
for nb_inputs in nb_list:
    wparams['nb_inputs'] = nb_inputs
    mask = torch.rand((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype)
    x_data = torch.zeros((wparams['batch_size'],wparams['nb_steps'],wparams['nb_inputs']), device=device, dtype=dtype, requires_grad=False)
    x_data[mask<prob] = 1.0 # Tensor filled wit spikes
    y_data = torch.tensor(1*(np.random.rand(wparams['batch_size'])<0.5), device=device, dtype = torch.long)
    # Finding the mean of std. dev. for one point
    for sample in sampling:
        wparams['sample'] = sample
        snn = SNN(device, dtype, **wparams)
        snn.spike_fn = SurrGradSpike.apply
        std_list = []
        accuracy_list = []
        # Finding the best std dev for each sample
        for beta in betas: 
            wparams['beta'] = beta
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
            accuracy_list.append(accuracy)
            std_list.append(beta)

        best_std = std_list[np.argmax(accuracy_list)]
        average_std_list.append(best_std)
        
    std_mean = np.mean(average_std_list) # Average of the standard deviation of the initial weights for an average performance and for a particular number of input neurons over Multiple Samples
    std_std = np.std(average_std_list) # standard deviation of the standard deviation of the initial weights for an average performance and for a particular number of input neurons over Multiple Samples
    std_mean_graph.append(std_mean)
    std_std_graph.append(std_std)

# Plotting and Save 
fig, ax = plt.subplots()
ax.plot(nb_list, std_mean_graph, marker = "o", color = "b")
ax.errorbar(nb_list, std_mean_graph, yerr=std_std_graph, fmt="o", color="r")
ax.set_xlabel("Number of Neurons in Input Layer")
ax.set_ylabel("Standard Deviation")
ax.set_title("Standard Deviation vs. Number of Neurons in Input Layer")
np.savetxt('std_mean.csv', std_mean_graph, delimiter=',')
np.savetxt('std_std.csv', std_std_graph, delimiter=',')
fig.savefig("StdVs.NumberofNeurons.png")

final_time_for_now = datetime.now()
 
# printing initial_date
print ('final_date:', str(final_time_for_now))
print('Time difference:', str(final_time_for_now - ini_time_for_now))