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
device = torch.device("cuda")

time_step = 1e-3
tau_syn = 5e-3

# Dictionary of Weight Parameters
wparams = {}
wparams['nb_steps']  = 200
wparams['nb_inputs']  = 150
wparams['nb_hidden']  = 4
wparams['nb_outputs'] = 2
wparams['batch_size'] = 256
wparams['alpha']   = float(np.exp(-time_step/tau_syn))  


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
tau_mems = np.arange(1e-4,  1e-2, 2e-4)
sample_list = np.arange(50,150, int(sys.argv[1]))

# Main Loop
for sample in sample_list:
    wparams['sample'] = sample
    acc_list = []
    std_w1_list = []
    std_w2_list = []
    
    for tau in tau_mems: 
        wparams['beta'] = float(np.exp(-time_step/tau))  
        # Instantiating SNN model and Using Surrogate Gradients
        snn = SNN(spike_fn,device, dtype, **wparams)
        optimizer = snn.init_train(**wparams)
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
        acc_list.append(accuracy)
        std_w1_list.append(snn.weight_scale/np.sqrt(wparams['nb_inputs']))
        std_w2_list.append(snn.weight_scale/np.sqrt(wparams['nb_hidden']))

    data={}
    sample_acc_list=[]
    sample_std_w1_list=[]
    sample_std_w2_list = []
    sample_list = []
    nb_inputs = []
    data['acc_list']=sample_acc_list
    data['std_w1_list']=sample_std_w1_list
    data['std_w2_list']=sample_std_w2_list
    data['sample_list']=sample_list
    data['nb_list'] = nb_inputs

    data['acc_list']=np.append(acc_list, data['acc_list'])
    data["std_w1_list"]=np.append(std_w1_list, data["std_w1_list"])
    data["std_w2_list"]=np.append(std_w2_list, data["std_w2_list"])
    data["sample_list"]=np.append(wparams['sample'] ,data["sample_list"])
    data["nb_inputs"]=np.append(wparams['nb_inputs'] ,data["nb_list"])

    np.savez("input{}sample{}.npz".format(wparams['nb_inputs'],sample),**data)
    
'''
data = np.load("result50sample50.npz")
data = dict(data)
data["acc_list"] =np.append(np.max(acc_list),data["acc_list"])
data["std_w1_list"]=np.append(best_std_w1, data["std_w1_list"])
data["std_w2_list"]=np.append(best_std_w2, data["std_w2_list"])
data["sample_list"]=np.append(wparams['sample'] ,data["sample_list"])
np.savez("result200",**data)
'''
final_time_for_now = datetime.now()
 
# printing initial_date
print ('final_date:', str(final_time_for_now))
print('Time difference:', str(final_time_for_now - ini_time_for_now))