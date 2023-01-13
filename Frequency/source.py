import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn


class SNN():
    def __init__(self, spk_fn, tau_syn, tau_mem, device, dtype, **kwargs):
        self.batch_size = kwargs['batch_size']
        
        self.alpha   = float(np.exp(-kwargs['time_step']/tau_syn))  
        self.beta  = float(np.exp(-kwargs['time_step']/tau_mem))  

        self.weight_scale = 7*(1- self.beta)# this should give us some spikes to begin with

        self.w1 = torch.empty((kwargs['nb_inputs'], kwargs['nb_hidden']),  device=device, dtype=dtype, requires_grad=True)

        self.w2 = torch.empty((kwargs['nb_hidden'], kwargs['nb_outputs']), device=device, dtype=dtype, requires_grad=True)        
    
        self.spike_fn = spk_fn
        
        
    def run_snn(self, inputs, device, dtype, **kwargs):
        h1 = torch.einsum("abc,cd->abd", (inputs, self.w1))
        syn = torch.zeros((kwargs['batch_size'],kwargs['nb_hidden']), device=device, dtype=dtype)
        mem = torch.zeros((kwargs['batch_size'],kwargs['nb_hidden']), device=device, dtype=dtype)

        mem_rec = []
        spk_rec = []

        # Compute hidden layer activity
        for t in range(kwargs['nb_steps']):
            mthr = mem-1.0
            out = self.spike_fn(mthr)
            rst = out.detach() # We do not want to backprop through the reset

            new_syn = self.alpha*syn +h1[:,t]
            new_mem = (self.beta*mem +syn)*(1.0-rst)

            mem_rec.append(mem)
            spk_rec.append(out)
            
            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec,dim=1)
        spk_rec = torch.stack(spk_rec,dim=1)

        # Readout layer
        h2= torch.einsum("abc,cd->abd", (spk_rec, self.w2))
        flt = torch.zeros((kwargs['batch_size'],kwargs['nb_outputs']), device=device, dtype=dtype)
        out = torch.zeros((kwargs['batch_size'],kwargs['nb_outputs']), device=device, dtype=dtype)
        out_rec = [out]
        for t in range(kwargs['nb_steps']):
            new_flt = self.alpha*flt +h2[:,t]
            new_out = self.beta*out +flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec,dim=1)
        other_recs = [mem_rec, spk_rec]
        return out_rec, other_recs
    
    def init_train(self, std_w1, std_w2, lr, **kwargs):
        torch.nn.init.normal_(self.w1, mean=0.0, std=std_w1)
        torch.nn.init.normal_(self.w2, mean=0.0, std=std_w2)
        params = [self.w1,self.w2] # The paramters we want to optimize
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9,0.999)) # The optimizer we are going to use
        return optimizer
        

    def compute_classification_accuracy(self, x_data, y_data, device, dtype, shuffle, **kwargs):
        """ Computes classification accuracy on supplied data in batches. """
        accs = []
        for x_local, y_local in self.sparse_data_generator(x_data, y_data, device, shuffle, **kwargs):
            output,_ = self.run_snn(x_local.to_dense(), device, dtype, **kwargs)
            m,_= torch.max(output,1) # max over time
            _,am=torch.max(m,1)      # argmax over output units
            tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
            accs.append(tmp)
        return np.mean(accs)
    
    def current2firing_time(self, x, tau, thr=0.2, tmax=1.0, epsilon=1e-7):
        """ Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

        Args:
        x -- The "current" values

        Keyword args:
        tau -- The membrane time constant of the LIF neuron to be charged
        thr -- The firing threshold value 
        tmax -- The maximum time returned 
        epsilon -- A generic (small) epsilon > 0

        Returns:
        Time to first spike for each "current" x
        """
        idx = x<thr
        x = np.clip(x,thr+epsilon,1e9)
        T = tau*np.log(x/(x-thr))
        T[idx] = tmax
        return T
 

    def sparse_data_generator(self, X, y, device, shuffle, **kwargs):
        """ This generator takes datasets in analog format and generates spiking network input as sparse tensors. 

        Args:
            X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
            y: The labels
        """

        labels_ = np.array(y,dtype=np.int64)
        number_of_batches = len(X)//kwargs['batch_size']
        sample_index = np.arange(len(X))

        # compute discrete firing times
        tau_eff = 20e-3/kwargs['time_step']
        firing_times = np.array(self.current2firing_time(X, tau_eff, tmax=kwargs['nb_steps']), dtype=np.int64)
        unit_numbers = np.arange(kwargs['nb_units'])

        if shuffle:
            np.random.shuffle(sample_index)

        total_batch_count = 0
        counter = 0
        while counter<number_of_batches:
            batch_index = sample_index[kwargs['batch_size']*counter:kwargs['batch_size']*(counter+1)]

            coo = [ [] for i in range(3) ]
            for bc,idx in enumerate(batch_index):
                c = firing_times[idx]<kwargs['nb_steps']
                times, units = firing_times[idx][c], unit_numbers[c]

                batch = [bc for _ in range(len(times))]
                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo).to(device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
        
            X_batch = torch.sparse.FloatTensor(i, v, torch.Size((kwargs['batch_size'],kwargs['nb_steps'],kwargs['nb_units']))).to(device)
            y_batch = torch.tensor(labels_[batch_index],device=device)

            yield X_batch.to(device=device), y_batch.to(device=device)

            counter += 1

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
