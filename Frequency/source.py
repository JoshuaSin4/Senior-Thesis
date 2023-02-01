import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn


class SNN():
    def __init__(self, spk_fn, tau_syn, tau_mem, device, dtype, **kwargs):
        
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
        torch.manual_seed(kwargs['sample'])
        torch.nn.init.normal_(self.w1, mean=0.0, std=std_w1)
        torch.manual_seed(kwargs['sample'])
        torch.nn.init.normal_(self.w2, mean=0.0, std=std_w2)
        params = [self.w1,self.w2] # The paramters we want to optimize
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9,0.999)) # The optimizer we are going to use
        return optimizer
        

    def compute_classification_accuracy(self, x_data, y_data, batch_size, shuffle, **kwargs):
        """ Computes classification accuracy on supplied data in batches. """
        accs = []
        for x_local, y_local in self.images2spike(x_data, y_data, batch_size, shuffle=True, **kwargs):
            output, _ = self.run_snn(x_local)
            spike_count =torch.sum(output,1)
            mean_firing_rate = spike_count/(kwargs['nb_steps']*kwargs['time_step'])
            _, am = torch.max(mean_firing_rate, 1)
            tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
            accs.append(tmp)
        return np.mean(accs)
    
    def images2spike(self, x, y, shuffle, device, **kwargs):  
        '''Converts images to spike trains'''
        labels_ = np.array(y,dtype=np.int64)
        number_of_batches = len(x)//kwargs['batch_size']
        sample_index = np.arange(len(x))

        if shuffle:
            np.random.shuffle(sample_index)

        total_batch_count = 0
        counter = 0

        batch_index = sample_index[kwargs['batch_size']*counter:kwargs['batch_size']*(counter+1)]

        while counter < number_of_batches:
            average_rate = torch.empty(len(x[batch_index]))
            x_batch = torch.empty((len(x[batch_index]), kwargs['nb_steps'], kwargs['nb_inputs'])).to(device)
            for i, image in enumerate(x[batch_index]):
                tensor_image = torch.Tensor(image)
                average_rate[i] = torch.mean(tensor_image/kwargs['time_step'])
                spike_train = torch.empty((kwargs['nb_steps'], kwargs['nb_inputs']))
                for t in range(kwargs['nb_steps']):
                    spike_t = torch.bernoulli(tensor_image)
                    spike_train[t] = spike_t
                x_batch[i] = spike_train
            y_batch = torch.tensor(labels_[batch_index]) .to(device)
            average_rate_of_batch = torch.mean(average_rate)
            
            yield x_batch,  y_batch, average_rate_of_batch

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
