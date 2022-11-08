import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn


class SNN():
    def __init__(self, device, dtype, **kwargs):
        self.batch_size = kwargs['batch_size']
        
        self.weight_scale = 7*(1.0-kwargs['beta']) # this should give us some spikes to begin with
        
        self.w1 = torch.empty((kwargs['nb_inputs'], kwargs['nb_hidden']),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.w1, mean=0.0, std=self.weight_scale/np.sqrt(kwargs['nb_inputs']))

        self.w2 = torch.empty((kwargs['nb_hidden'], kwargs['nb_outputs']), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.w2, mean=0.0, std=self.weight_scale/np.sqrt(kwargs['nb_hidden']))

    def spike_fn(self, x):
        out = torch.zeros_like(x)
        out[x > 0] = 1.0
        return out

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

            new_syn = kwargs['alpha']*syn +h1[:,t]
            new_mem = (kwargs['beta']*mem +syn)*(1.0-rst)

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
            new_flt = kwargs['alpha']*flt +h2[:,t]
            new_out = kwargs['beta']*out +flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec,dim=1)
        other_recs = [mem_rec, spk_rec]
        return out_rec, other_recs
    
    def optimize_loss_function(self,inputs, target, device, dtype, **kwargs):
        params = [self.w1,self.w2] # The paramters we want to optimize
        optimizer = torch.optim.Adam(params, lr=2e-3, betas=(0.9,0.999)) # The optimizer we are going to use
        log_softmax_fn = nn.LogSoftmax(dim=1) # The log softmax function across output units
        loss_fn = nn.NLLLoss() # The negative log likelihood loss function
        # The optimization loop
        loss_hist = []
        for e in range(1000):
            # run the network and get output
            output,_ = self.run_snn(inputs, device, dtype, **kwargs) 
            # compute the loss
            m,_=torch.max(output,1) # maximum of the potential
            log_p_y = log_softmax_fn(m) # softmax
            loss_val = loss_fn(log_p_y, target)

            # update the weights
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            # store loss value
            loss_hist.append(loss_val.item())

    def plot_voltage_traces(self, mem, spk=None, dim=(3,5), spike_height=5):
        gs=GridSpec(*dim)
        if spk is not None:
            dat = 1.0*mem
            dat[spk>0.0] = spike_height
            dat = dat.detach().cpu().numpy()
        else:
            dat = mem.detach().cpu().numpy()
        for i in range(np.prod(dim)):
            if i==0: a0=ax=plt.subplot(gs[i])
            else: ax=plt.subplot(gs[i],sharey=a0)
            ax.plot(dat[i])
            ax.axis("off")

    def print_classification_accuracy(self, inputs, target,device,dtype, **kwargs):
        output,_ = self.run_snn(inputs, device, dtype, **kwargs)
        m,_= torch.max(output,1) # max over time
        _,am=torch.max(m,1) # argmax over output units
        acc = np.mean((target==am).detach().cpu().numpy()) # compare to labels
        print("Accuracy %.3f"%acc)

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
    
