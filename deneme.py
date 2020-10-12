import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import conv1d

from scipy import fft, fftpack

import matplotlib.pyplot as plt

# Creating filters

d = 4096 # size of windows

def create_filters(d):
    x = np.arange(0, d, 1)
    wsin = np.empty((d,1,d), dtype=np.float32)
    wcos = np.empty((d,1,d), dtype=np.float32)
    window_mask = 1.0-1.0*np.cos(x)
    for ind in range(d):
        wsin[ind,0,:] = np.sin(2*np.pi*((ind+1)/d)*x)
        wcos[ind,0,:] = np.cos(2*np.pi*((ind+1)/d)*x)

    return wsin,wcos

wsin, wcos = create_filters(d)
wsin_var = Variable(torch.from_numpy(wsin), requires_grad=False)
wcos_var_orig = Variable(torch.from_numpy(wcos),requires_grad=False)

# Creating signal

t = np.linspace(0,1,4096)
x = np.sin(2*np.pi*100*t)+np.sin(2*np.pi*200*t)+np.random.normal(scale=5,size=(4096))

signal_input = torch.from_numpy(x.reshape(1,-1),)[:,None,:4096]

signal_input = signal_input.float()

#print(signal_input.size(), wcos_var.size())

#zx = conv1d(signal_input, wcos_var, stride=1)#.pow(2)+conv1d(signal_input, wcos_var, stride=1).pow(2)

#print(zx.size())

#zx = zx.pow(0.5)


# plt.plot(x)

from torch import conv_transpose1d, transpose
#xz_revert = conv_transpose1d(zx, transpose(wcos_var,0,2))
#print(xz_revert.size())




###
from torch import transpose



from torch.autograd import Variable
from torch.nn.functional import conv1d

from scipy.signal.windows import hann

y = np.sin(2*np.pi*50*np.linspace(0,10,2048))+np.sin(2*np.pi*20*np.linspace(0,10,2048)) + np.random.normal(scale=1,size=2048)

stride = 512

def create_filters2(window_size, out_size, low=50, high=6000):
    x = np.arange(0, window_size, 1)
    wsin = np.empty((out_size, window_size), dtype=np.float32)
    wcos = np.empty((out_size, window_size), dtype=np.float32)
    start_freq = low
    end_freq = high
    # num_cycles = start_freq*d/44000.
    # scaling_ind = np.log(end_freq/start_freq)/k

    window_mask = hann(2048, sym=False) # same as 0.5-0.5*np.cos(2*np.pi*x/(k))
    for ind in range(out_size):
        wsin[ind,:] = window_mask*np.sin(2 * np.pi * ind/out_size * x)
        wcos[ind,:] = np.cos(2 * np.pi * ind/out_size * x)

    return wsin,wcos


wsin, wcos = create_filters2(2048,2048)

wsin_var = Variable(torch.from_numpy(wsin), requires_grad=False)
wcos_var = Variable(torch.from_numpy(wcos),requires_grad=False)
# from torch import randn, ones, tensor
# wcos_var = tensor(wcos_var.view(2048,2048)) # ones(100,2048)

inp = torch.from_numpy(y).float()
inp = inp.reshape(1,-1,1)



import config
config.frame_stride = stride
from model import FF, convolve, deconvolve


print('inp size:',inp.size())
print('conv w size:', wcos_var.size())


zx = wcos_var @ inp

print('convolved size:',zx.size())


xz = wcos_var * zx

print('deconvolved size:',xz.size())

xz = xz.sum(1)

print('deconvolved summed size:',xz.size())

xz = xz/2048


#print('sums pre:', zx.sum(), conv1d(inp.view(1,1,-1),wcos_var_orig).sum())

print('sums:',inp.sum(),xz.sum())

#plt.plot(xz[0,:1025,0])



# from model import *
#
# inp = randn(2,1,6)
# c = make_convolver(4, 3)
# d = make_deconvolver(4, 3)
#
# print(inp.size())
# ced = prop_convolver(c, inp)
# print(ced.size())
# ded = prop_deconvolver(d, ced)
# print(ded.size())


