import config
from ext import pickle_save, pickle_load
from data import hann, ihann
from model import FF, FFS, FFT, make_model, prop_Flayer
from model import sequence_loss

from torch import tensor, Tensor, cat, stack
from torch import zeros, ones, eye, randn
from torch import sin, cos, acos, arange
from torch import conv1d, conv_transpose1d, transpose
from torch import sigmoid, tanh, relu, softmax
from torch import pow, log, exp, sqrt, norm, mean, abs
from torch import float32, no_grad

from torch.nn.init import xavier_normal_

from numpy import pi


##


def prop_model(model, io):

    for layer in model:
        io = prop_Flayer(layer,io)
        # dropout(out, inplace=True)

    return io


def make_model_higher():

    w_conv = randn(config.frame_out,config.frame_len, requires_grad=config.conv_deconv_grad)
    if config.init_fourier:
        with no_grad():
            for f in range(config.frame_out):
                w_conv[f,...] = cos(2*pi * (f+1)/config.frame_len * arange(0,config.frame_len,1))
        convolver = FF(w_conv)
    else:
        if config.init_xavier:
            xavier_normal_(w_conv, gain=5/3)
        convolver = FFT(w_conv)

    if config.conv_deconv_same:
        deconvolver = convolver
    else:
        if config.init_fourier:
            w_deconv = w_conv.detach()
            w_deconv.requires_grad = config.conv_deconv_grad
            deconvolver = FF(w_deconv)
        else:
            w_deconv = randn(config.frame_out, config.frame_len, requires_grad=config.conv_deconv_grad)
            if config.init_xavier:
                xavier_normal_(w_deconv, gain=5/3)
            deconvolver = FFT(w_deconv)

    convolver = [convolver]
    deconvolver = [deconvolver]

    body = config.creation_info[1:-1]
    enc = make_model([config.timestep_size*2] +body+ [1])
    dec = make_model([config.timestep_size] +body+ [config.timestep_size])

    return [convolver, enc,dec, deconvolver]


##


def respond_to(model, sequences, training_run=True, extra_steps=0):

    responses = [[] for _ in range(len(sequences))]
    loss = 0

    convolver, enc,dec, deconvolver = model

    with no_grad():
        #print(convolver[0].w.size(), hann().size())
        convolver[0].w *= hann()
    #     deconvolver[0].w *= ihann(deconvolver[0].w)

    for i,sequence in enumerate(sequences):

        #print(f'seq{i}/{len(sequences)}')

        #print('in size:',sequence.size(),'conv_w size:',convolver[0].w.unsqueeze(1).size())

        sequence = conv1d(sequence, convolver[0].w.unsqueeze(1), stride=config.frame_stride)
        sequence = transpose(sequence,1,2)
        sequence /=config.frame_len

        #print('conved size:',sequence.size())

        # make key,query from all here.. => the transformer stuff

        for t in range(sequence.size(1)-1):

            curr_inp = sequence[:,t:t+1,:]
            prev_inps = sequence[:,:t+1,:]
            lbl = sequence[:,t+1:t+2,:]

            #print(f'{t}/{sequence.size(1)}')

            # print('t:',t,',prev inps size:',prev_inps.size(),'curr inp size:',curr_inp.size())

            #todo: hmmmm..
            inp = cat([prev_inps,curr_inp.repeat(1,t+1,1)], -1)

            # if config.seq_force_ratio != 1 and t>=2:
            #     seq_force_ratio = config.seq_force_ratio**t
            #     inp *= seq_force_ratio
            #     inp +=

            #print('inp size:',inp.size())

            enced = prop_model(enc,inp)

            # print('enced size:', enced.size())

            attn_inp = (softmax(enced,1) * prev_inps).sum(1)

            # print('attnded size:', attn_inp.size())

            deced = prop_model(dec,attn_inp)

            loss += sequence_loss(lbl,deced)

            responses[-1].append(deced)
            # input("halt here")

        #input('halt here..')

    if training_run:
        loss.backward()
        return float(loss)

    else:

        #print("seq size", sequence.size(1), 'hm resps', len(responses[-1]))

        if len(sequences)==1:

            for t_extra in range(extra_steps):
                t = sequence.size(1)+t_extra-1

                #print(f't extra:{t}')

                curr_inp = responses[-1][t-1]

                # print(sequence[:,:,:].size(), stack(responses[-1][sequence.size(1)-1-1:],1).size())

                prev_inps = cat([sequence[:,:-1,:], stack(responses[-1][sequence.size(1)-1-1:],1)],1)

                inp = cat([prev_inps,curr_inp.repeat(1,t+1,1)], -1)

                #print(inp.size())

                enced = prop_model(enc, inp)

                # print('enced size:', enced.size())

                attn_inp = (softmax(enced,1) * prev_inps).sum(1)

                # print('attnded size:', attn_inp.size())

                deced = prop_model(dec, attn_inp)

                responses[-1].append(deced)

            responses = responses[-1]
            responses = [(deconvolver[0].w * resp).sum(1) for resp in responses]
            responses = [ihann(resp) for resp in responses]

            responses = [] # todo: stitch together responses here..

        return float(loss), responses
