import config
from ext import pickle_save, pickle_load

from torch import tensor, Tensor, cat, stack
from torch import zeros, ones, eye, randn
from torch import sin, cos, acos, arange
from torch import conv1d, conv_transpose1d, transpose
from torch import sigmoid, tanh, relu, softmax
from torch import pow, log, exp, sqrt, norm, mean, abs
from torch import float32, no_grad

from torch.nn.init import xavier_normal_
from torch.distributions import Normal

from collections import namedtuple
from copy import deepcopy
from math import ceil

from numpy import pi

##


FF = namedtuple('FF', 'w')
FFS = namedtuple('FFS', 'w')
FFT = namedtuple('FFT', 'w')
LSTM = namedtuple('LSTM', 'wf wk wi ws')


def make_Llayer(in_size, layer_size):

    layer = LSTM(
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
    )

    with no_grad():
        for k,v in layer._asdict().items():
            if k == 'bf':
                v += config.forget_bias

    if config.init_xavier:
        xavier_normal_(layer.wf)
        xavier_normal_(layer.wk)
        xavier_normal_(layer.ws)
        xavier_normal_(layer.wi, gain=5/3)

    return layer

def make_Flayer(in_size, layer_size, act=None):

    layer_type = FF if not act else (FFS if act=='s' else FFT)

    layer = layer_type(
        randn(in_size, layer_size, requires_grad=True, dtype=float32),
    )

    if config.init_xavier:
        if act == 's':
            xavier_normal_(layer.w)
        elif act == 't':
            xavier_normal_(layer.w, gain=5/3)

    return layer


make_layer = {
    'l': make_Llayer,
    'f': make_Flayer,
    'fs': lambda i,l: make_Flayer(i,l,act='s'),
    'ft': lambda i,l: make_Flayer(i,l,act='t'),
}


def prop_Llayer(layer, state, input):

    layer_size = layer.wf.size(1)

    prev_out = state[:,:layer_size]
    state = state[:,layer_size:]

    inp = cat([input,prev_out],dim=1)

    reset   = sigmoid(inp@layer.wf)# + layer.bf)
    write   = sigmoid(inp@layer.wk)# + layer.bk)
    context = tanh   (inp@layer.wi)# + layer.bi)
    read    = sigmoid(inp@layer.ws)# + layer.bs)

    state = reset*state + write*context
    out = read*tanh(state)

    return out, cat([out,state],dim=1)

def prop_Flayer(layer, inp):

    return inp@layer.w


prop_layer = {
    LSTM: prop_Llayer,
    FF: prop_Flayer,
    FFS: lambda l,i: sigmoid(prop_Flayer(l,i)),
    FFT: lambda l,i: tanh(prop_Flayer(l,i)),
}


def make_model():

    layer_sizes = [e for e in config.creation_info if type(e)==int]
    layer_types = [e for e in config.creation_info if type(e)==str]

    return [make_layer[layer_type](layer_sizes[i], layer_sizes[i+1]) for i,layer_type in enumerate(layer_types)]

def prop_model(model, states, inp):
    new_states = []

    out = inp

    state_ctr = 0

    for layer in model:

        if type(layer) not in [FF, FFS, FFT]:

            out, state = prop_layer[type(layer)](layer, states[state_ctr], out)
            new_states.append(state)
            state_ctr += 1

        else:

            out = prop_Flayer(layer, out)

        # dropout(out, inplace=True)

    if not config.act_classical_rnn:

        centers = out[:,:config.out_size//3].view(out.size(0), config.timestep_size, config.hm_modalities)
        spreads = out[:,config.out_size//3:-config.out_size//3].view(out.size(0), config.timestep_size, config.hm_modalities)
        multipliers = out[:,-config.out_size//3:].view(out.size(0), config.timestep_size, config.hm_modalities)

        spreads = exp(spreads)
        multipliers = softmax(multipliers, -1)

        out = [centers, spreads, multipliers]

    return out, new_states


##


# hann = (0.5-0.5 * cos(2*pi * arange(0,config.conv_window_size,1)/config.conv_window_size))
# inv_hann = lambda window: acos(-2*window +1) /(2*pi)


def convolve(layer, window):

    return conv1d(window, layer.w, stride=config.frame_stride)

def deconvolve(layer, window, true=False): # make True.

    if not true:
        return conv_transpose1d(window, layer.w, stride=config.frame_stride)
    else:
        pass # todo


def make_model2():

    init_fourier = config.frame_len == config.frame_out

    w_conv = randn(config.frame_out,config.frame_len, requires_grad=True)
    if init_fourier:
        with no_grad():
            for f in range(config.frame_len):
                w_conv[f,...] = cos(2*pi * (f+1)/config.frame_len * arange(0,config.frame_len,1))
        convolver = FF(w_conv)
    else:
        if config.init_xavier:
            xavier_normal_(w_conv, gain=5/3)
        convolver = FFT(w_conv)

    w_deconv = randn(config.frame_out,config.frame_len, requires_grad=True)
    if init_fourier:
        deconvolver = convolver
        #deconvolver = FF(w_deconv)
    else:
        if config.init_xavier:
            xavier_normal_(w_deconv, gain=5/3)
        deconvolver = FFT(w_deconv)

    return [convolver, make_model(), deconvolver]


def respond_to(model, sequences, state=None, training_run=True, extra_steps=0):

    responses = []
    loss = 0

    convolver, model, deconvolver = model

    if not state:
        state = empty_state(model, len(sequences))

    #print('size before convolve:',sequences[0].size())

    # sequences_pure = sequences
    #
    # sequences = [convolve(convolver,sequence) for sequence in sequences]
    #
    # #print('size after convolve:', sequences[0].size())
    #
    # sequences = [transpose(sequence,1,2) for sequence in sequences]

    #print('starting deconv experiments')

    #sequences = [transpose(sequence,1,2) for sequence in sequences]
    #sequences = [deconvolve(deconvolver,sequence) for sequence in sequences]

    # print(sequences_copy[0][0].sum())
    # print(sequences[0][0].sum())
    # print('size after transpose:',[sequence.size() for sequence in sequences])
    # input("Halt")

    max_seq_len = max(len(sequence) for sequence in sequences)
    hm_windows = ceil(max_seq_len/config.seq_stride_len)
    has_remaining = list(range(len(sequences)))

    #print('max seq len:',max_seq_len)

    for window_ctr in range(hm_windows):

        #print('window ctr:', window_ctr)

        window_start = window_ctr*config.seq_stride_len
        is_last_window = window_start+config.seq_window_len>=max_seq_len
        window_end = window_start+config.seq_window_len if not is_last_window else max_seq_len
        window_len = window_end-window_start

        #print('window ctr:',window_ctr)

        for window_t in range(window_len -1):

            #print('window t:',window_t)

            seq_force_ratio = config.seq_force_ratio**window_t

            t = window_start+window_t

            has_remaining = [i for i in has_remaining if sequences[i][t+1:t+2]]

            inp = cat([sequences[i][t] for i in has_remaining], 0)

            #print('inp size:', inp.size())

            #print('convolver size:', convolver.w.size())

            inp = convolver.w @ inp


            #print('conv"ed size:',inp.size())

            #input("Halt")

            if seq_force_ratio != 1:
                inp = inp * seq_force_ratio
                inp = inp + cat([responses[t-1][i].view(1,-1,1) for i in has_remaining], 0)  * (1-seq_force_ratio)

            # if window_t:
            #     inp = cat([sequences[i][:,t:t+1,:] for i in has_remaining], dim=0) *seq_force_ratio
            #     if seq_force_ratio != 1:
            #         inp = inp + stack([responses[t-1][i] for i in has_remaining],dim=0) *(1-seq_force_ratio)
            # else:
            #     inp = cat([sequences[i][:,t:t+1,:] for i in has_remaining], dim=0)

            # for ii in range(1,config.hm_steps_back+1):
            #     t_prev = t-ii
            #     if t_prev>=0:
            #         prev_inp = cat([sequences[i][:,t_prev:t_prev+1,:] for i in has_remaining],dim=0) *seq_force_ratio
            #     else:
            #         prev_inp = zeros(len(has_remaining),config.timestep_size) if not config.use_gpu else zeros(len(has_remaining),config.timestep_size).cuda()
            #     if seq_force_ratio != 1 and t_prev-1>=0:
            #         prev_inp = prev_inp + stack([responses[t_prev-1][i] for i in has_remaining], dim=0) *(1-seq_force_ratio)
            #     inp = cat([inp,prev_inp],dim=1)

            partial_state = [stack([layer_state[i] for i in has_remaining], dim=0) for layer_state in state]

            inp = inp.view(inp.size(0),inp.size(1))

            out, partial_state = prop_model(model, partial_state, inp)

            out = out.view(out.size(0),out.size(1),1)

            #print('out size exp"ed:', out.size())

            #print('deconvolver w size:', deconvolver.w.size())

            out = (deconvolver.w * out).sum(1) /config.frame_len

            #print('after deconvolution + sum(1) size:', out.size())

            lbl = cat([sequences[i][t+1] for i in has_remaining], 0)
            lbl = lbl.view(lbl.size(0),lbl.size(1))

            #print('lbl size:',lbl.size())


            #window_ctr*window_size+config.conv_window_size+config.conv_window_stride:(window_ctr+1)*window_size-config.conv_window_stride]

            #
            #input('Halt 2')

            # if not config.act_classical_rnn:
            #     out = sample_from_out(out)

            # print('out size2:', out.size())
            #

            loss += sequence_loss(lbl, out)

            if t >= len(responses):
                responses.append([out[has_remaining.index(i)] if i in has_remaining else None for i in range(len(sequences))])
            else:
                responses[t] = [out[has_remaining.index(i)] if i in has_remaining else None for i in range(len(sequences))]

            for s, ps in zip(state, partial_state):
                for ii,i in enumerate(has_remaining):
                    s[i] = ps[ii]

            if window_t+1 == config.seq_stride_len:
                state_to_transfer = [e.detach() for e in state]

        #print('has remaining start:',has_remaining_start)

        #for i in has_remaining_start:
            #print('\tworking on:',i)

            #print('\tpure seq size:',sequences_pure[i].size())
            #window_size = config.conv_window_size * config.seq_window_len

            # is this out / lbl business corrcetly indexed ?

            #print(window_len,config.conv_window_size*window_len)

            # lbl = sequences_pure[i][:,:, config.frame_len:-config.frame_stride]#window_ctr*window_size+config.conv_window_size+config.conv_window_stride:(window_ctr+1)*window_size-config.conv_window_stride]
            # #print('\tlbl size:', lbl.size())
            # out = cat([resp_t[i][None,:,:] for resp_t in responses if resp_t[i] is not None],2)
            # #print('out size initial:', out.size())
            # out = deconvolve(deconvolver, out)
            # #print('out after deconv:', out.size())
            #
            # # works with 1 file case but...  (or rather except all but one!!) -> rm the -conv//2 and make it conv_window_size
            #
            # out = out[:,:, config.frame_stride:-config.frame_stride]
            # #print('\tout size after cut:',out.size())
            #
            # #input('halt')
            #
            # # if not config.act_classical_rnn:
            # #     loss += distribution_loss(lbl, out)
            # # else:

        if not is_last_window:
            state = state_to_transfer
            responses = [[r.detach() if r is not None else None for r in resp] if t>=window_start else resp for t,resp in enumerate(responses)]
        else: break

    if training_run:
        loss.backward()
        return float(loss)

    else:

        if len(sequences) == 1:

            for t_extra in range(extra_steps):
                t = max_seq_len+t_extra-1

                # prev_responses = [response[0].view(1,response[0].size(0)) for response in reversed(responses[-(config.hm_steps_back+1):])]
                # for i in range(1, config.hm_steps_back+1): # tdo: do ?
                #     if len(sequences[0][t-1:t]):
                #         prev_responses[i-1] = sequences[0][t-1]

                # inp = cat([response for response in prev_responses],dim=-1)

                out = responses[-1][0].view(1,-1)

                out, partial_state = prop_model(model, partial_state, out)

                out = out.view(out.size(0), out.size(1), 1)

                # if not config.act_classical_rnn:
                #     out = sample_from_out(out)

                responses.append([out[0]])


            # TODO: now is reconstruction time from responses


            responses = cat([ee.view(1,-1,1) for e in responses for ee in e], dim=2)

            # responses = deconvolve(deconvolver,responses)

            print(responses.size())

            responses = responses[:,:,:-config.frame_out // 2]

        return float(loss), responses


##


def sequence_loss(label, out, do_stack=False):

    if do_stack:
        label = stack(label,dim=0)
        out = stack(out, dim=0)

    loss = pow(label-out, 2) if config.loss_squared else (label-out).abs()

    return loss.sum()


def distribution_loss(label, out):

    centers, spreads, multipliers = out

    label = label.view(label.size(0),label.size(1),1).repeat(1,1,config.hm_modalities)

    loss = 1/sqrt(2*pi) * exp( -.5 * pow((label-centers)/spreads,2) ) /spreads

    loss = (loss*multipliers).sum(-1)
    loss = -log(loss +1e-10)

    return loss.sum()

def sample_from_out(out):

    centers, spreads, multipliers = out

    sample = Normal(centers,spreads).rsample()
    sample = (sample*multipliers).sum(-1)

    return sample


##


def sgd(model, lr=None, batch_size=None):

    conv, model, deconv = model
    model = [conv] + model # + [deconv]

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    with no_grad():

        for layer in model:
            for param in layer._asdict().values():
                if param.requires_grad:

                    param.grad /=batch_size

                    if config.gradient_clip:
                        param.grad.clamp(min=-config.gradient_clip,max=config.gradient_clip)

                    param -= lr * param.grad
                    param.grad = None


moments, variances, ep_nr = [], [], 0


def adaptive_sgd(model, lr=None, batch_size=None,
                 alpha_moment=0.9, alpha_variance=0.999, epsilon=1e-8,
                 do_moments=True, do_variances=True, do_scaling=False):

    conv, model, deconv = model
    model = [conv] + model # + [deconv]

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    global moments, variances, ep_nr
    if not (moments or variances):
        if do_moments: moments = [[zeros(weight.size()) if not config.use_gpu else zeros(weight.size()).cuda() for weight in layer._asdict().values()] for layer in model]
        if do_variances: variances = [[zeros(weight.size()) if not config.use_gpu else zeros(weight.size()).cuda() for weight in layer._asdict().values()] for layer in model]

    ep_nr +=1

    with no_grad():
        for _, layer in enumerate(model):
            for __, weight in enumerate(layer._asdict().values()):
                if weight.requires_grad:
                    weight.grad /= batch_size
                    lr_ = lr

                    if do_moments:
                        moments[_][__] = alpha_moment * moments[_][__] + (1-alpha_moment) * weight.grad
                        moment_hat = moments[_][__] / (1-alpha_moment**(ep_nr+1))
                    if do_variances:
                        variances[_][__] = alpha_variance * variances[_][__] + (1-alpha_variance) * weight.grad**2
                        variance_hat = variances[_][__] / (1-alpha_variance**(ep_nr+1))
                    if do_scaling:
                        lr_ *= norm(weight)/norm(weight.grad)

                    weight -= lr_ * (moment_hat if do_moments else weight.grad) / ((sqrt(variance_hat)+epsilon) if do_variances else 1)
                    weight.grad = None


##


def load_model(path=None, fresh_meta=None):
    if not path: path = config.model_path
    if not fresh_meta: fresh_meta = config.fresh_meta
    path = path+'.pk'
    obj = pickle_load(path)
    if obj:
        model, meta, configs = obj
        conv, model, deconv = model
        #deconv = conv # Todo: now manually set here, is it good ?
        if config.use_gpu:
            conv = type(conv)(conv.w.cuda())
            TorchModel(model).cuda()
            deconv = conv # Todo: now manually set here good ?
        global moments, variances, ep_nr
        if fresh_meta:
            moments, variances, ep_nr = [], [], 0
        else:
            moments, variances, ep_nr = meta
            if config.use_gpu:
                moments = [[e2.cuda() for e2 in e1] for e1 in moments]
                variances = [[e2.cuda() for e2 in e1] for e1 in variances]
        for k_saved, v_saved in configs:
            v = getattr(config, k_saved)
            if v != v_saved:
                print(f'config conflict resolution: {k_saved} {v} -> {v_saved}')
                setattr(config, k_saved, v_saved)

        for layer in [conv] + model + [deconv]: # todo: instead, while pulling from gpu, update the grad info.
            try:
                if not layer.w.requires_grad: layer.w.requires_grad = True
            except:
                if not layer.wi.requires_grad: layer.wi.requires_grad = True

        return [conv, model, deconv]

def save_model(model, path=None):
    from warnings import filterwarnings
    filterwarnings("ignore")
    conv, model, deconv = model
    if not path: path = config.model_path
    path = path+'.pk'
    if config.use_gpu:
        moments_ = [[e2.detach().cuda() for e2 in e1] for e1 in moments]
        variances_ = [[e2.detach().cuda() for e2 in e1] for e1 in variances]
        meta = [moments_, variances_]
        model = pull_copy_from_gpu(model)
        conv, deconv = pull_copy_from_gpu([conv])[0], pull_copy_from_gpu([deconv])[0]
    else:
        meta = [moments, variances]
    model = [conv,model,deconv]
    meta.append(ep_nr)
    configs = [[field,getattr(config,field)] for field in dir(config) if field in config.config_to_save]
    pickle_save([model,meta,configs],path)


def empty_state(model, batch_size=1):
    states = []
    for layer in model:
        if type(layer) != FF and type(layer) != FFS and type(layer) != FFT:
            state = zeros(batch_size, getattr(layer,layer._fields[0]).size(1))
            if type(layer) == LSTM and prop_layer[LSTM] == prop_Llayer:
                state = cat([state]*2,dim=1)
            if config.use_gpu: state = state.cuda()
            states.append(state)
    return states


##


from torch.nn import Module, Parameter

class TorchModel(Module):

    def __init__(self, model):
        super(TorchModel, self).__init__()
        for layer_name, layer in enumerate(model):
            for field_name, field in layer._asdict().items():
                if type(field) != Parameter:
                    field = Parameter(field)
                setattr(self,f'layer{layer_name}_field{field_name}',field)
            setattr(self,f'layertype{layer_name}',type(layer))

            model[layer_name] = (getattr(self, f'layertype{layer_name}')) \
                (*[getattr(self, f'layer{layer_name}_field{field_name}') for field_name in getattr(self, f'layertype{layer_name}')._fields])
        self.model = model

    def forward(self, states, inp):
        prop_model(self.model, states, inp)


def pull_copy_from_gpu(model):
    return [type(layer)(*[weight.detach().cpu() for weight in layer._asdict().values()]) for layer in model]
