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


from model import FF, FFS, FFT, make_model, prop_Flayer

def prop_model(model, io):

    for layer in model:
        io = prop_Flayer(layer,io)
        # dropout(out, inplace=True)

    return io


#hann = (0.5-0.5 * cos(2*pi * arange(0,config.frame_len,1)/config.frame_len))
# inv_hann = lambda window: acos(-2*window +1) /(2*pi)


def make_model_higher():

    w_conv = randn(config.frame_out,config.frame_len, requires_grad=True)
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
            w_deconv.requires_grad = True
            deconvolver = FF(w_deconv)
        else:
            w_deconv = randn(config.frame_out, config.frame_len, requires_grad=True)
            if config.init_xavier:
                xavier_normal_(w_deconv, gain=5/3)
            deconvolver = FFT(w_deconv)

    body = config.creation_info[1:-1]
    enc = make_model([config.timestep_size*2] +body+ [1])
    dec = make_model([config.timestep_size] +body+ [config.timestep_size])

    return [[convolver], [enc,dec], [deconvolver]]


def respond_to(model, sequences, training_run=True, extra_steps=0):

    responses = []
    loss = 0

    convolver, [enc,dec], deconvolver = model

    with no_grad():
        convolver[0].w *= hann
        deconvolver[0].w *= inv_hann


    for sequence in sequences:

        sequence_conv = conv1d(sequence,)



    input("Halt here")





    sequences_conv = [] # todo: convolve here







    max_seq_len = max(len(sequence) for sequence in sequences)
    hm_windows = ceil(max_seq_len/config.seq_stride_len)
    has_remaining = list(range(len(sequences)))

    for window_ctr in range(hm_windows):

        window_start = window_ctr*config.seq_stride_len
        is_last_window = window_start+config.seq_window_len>=max_seq_len
        window_end = window_start+config.seq_window_len if not is_last_window else max_seq_len
        window_len = window_end-window_start
        has_remaining_start = has_remaining

        for window_t in range(window_len -1):

            seq_force_ratio = config.seq_force_ratio**window_t

            t = window_start+window_t

            has_remaining = [i for i in has_remaining if sequences[i][t+1:t+2]]

            inp = cat([sequences[i][t] for i in has_remaining],0)

            inp = convolver.w @ inp

            if seq_force_ratio != 1:
                inp = inp * seq_force_ratio
                inp = inp + cat([responses[t-1][i].view(1,-1,1) for i in has_remaining], 0)  * (1-seq_force_ratio)

            inp = inp.view(inp.size(0),inp.size(1))

            out = prop_model(model, inp)

            out = out.view(out.size(0),out.size(1),1)

            out = (deconvolver.w * out).sum(1) /config.frame_len

            lbl = cat([sequences[i][t+1] for i in has_remaining], 0)

            lbl = lbl.view(lbl.size(0),lbl.size(1))

            loss += sequence_loss(lbl, out)

            if t >= len(responses):
                responses.append([out[has_remaining.index(i)] if i in has_remaining else None for i in range(len(sequences))])
            else:
                responses[t] = [out[has_remaining.index(i)] if i in has_remaining else None for i in range(len(sequences))]


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


def sequence_loss(label, out, do_stack=False):

    if do_stack:
        label = stack(label,dim=0)
        out = stack(out, dim=0)

    loss = pow(label-out, 2) if config.loss_squared else (label-out).abs()

    return loss.sum()


##


def sgd(model, lr=None, batch_size=None):

    conv, model, deconv = model
    model = conv + model + (deconv if not config.conv_deconv_same else [])

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    with no_grad():

        for layer in model:
            for param in layer._asdict().values():

                param.grad /= batch_size

                if config.gradient_clip:
                    param.grad.clamp(min=-config.gradient_clip,max=config.gradient_clip)

                param -= lr * param.grad
                param.grad = None


moments, variances, ep_nr = [], [], 0


def adaptive_sgd(model, lr=None, batch_size=None,
                 alpha_moment=0.9, alpha_variance=0.999, epsilon=1e-8,
                 do_moments=True, do_variances=True, do_scaling=False):

    conv, model, deconv = model
    model = conv + model + (deconv if not config.conv_deconv_same else [])

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
                weight.grad /= batch_size
                lr_ = lr

                #print(f'{list(layer._asdict().keys())[__]}',weight.grad.pow(2).sum())

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
        if config.conv_deconv_same:
            deconv = conv
        if config.use_gpu:
            TorchModel(conv).cuda()
            TorchModel(model).cuda()
            if config.conv_deconv_same:
                deconv = conv
            else:
                TorchModel(deconv).cuda()
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
        conv, deconv = pull_copy_from_gpu(conv), pull_copy_from_gpu(deconv)
    else:
        meta = [moments, variances]
    model = [conv,model,deconv]
    meta.append(ep_nr)
    configs = [[field,getattr(config,field)] for field in dir(config) if field in config.config_to_save]
    pickle_save([model,meta,configs],path)


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
    model_copy = [type(layer)(*[weight.detach().cpu() for weight in layer._asdict().values()]) for layer in model]
    for layer in model_copy:
        for w in layer._asdict().values():
            w.requires_grad = True
    return model_copy
