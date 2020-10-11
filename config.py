## base params

act_classical_rnn = True

## data params

sample_rate = 8_000 # 22_050 # 44_100

silence_thr_db = 0

data_path = 'data'
dev_ratio = 0

## model params

conv_window_size = 1024
conv_window_stride = conv_window_size//2
conv_out_size = 1024

hm_steps_back = 0
timestep_size = conv_out_size
in_size = timestep_size*(hm_steps_back+1)
hm_modalities = 1
out_size = timestep_size*hm_modalities*3 if not act_classical_rnn else timestep_size
creation_info = [in_size,'l',128,'ft' if act_classical_rnn else 'f',out_size]

init_xavier = True
forget_bias = 0

## train params

seq_window_len = 9999
seq_stride_len = seq_window_len-1
seq_force_ratio = 1 #0

loss_squared = True

learning_rate = 2e-3

batch_size = 2
gradient_clip = 0
hm_epochs = 100
optimizer = 'custom'

model_path = 'models/model'
fresh_model = True
fresh_meta = True
ckp_per_ep = hm_epochs//10

use_gpu = False

## interact params

hm_extra_steps = 1000 #seq_window_len

hm_wav_gen = 5

output_file = 'resp'

##

config_to_save = [
'sample_rate', 'conv_window_size', 'conv_window_stride', 'conv_hm_convolutions',
'hm_steps_back', 'in_size', 'hm_modalities', 'out_size',
'creation_info', 'act_classical_rnn',
]
