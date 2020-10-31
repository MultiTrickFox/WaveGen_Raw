## base params

conv_deconv_same = True

init_fourier = True

attention_only = True

## data params

sample_rate = 8_000 # 22_050 # 44_100

silence_thr_db = 0

data_path = 'data'
dev_ratio = 0

## model params

frame_len = 2048
frame_stride = frame_len//4
frame_out = frame_len//2+1

hm_steps_back = 0
timestep_size = frame_out
in_size = timestep_size *(hm_steps_back+1)
out_size = timestep_size
creation_info = [in_size,'l',128,'ft',out_size]

init_xavier = True
forget_bias = 0

## train params

seq_window_len = 9999
seq_stride_len = seq_window_len-1
seq_force_ratio = 1 #0

loss_squared = True

learning_rate = 1e-2

batch_size = 0
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
'sample_rate', 'frame_len', 'frame_stride', 'frame_out', 'conv_deconv_same',
'hm_steps_back', 'in_size', 'hm_modalities', 'out_size',
'creation_info', 'act_classical_rnn',
]
