# speaker_id.py
# Mirco Ravanelli
# Mila - University of Montreal

# July 2018

# Description:
# This code performs a speaker_id experiments with SincNet.

# How to run it:
# python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg

import os
# import scipy.io.wavfile
import soundfile as sf
import torch

import numpy as np
import random
import shutil
from dnn_models import MLP, flip
from dnn_models import SincNet as CNN
from data_io import ReadList, read_conf, str_to_bool
from loss_function import TripletLossEuclidean_Criteria as TLEC
from loss_function import TripletLossHamming_Criteria as TLHC
from utils import *


def copy_folder(in_folder,out_folder):
    if not(os.path.isdir(out_folder)):
        shutil.copytree(in_folder, out_folder, ignore=ig_f)

def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


# Reading cfg file
options = read_conf()

# [data]
tr_lst = options.tr_lst
te_lst = options.te_lst
al_lst = options.al_lst
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder + '/'
output_folder = options.output_folder

# [windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

# [cnn]
cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act = list(map(str, options.cnn_act.split(',')))
cnn_drop = list(map(float, options.cnn_drop.split(',')))

# [dnn]
fc_lay = list(map(int, options.fc_lay.split(',')))
fc_drop = list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act = list(map(str, options.fc_act.split(',')))

# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
alpha = float(options.alpha)
beta = float(options.beta)
margin = float(options.margin)
n_intra_sample = int(options.nintra)
loss_type=options.loss_type
N_epochs = int(options.N_epochs)
start_epoch = int(options.start_epoch)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)

# all list
wav_lst_all = ReadList(al_lst)
snt_al = len(wav_lst_all)

# training list
wav_lst_tr = ReadList(tr_lst)
snt_tr = len(wav_lst_tr)

# test list
wav_lst_te = ReadList(te_lst)
snt_te = len(wav_lst_te)

# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder)

# embeddings folder creation
ebd_output_folder = output_folder + "ebd/"
# try:
#     os.stat(ebd_output_folder)
# except:
#     os.mkdir(ebd_output_folder)

copy_folder(data_folder, ebd_output_folder)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)
wshift = int(fs * cw_shift / 1000.00)

# Batch_dev
Batch_dev = 128

# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
            'fs': fs,
            'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt': cnn_len_filt,
            'cnn_max_pool_len': cnn_max_pool_len,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
            'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm': cnn_use_laynorm,
            'cnn_use_batchnorm': cnn_use_batchnorm,
            'cnn_act': cnn_act,
            'cnn_drop': cnn_drop,
            }

CNN_net = CNN(CNN_arch)
CNN_net.cuda()

# Loading label dictionary
lab_dict = np.load(class_dict_file, allow_pickle=True).item()
N_spks = max(lab_dict.values()) + 1

DNN1_arch = {'input_dim': CNN_net.out_dim,
             'fc_lay': fc_lay,
             'fc_drop': fc_drop,
             'fc_use_batchnorm': fc_use_batchnorm,
             'fc_use_laynorm': fc_use_laynorm,
             'fc_use_laynorm_inp': fc_use_laynorm_inp,
             'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
             'fc_act': fc_act,
             }

DNN1_net = MLP(DNN1_arch)
DNN1_net.cuda()

d_vector_dim = fc_lay[-1]

if pt_file != 'none':
    checkpoint_load = torch.load(pt_file)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])

# Other cfg for evaluation
is_limit = True
print("n intra in dev: ", n_intra_sample)
print("n all: ", snt_al)
intra_arr_limit = snt_te*n_intra_sample
wav_lst_spk_dict = get_list_wav_of_speaker_label(lab_dict)

intra_cnt_looper = 0
inter_cnt_looper = 0

CNN_net.eval()
DNN1_net.eval()
test_flag = 1
loss_sum = 0

ec_intra_arr = np.array([])
ec_inter_arr = np.array([])
hm_intra_arr = np.array([])
hm_inter_arr = np.array([])
ec_gr_inter_arr = np.array([])
hm_gr_inter_arr = np.array([])

dvec_list = np.zeros([snt_te, d_vector_dim])

# for vkyc
wav_list_ex = [file for file in wav_lst_all if file not in wav_lst_tr and file not in wav_lst_te]
print("ntrain + ndev", snt_tr+snt_te)

# for timit
# wav_list_ex = wav_lst_te

print("n extracted templates: ", len(wav_list_ex))

with torch.no_grad():
    # compute all d-vectors and save to txt file
    for idx, file_ex in enumerate(wav_list_ex):
        [sig, fs] = sf.read(data_folder + file_ex)
        dvec_out = compute_d_vector(sig, file_ex, wlen, wshift, Batch_dev, d_vector_dim, DNN1_net, CNN_net)
        dvec_list = dvec_out.cpu().detach().numpy()[0, :]
        np.savetxt(ebd_output_folder+file_ex.replace('.wav','.txt'), dvec_list)
        print('finish %d'%idx, file_ex)
