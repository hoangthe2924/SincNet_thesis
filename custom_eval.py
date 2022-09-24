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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
import numpy as np
from dnn_models import MLP, flip
from dnn_models import SincNet as CNN
from data_io import ReadList, read_conf, str_to_bool
from loss_function import TripletLossWithHammingDistance as TLHD
from loss_function import TripletLoss as TL
import random

def get_list_seq_speaker_by_sample(lab_dict, sample):
    # sample is expected to be a filepath
    label = lab_dict[sample]
    
    return [key for key, val in lab_dict.items() if val == label]


def get_negative_file(list_file, list_file_speaker):
    filepath = random.choice(list_file)
    while (filepath in list_file_speaker):
        filepath = random.choice(list_file)
    return filepath


def get_positive_file(list_file, list_file_speaker, anc_file):
    filepath = random.choice(list_file_speaker)
    while (filepath not in list_file or filepath == anc_file):
        filepath = random.choice(list_file_speaker)
    return filepath


def access_random_chunk(signal, wlen, filepath):
    snt_len = signal.shape[0]
    snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
    snt_end = snt_beg + wlen

    channels = len(signal.shape)
    if channels == 2:
        print('WARNING: stereo to mono: ' + filepath)
        signal = signal[:, 0]
    return signal[snt_beg:snt_end]


def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict, fact_amp):
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    anc_sig_batch = np.zeros([batch_size, wlen])
    pos_sig_batch = np.zeros([batch_size, wlen])
    neg_sig_batch = np.zeros([batch_size, wlen])
    lab_batch = np.zeros(batch_size)

    snt_id_arr = np.random.randint(N_snt, size=batch_size)

    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

    for i in range(batch_size):
        # select a random sentence from the list
        # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        # signal=signal.astype(float)/32768

        list_file_of_speaker = get_list_seq_speaker_by_sample(lab_dict, wav_lst[snt_id_arr[i]])
        pos_file = get_positive_file(wav_lst, list_file_of_speaker, wav_lst[snt_id_arr[i]])
        neg_file = get_negative_file(wav_lst, list_file_of_speaker)

        [anc_sig, anc_fs] = sf.read(data_folder + wav_lst[snt_id_arr[i]])
        [pos_sig, pos_fs] = sf.read(data_folder + pos_file)
        [neg_sig, neg_fs] = sf.read(data_folder + neg_file)

        # accesing to a random chunk
        anc_sig_batch[i, :] = access_random_chunk(anc_sig, wlen, data_folder + wav_lst[snt_id_arr[i]]) * rand_amp_arr[i]
        pos_sig_batch[i, :] = access_random_chunk(pos_sig, wlen, data_folder + pos_file) * rand_amp_arr[i]
        neg_sig_batch[i, :] = access_random_chunk(neg_sig, wlen, data_folder + neg_file) * rand_amp_arr[i]

        # get label of anchor
        label = lab_dict[wav_lst[snt_id_arr[i]]]

        lab_batch[i] = label

    anc = Variable(torch.from_numpy(anc_sig_batch).float().cuda().contiguous())
    pos = Variable(torch.from_numpy(pos_sig_batch).float().cuda().contiguous())
    neg = Variable(torch.from_numpy(neg_sig_batch).float().cuda().contiguous())
    lab = Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())

    # return  torch.tensor(anc, dtype=torch.float64), torch.tensor(pos, dtype=torch.float64), torch.tensor(neg, dtype=torch.float64), lab
    return anc, pos, neg, lab

def compute_d_vector(signal, wlen, wshift, Batch_dev, d_vector_dim, DNN1_net, CNN_net, avoid_small_en_fr = True):
    # Amplitude normalization
    signal = signal / np.max(np.abs(signal))

    signal = torch.from_numpy(signal).float().cuda().contiguous()

    if avoid_small_en_fr:
        # computing energy on each frame:
        beg_samp = 0
        end_samp = wlen

        N_fr = int((signal.shape[0] - wlen) / (wshift))
        Batch_dev = N_fr
        en_arr = torch.zeros(N_fr).float().contiguous().cuda()
        count_fr = 0
        count_fr_tot = 0
        while end_samp < signal.shape[0]:
            en_arr[count_fr] = torch.sum(signal[beg_samp:end_samp].pow(2))
            beg_samp = beg_samp + wshift
            end_samp = beg_samp + wlen
            count_fr = count_fr + 1
            count_fr_tot = count_fr_tot + 1
            if count_fr == N_fr:
                break

        en_arr_bin = en_arr > torch.mean(en_arr) * 0.1
        en_arr_bin.cuda()
        n_vect_elem = torch.sum(en_arr_bin)

        if n_vect_elem < 10:
            print('only few elements used to compute d-vectors')
            sys.exit(0)

    # split signals into chunks
    beg_samp = 0
    end_samp = wlen

    N_fr = int((signal.shape[0] - wlen) / (wshift))

    sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()
    dvects = Variable(torch.zeros(N_fr, d_vector_dim).float().cuda().contiguous())
    count_fr = 0
    count_fr_tot = 0
    while end_samp < signal.shape[0]:
        sig_arr[count_fr, :] = signal[beg_samp:end_samp]
        beg_samp = beg_samp + wshift
        end_samp = beg_samp + wlen
        count_fr = count_fr + 1
        count_fr_tot = count_fr_tot + 1
        if count_fr == Batch_dev:
            inp = Variable(sig_arr)
            dvects[count_fr_tot - Batch_dev:count_fr_tot, :] = (DNN1_net(CNN_net(inp)))
            count_fr = 0
            sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()
    

    if count_fr > 0:
        inp = Variable(sig_arr[0:count_fr])
        dvects[count_fr_tot - count_fr:count_fr_tot, :] = (DNN1_net(CNN_net(inp)))
    
    if avoid_small_en_fr:
        dvects = dvects.index_select(0, torch.nonzero((en_arr_bin == 1), as_tuple=False).view(-1))

    # averaging and normalizing all the d-vectors
    d_vect_out = torch.mean(dvects / dvects.norm(p=2, dim=1).view(-1, 1), dim=0)

    # checks for nan
    nan_sum = torch.sum(torch.isnan(d_vect_out))

    if nan_sum > 0:
        print('nan', wav_lst_te[i])
        sys.exit(0)

    return d_vect_out.view(1,-1)


# Reading cfg file
options = read_conf()

# [data]
tr_lst = options.tr_lst
te_lst = options.te_lst
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder + "/"
output_folder = options.output_folder

# [windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

# [cnn]
cnn_N_filt = list(map(int, options.cnn_N_filt.split(",")))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(",")))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(",")))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(",")))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(",")))
cnn_act = list(map(str, options.cnn_act.split(",")))
cnn_drop = list(map(float, options.cnn_drop.split(",")))


# [dnn]
fc_lay = list(map(int, options.fc_lay.split(",")))
fc_drop = list(map(float, options.fc_drop.split(",")))
fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(",")))
fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(",")))
fc_act = list(map(str, options.fc_act.split(",")))

# [class]
class_lay = list(map(int, options.class_lay.split(",")))
class_drop = list(map(float, options.class_drop.split(",")))
class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(",")))
class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(",")))
class_act = list(map(str, options.class_act.split(",")))


# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)


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


# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
# cost = nn.NLLLoss()
# cost = TL()
cost = TLHD()

# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)
wshift = int(fs * cw_shift / 1000.00)

# Batch_dev
Batch_dev = 128


# Feature extractor CNN
CNN_arch = {
    "input_dim": wlen,
    "fs": fs,
    "cnn_N_filt": cnn_N_filt,
    "cnn_len_filt": cnn_len_filt,
    "cnn_max_pool_len": cnn_max_pool_len,
    "cnn_use_laynorm_inp": cnn_use_laynorm_inp,
    "cnn_use_batchnorm_inp": cnn_use_batchnorm_inp,
    "cnn_use_laynorm": cnn_use_laynorm,
    "cnn_use_batchnorm": cnn_use_batchnorm,
    "cnn_act": cnn_act,
    "cnn_drop": cnn_drop,
}

CNN_net = CNN(CNN_arch)
CNN_net.cuda()
# print(CNN_net.parameters)
# for name, param in CNN_net.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# Loading label dictionary
lab_dict = np.load(class_dict_file, allow_pickle=True).item()



DNN1_arch = {
    "input_dim": CNN_net.out_dim,
    "fc_lay": fc_lay,
    "fc_drop": fc_drop,
    "fc_use_batchnorm": fc_use_batchnorm,
    "fc_use_laynorm": fc_use_laynorm,
    "fc_use_laynorm_inp": fc_use_laynorm_inp,
    "fc_use_batchnorm_inp": fc_use_batchnorm_inp,
    "fc_act": fc_act,
}

DNN1_net = MLP(DNN1_arch)
DNN1_net.cuda()



# DNN2_arch = {
#     "input_dim": fc_lay[-1],
#     "fc_lay": class_lay,
#     "fc_drop": class_drop,
#     "fc_use_batchnorm": class_use_batchnorm,
#     "fc_use_laynorm": class_use_laynorm,
#     "fc_use_laynorm_inp": class_use_laynorm_inp,
#     "fc_use_batchnorm_inp": class_use_batchnorm_inp,
#     "fc_act": class_act,
# }


# DNN2_net = MLP(DNN2_arch)
# DNN2_net.cuda()


if pt_file != "none":
    checkpoint_load = torch.load(pt_file)
    CNN_net.load_state_dict(checkpoint_load["CNN_model_par"])
    DNN1_net.load_state_dict(checkpoint_load["DNN1_model_par"])
    # DNN2_net.load_state_dict(checkpoint_load["DNN2_model_par"])

CNN_net.eval()
DNN1_net.eval()
test_flag = 1
loss_sum = 0
get_sample = False
inter_arr = []
outer_arr = []
d_vector_dim=fc_lay[-1]
is_limit = True
inter_arr_limit = 1386
inter_cnt_looper = 0
outer_arr_limit = 4000
outer_cnt_looper = 0

print('Start Evaluation')

with torch.no_grad():
    for i in range(snt_te):
        
        [anc_sig, anc_fs] = sf.read(data_folder + wav_lst_te[i])
        anc_pout = compute_d_vector(anc_sig, wlen, wshift, Batch_dev, d_vector_dim, DNN1_net, CNN_net)

        start_j = i + 1
        end_j = i+3
        if (end_j > snt_te):
            end_j = snt_te
            
        for j in range(start_j, end_j):
            
            [compare_anc_sig, compare_anc_fs] = sf.read(data_folder + wav_lst_te[j])
            compare_anc_pout = compute_d_vector(compare_anc_sig, wlen, wshift, Batch_dev, d_vector_dim, DNN1_net, CNN_net)
            
            compare_result = torch.norm(anc_pout - compare_anc_pout)
            if(lab_dict[wav_lst_te[i]] == lab_dict[wav_lst_te[j]]):
                inter_arr.append(compare_result.cpu().detach().numpy())
                inter_cnt_looper+=1
            
            # if (inter_cnt_looper >= inter_arr_limit and is_limit):
            #     break;

        if (inter_cnt_looper >= inter_arr_limit and is_limit):
            break
np.savetxt(output_folder + '/statistical_inter.txt', inter_arr, delimiter='\n')
print('Done inter');

with torch.no_grad():
    snt_wav_random = torch.randperm(snt_te)
    for i in range(snt_te):
        
        [anc_sig, anc_fs] = sf.read(data_folder + wav_lst_te[i])
        anc_pout = compute_d_vector(anc_sig, wlen, wshift, Batch_dev, d_vector_dim, DNN1_net, CNN_net)

        for j in range(i+1, snt_te):            
            
            [compare_anc_sig, compare_anc_fs] = sf.read(data_folder + wav_lst_te[j])
            compare_anc_pout = compute_d_vector(compare_anc_sig, wlen, wshift, Batch_dev, d_vector_dim, DNN1_net, CNN_net)
            
            compare_result = torch.norm(anc_pout - compare_anc_pout)
            if (lab_dict[wav_lst_te[j]] != lab_dict[wav_lst_te[i]]):
                outer_arr.append(compare_result)
                outer_cnt_looper+=1
            
            if (outer_cnt_looper >= outer_arr_limit and is_limit):
                break;
        print("loop: ", i,", ", outer_cnt_looper)
        if (outer_cnt_looper >= outer_arr_limit and is_limit):
            break;


np.savetxt(output_folder + '/statistical_outer.txt', outer_arr, delimiter='\n')
print('Done inter');

# print("epoch %i, loss_tr=%f loss_te=%f, outer_eval_mean=%f, inner_eval_mean=%f" % (
#     epoch, loss_tot, loss_tot_dev, outer_eval_sum / snt_te, inter_eval_sum / snt_te))

# with open(output_folder + "/res.res", "a") as res_file:
#     res_file.write("epoch %i, loss_tr=%f loss_te=%f, outer_err=%f, inner_err=%f \n" % (
#         epoch, loss_tot, loss_tot_dev, outer_eval_sum / snt_te, inter_eval_sum / snt_te))

# checkpoint = {'CNN_model_par': CNN_net.state_dict(),
#                 'DNN1_model_par': DNN1_net.state_dict(),
#             }
        