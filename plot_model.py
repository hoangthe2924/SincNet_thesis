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
from dnn_models import MLP, flip
from dnn_models import SincNet as CNN
from data_io import ReadList, read_conf, str_to_bool
from loss_function import TripletLossEuclidean_Criteria as TLEC
from loss_function import TripletLossHamming_Criteria as TLHC
from utils import *

# Reading cfg file
options = read_conf()

# [data]
tr_lst = options.tr_lst
te_lst = options.te_lst
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

# Plot folder creation
plot_output_folder = output_folder + "/plots"
try:
    os.stat(plot_output_folder)
except:
    os.mkdir(plot_output_folder)

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
print("n test sample: ", snt_te)
# for wav in wav_lst_te:
#     print(wav)

wav_lst_spk_dict = get_list_wav_of_speaker_label(lab_dict)

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

print(d_vector_dim)

with torch.no_grad():
    # compute all d-vectors and store them in a list
    for i in range(snt_te):
        [sig, fs] = sf.read(data_folder + wav_lst_te[i])
        dvec_out = compute_d_vector(sig, wav_lst_te[i], wlen, wshift, Batch_dev, d_vector_dim, DNN1_net, CNN_net)
        dvec_list[i,:] = dvec_out.cpu().detach().numpy()[0, :]
    
    # evaluate intra
    for i in range(snt_te):
        anc = dvec_list[i]

        start_j = i + 1
        end_j = i + n_intra_sample
        if end_j > snt_te:
            end_j = snt_te

        for j in range(start_j, end_j):
            if (lab_dict[wav_lst_te[i]] == lab_dict[wav_lst_te[j]]):
                compared_pos = dvec_list[j]

                # compute euclidean distance of intra
                ec_res = compute_euclidean_dist(anc, compared_pos, d_vector_dim)
                ec_intra_arr = np.concatenate((ec_intra_arr, ec_res))

                # compute hamming distance of intra
                hm_res = compute_hamming_dist(anc, compared_pos, d_vector_dim, ver='thd')
                hm_intra_arr = np.concatenate((hm_intra_arr, hm_res))
    
    
    for i in range(0, snt_te-n_intra_sample, n_intra_sample):
        for j in range(n_intra_sample):
            anc = dvec_list[i+j]
            ec_res = compute_euclidean_dist(anc, dvec_list[i+n_intra_sample:], d_vector_dim)
            ec_inter_arr = np.concatenate((ec_inter_arr, ec_res))
            
            hm_res = compute_hamming_dist(anc, dvec_list[i+n_intra_sample:], d_vector_dim, ver='thd')
            hm_inter_arr = np.concatenate((hm_inter_arr, hm_res))
    
#     k=4
#     mean_dvec_list = mean_each_kElements(k, dvec_list)
#     snt_new_list = mean_dvec_list.shape[0]
#     n_template_per_speaker_in_mean_list = n_intra_sample//2
#     for i in range(0, snt_new_list-n_template_per_speaker_in_mean_list, n_template_per_speaker_in_mean_list):
#         for j in range(n_template_per_speaker_in_mean_list):
#             anc = mean_dvec_list[i+j]
#             ec_res = compute_euclidean_dist(anc, mean_dvec_list[i+n_template_per_speaker_in_mean_list:], d_vector_dim)
#             ec_gr_inter_arr = np.concatenate((ec_gr_inter_arr, ec_res))
            
#             hm_res = compute_hamming_dist(anc, mean_dvec_list[i+n_template_per_speaker_in_mean_list:], d_vector_dim, ver='thd')
#             hm_gr_inter_arr = np.concatenate((hm_gr_inter_arr, hm_res))
     
    # Compute mean
    ec_intra_mean = np.mean(ec_intra_arr)
    ec_inter_mean = np.mean(ec_inter_arr)
    hm_intra_mean = np.mean(hm_intra_arr)
    hm_inter_mean = np.mean(hm_inter_arr)
    hm_intra_var = np.var(hm_intra_arr)
    hm_inter_var = np.var(hm_inter_arr)
    print("hm_intra_mean=%f intra_var=%f hm_inter_mean=%f inter_var=%f ec_intra_mean=%f ec_inter_mean=%f" % (
            hm_intra_mean, hm_intra_var, hm_inter_mean, hm_inter_var, ec_intra_mean, ec_inter_mean))
    
    with open(output_folder + "/res.res", "a") as res_file:
        res_file.write("\n evaluation: hm_intra_mean=%f intra_var=%f hm_inter_mean=%f inter_var=%f ec_intra_mean=%f ec_inter_mean=%f\n"%(hm_intra_mean, hm_intra_var, hm_inter_mean, hm_inter_var, ec_intra_mean, ec_inter_mean))
    

visualize_distribution(ec_intra_arr, ec_inter_arr, "Euclidean Distance Evaluation",
                       plot_output_folder + "/Euclidean_Model-f.png", 2.0, 0.02, 2.0, 0.2)
visualize_distribution(hm_intra_arr, hm_inter_arr, "Hamming Distance Evaluation",
                       plot_output_folder + "/Hamming_Model-f.png", 1.0, 0.01, 1.0, 0.2)

# visualize_distribution(ec_intra_arr, ec_gr_inter_arr, "Euclidean Distance Evaluation of Model grouped by %d - final"%(k),
#                        plot_output_folder + "/Euclidean_Model-gr%d-f.png"%(k), 2.0, 0.02, 2.0, 0.2)
# visualize_distribution(hm_intra_arr, hm_gr_inter_arr, "Hamming Distance Evaluation of Model grouped by %d - final"%(k),
#                        plot_output_folder + "/Hamming_Model-gr%d-f.png"%(k), 1.0, 0.01, 1.0, 0.2)
