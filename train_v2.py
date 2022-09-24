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

import numpy as np
import random
from dnn_models import MLP, flip
from dnn_models import SincNet as CNN
from data_io import ReadList, read_conf, str_to_bool
from loss_function import TripletLossEuclidean_Criteria as TLEC
from loss_function import TripletLossEuclidean_Criteria_V2 as TLEC2
from loss_function import TripletLossHamming_Criteria as TLHC
from loss_function import TripletLossHamming_Criteria_V2 as TLHC2
from loss_function import TripletLossWithHammingDistance as TLWHD
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

# loss function
if loss_type == 'hm':
    cost = TLHC(alpha=alpha, beta=beta, margin = margin)
elif loss_type == 'ec':
    cost = TLEC(alpha=alpha, beta=beta, margin = margin)
elif loss_type == 'hm2':
    cost = TLHC2(alpha=alpha, beta=beta, margin = margin)
elif loss_type == 'ec2':
    cost = TLEC2(alpha=alpha, beta=beta, margin = margin)
else:
    cost = TLHC(alpha=alpha, beta=beta, margin = margin)

if pt_file != 'none':
    checkpoint_load = torch.load(pt_file)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])

# optimization algorithm: RMSprop
optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)

# learning rate decay
# decayRate = 0.9
# step_size_decay = 100
# my_lr_scheduler_CNN = torch.optim.lr_scheduler.StepLR(optimizer_CNN, step_size=step_size_decay, gamma=decayRate)
# my_lr_scheduler_DNN = torch.optim.lr_scheduler.StepLR(optimizer_DNN1, step_size=step_size_decay, gamma=decayRate)

# Other cfg for evaluation
is_limit = True
print("n intra in dev: ", n_intra_sample)
intra_arr_limit = snt_te*n_intra_sample
wav_lst_spk_dict = get_list_wav_of_speaker_label(lab_dict)

print('Traning...')
for epoch in range(start_epoch, N_epochs + 1):
    print('\n Start epoch: %d\n' % epoch)
    test_flag = 0
    CNN_net.train()
    DNN1_net.train()
    # DNN2_net.train()

    loss_sum = 0
    start_idx = 0
    # Note: this line modifies the original list
    random.shuffle(wav_lst_tr)


    for i in range(N_batches):
        [anc, pos, neg, lab, new_wav_lst_tr, new_start_id] = create_batches_rnd(batch_size, data_folder, wav_lst_tr,
                                                                                snt_tr, wlen, lab_dict, wav_lst_spk_dict, 0.2, start_idx)

        # update for next batch
        wav_lst_tr = new_wav_lst_tr
        start_idx = new_start_id

        anc_pout = DNN1_net(CNN_net(anc))
        pos_pout = DNN1_net(CNN_net(pos))
        neg_pout = DNN1_net(CNN_net(neg))

        loss = cost(anc_pout, pos_pout, neg_pout)

        optimizer_CNN.zero_grad()
        optimizer_DNN1.zero_grad()

        loss.backward()
        optimizer_CNN.step()
        optimizer_DNN1.step()

        loss_sum = loss_sum + loss.detach()
        print("--Batch %d, loss: %f" % (i, loss))

    loss_tot = loss_sum / N_batches
#     my_lr_scheduler_CNN.step()
#     my_lr_scheduler_DNN.step()

    # Full Validation  new
    if epoch % N_eval_epoch == 0 and epoch!=0:

        CNN_net.eval()
        DNN1_net.eval()
        test_flag = 1
        loss_sum = 0
        ec_intra_arr = np.array([])
        ec_inter_arr = np.array([])
        hm_intra_arr = np.array([])
        hm_inter_arr = np.array([])
        dvec_list = np.zeros([snt_te, d_vector_dim])
        # dvec_dict = {}

        intra_cnt_looper = 0
        inter_cnt_looper = 0

        with torch.no_grad():
            # compute all d-vectors and store them in a list
            for i in range(snt_te):
                [sig, fs] = sf.read(data_folder + wav_lst_te[i])
                dvec_out = compute_d_vector(sig, wav_lst_te[i], wlen, wshift, Batch_dev, d_vector_dim, DNN1_net,
                                            CNN_net)
                dvec_list[i, :] = dvec_out.cpu().detach().numpy()[0, :]

            # evaluate intra
            for i in range(snt_te):
                anc = dvec_list[i]

                start_j = i + 1
                end_j = i + n_intra_sample
                if end_j > snt_te:
                    end_j = snt_te

                for j in range(start_j, end_j):
                    if lab_dict[wav_lst_te[i]] == lab_dict[wav_lst_te[j]]:
                        compared_pos = dvec_list[j]

                        # compute euclidean distance of intra
                        ec_res = compute_euclidean_dist(anc, compared_pos, d_vector_dim)
                        ec_intra_arr = np.concatenate((ec_intra_arr, ec_res))

                        # compute hamming distance of intra
                        hm_res = compute_hamming_dist(anc, compared_pos, d_vector_dim, ver='thd')
                        hm_intra_arr = np.concatenate((hm_intra_arr, hm_res))

                if intra_cnt_looper >= intra_arr_limit and is_limit:
                    break

            for i in range(0, snt_te - n_intra_sample, n_intra_sample):
                for j in range(n_intra_sample):
                    anc = dvec_list[i + j]
                    ec_res = compute_euclidean_dist(anc, dvec_list[i + n_intra_sample:], d_vector_dim)
                    ec_inter_arr = np.concatenate((ec_inter_arr, ec_res))

                    hm_res = compute_hamming_dist(anc, dvec_list[i + n_intra_sample:], d_vector_dim, ver='thd')
                    hm_inter_arr = np.concatenate((hm_inter_arr, hm_res))

        # Compute mean
        ec_intra_mean = np.mean(ec_intra_arr)
        ec_inter_mean = np.mean(ec_inter_arr)
        hm_intra_mean = np.mean(hm_intra_arr)
        hm_inter_mean = np.mean(hm_inter_arr)

        print("epoch %i, loss_tr=%f hm_intra_mean=%f hm_inter_mean=%f ec_intra_mean=%f ec_inter_mean=%f" % (
            epoch, loss_tot, hm_intra_mean, hm_inter_mean, ec_intra_mean, ec_inter_mean))

        visualize_distribution(ec_intra_arr, ec_inter_arr, "Euclidean Distance Evaluation at Epoch %d" % epoch,
                               plot_output_folder + "/Euclidean_Epoch_%04d.png" % epoch, 2.0, 0.02, 2.0, 0.2)
        visualize_distribution(hm_intra_arr, hm_inter_arr, "Hamming Distance Evaluation at Epoch %d" % epoch,
                               plot_output_folder + "/Hamming_Epoch_%04d.png" % epoch, 1.0, 0.01, 1.0, 0.2)

        with open(output_folder + "/res.res", "a") as res_file:
            res_file.write(
                "epoch %i, loss_tr=%f hm_intra_mean=%f hm_inter_mean=%f ec_intra_mean=%f ec_inter_mean=%f\n" % (
                    epoch, loss_tot, hm_intra_mean, hm_inter_mean, ec_intra_mean, ec_inter_mean))

        checkpoint = {'CNN_model_par': CNN_net.state_dict(),
                      'DNN1_model_par': DNN1_net.state_dict(),
                      }
        torch.save(checkpoint, output_folder + '/model.pkl')

    else:
        print("\n Epoch %i, loss_tr=%f" % (epoch, loss_tot))
