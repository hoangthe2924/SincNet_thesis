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
import random
from dnn_models import MLP, flip
from dnn_models import SincNet as CNN
from data_io import ReadList, read_conf, str_to_bool
from loss_function import TripletLossWithHammingDistance as TLHD
from loss_function import TripletLoss as TL


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
            dvects[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN1_net(CNN_net(inp))
            count_fr = 0
            sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()
    

    if count_fr > 0:
        inp = Variable(sig_arr[0:count_fr])
        dvects[count_fr_tot - count_fr:count_fr_tot, :] = DNN1_net(CNN_net(inp))
    
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

# [class]
class_lay = list(map(int, options.class_lay.split(',')))
class_drop = list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act = list(map(str, options.class_act.split(',')))

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
cost = TL()
# cost = TLHD()

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

DNN2_arch = {'input_dim': fc_lay[-1],
             'fc_lay': class_lay,
             'fc_drop': class_drop,
             'fc_use_batchnorm': class_use_batchnorm,
             'fc_use_laynorm': class_use_laynorm,
             'fc_use_laynorm_inp': class_use_laynorm_inp,
             'fc_use_batchnorm_inp': class_use_batchnorm_inp,
             'fc_act': class_act,
             }

DNN2_net = MLP(DNN2_arch)
DNN2_net.cuda()

d_vector_dim=fc_lay[-1]

if pt_file != 'none':
    checkpoint_load = torch.load(pt_file)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
#     DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)

print('Traning...')
for epoch in range(N_epochs):
    print('Start epoch: ', epoch)
    test_flag = 0
    CNN_net.train()
    DNN1_net.train()
    # DNN2_net.train()

    loss_sum = 0

    for i in range(N_batches):
        [anc, pos, neg, lab] = create_batches_rnd(batch_size, data_folder, wav_lst_tr, snt_tr, wlen, lab_dict, 0.2)
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
        print("-Batch %d, loss: %f" % (i, loss))

    loss_tot = loss_sum / N_batches

    # Full Validation  new
    if epoch % N_eval_epoch == 0:

        CNN_net.eval()
        DNN1_net.eval()
        test_flag = 1
        loss_sum = 0
        get_sample = False
        inner_eval_sum = 0
        outer_eval_sum = 0

        with torch.no_grad():
            for i in range(snt_te):
#             for i in range(10):

                # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst_te[i])
                # signal=signal.astype(float)/32768

                list_file_of_speaker_te = get_list_seq_speaker_by_sample(lab_dict, wav_lst_te[i])
                pos_file = get_positive_file(wav_lst_te, list_file_of_speaker_te, wav_lst_te[i])
                neg_file = get_negative_file(wav_lst_te, list_file_of_speaker_te)

                [anc_sig, anc_fs] = sf.read(data_folder + wav_lst_te[i])
                [pos_sig, pos_fs] = sf.read(data_folder + pos_file)
                [neg_sig, neg_fs] = sf.read(data_folder + neg_file)

                anc_pout = compute_d_vector(anc_sig, wlen, wshift, Batch_dev, d_vector_dim, DNN1_net, CNN_net)
                pos_pout = compute_d_vector(pos_sig, wlen, wshift, Batch_dev, d_vector_dim, DNN1_net, CNN_net)
                neg_pout = compute_d_vector(neg_sig, wlen, wshift, Batch_dev, d_vector_dim, DNN1_net, CNN_net)
                
                if(get_sample):
                    print('sample_embedding: ', anc_pout)
                    print(torch.max(anc_pout))
                    get_sample = False

                inner_eval_sum = inner_eval_sum + torch.abs((torch.sgn(anc_pout) - torch.sgn(pos_pout))).sum()/2/d_vector_dim
                outer_eval_sum = outer_eval_sum + torch.abs((torch.sgn(anc_pout) - torch.sgn(neg_pout))).sum()/2/d_vector_dim
                
                loss = cost(anc_pout, pos_pout, neg_pout)
                loss_sum = loss_sum + loss.detach()

            loss_tot_dev = loss_sum / snt_te

        print("epoch %i, loss_tr=%f loss_te=%f, outer_err=%f, inner_err=%f" % (
            epoch, loss_tot, loss_tot_dev, outer_eval_sum / snt_te, inner_eval_sum / snt_te))

        with open(output_folder + "/res.res", "a") as res_file:
            res_file.write("epoch %i, loss_tr=%f loss_te=%f, outer_err=%f, inner_err=%f \n" % (
                epoch, loss_tot, loss_tot_dev, outer_eval_sum / snt_te, inner_eval_sum / snt_te))

        checkpoint = {'CNN_model_par': CNN_net.state_dict(),
                      'DNN1_model_par': DNN1_net.state_dict(),
                      }
        torch.save(checkpoint, output_folder + '/new_model_tanh.pkl')

    else:
        print("epoch %i, loss_tr=%f" % (epoch, loss_tot))

    # # Full Validation  new
    # if epoch % N_eval_epoch == 0:
    #
    #     CNN_net.eval()
    #     DNN1_net.eval()
    #     DNN2_net.eval()
    #     test_flag = 1
    #     loss_sum = 0
    #     err_sum = 0
    #     err_sum_snt = 0
    #
    #     with torch.no_grad():
    #         for i in range(snt_te):
    #
    #             # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst_te[i])
    #             # signal=signal.astype(float)/32768
    #
    #             [signal, fs] = sf.read(data_folder + wav_lst_te[i])
    #
    #             signal = torch.from_numpy(signal).float().cuda().contiguous()
    #             lab_batch = lab_dict[wav_lst_te[i]]
    #
    #             # split signals into chunks
    #             beg_samp = 0
    #             end_samp = wlen
    #
    #             N_fr = int((signal.shape[0] - wlen) / (wshift))
    #
    #             sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()
    #             lab = Variable((torch.zeros(N_fr + 1) + lab_batch).cuda().contiguous().long())
    #             pout = Variable(torch.zeros(N_fr + 1, class_lay[-1]).float().cuda().contiguous())
    #             count_fr = 0
    #             count_fr_tot = 0
    #             while end_samp < signal.shape[0]:
    #                 sig_arr[count_fr, :] = signal[beg_samp:end_samp]
    #                 beg_samp = beg_samp + wshift
    #                 end_samp = beg_samp + wlen
    #                 count_fr = count_fr + 1
    #                 count_fr_tot = count_fr_tot + 1
    #                 if count_fr == Batch_dev:
    #                     inp = Variable(sig_arr)
    #                     pout[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))
    #                     count_fr = 0
    #                     sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()
    #
    #             if count_fr > 0:
    #                 inp = Variable(sig_arr[0:count_fr])
    #                 pout[count_fr_tot - count_fr:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))
    #
    #             pred = torch.max(pout, dim=1)[1]
    #             loss = cost(pout, lab.long())
    #             err = torch.mean((pred != lab.long()).float())
    #
    #             [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
    #             err_sum_snt = err_sum_snt + (best_class != lab[0]).float()
    #
    #             loss_sum = loss_sum + loss.detach()
    #             err_sum = err_sum + err.detach()
    #
    #         err_tot_dev_snt = err_sum_snt / snt_te
    #         loss_tot_dev = loss_sum / snt_te
    #         err_tot_dev = err_sum / snt_te
    #
    #     print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (
    #     epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))
    #
    #     with open(output_folder + "/res.res", "a") as res_file:
    #         res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (
    #         epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))
    #
    #     checkpoint = {'CNN_model_par': CNN_net.state_dict(),
    #                   'DNN1_model_par': DNN1_net.state_dict(),
    #                   'DNN2_model_par': DNN2_net.state_dict(),
    #                   }
    #     torch.save(checkpoint, output_folder + '/model_raw.pkl')
    #
    # else:
    #     print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))


