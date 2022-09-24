import numpy as np
import torch
from torch.autograd import Variable
import soundfile as sf
import sys
import random
import matplotlib.pyplot as plt
import seaborn as sns

def compute_euclidean_dist(x1, x2, dvecdims):
    x1 = x1.reshape(-1,dvecdims)
    x2 = x2.reshape(-1,dvecdims)
    return np.linalg.norm(x1 - x2, axis=1)


def compute_hamming_dist(x1, x2, dvecdims, ver='thd'):
    # we have to convert them to binary vectors
    # ver1: use threshold 'thd' = 0, ver2: use median
    
    x1 = x1.reshape(-1,dvecdims)
    x2 = x2.reshape(-1,dvecdims)
    
    if ver=='thd':
        b1 = (x1 > 0)
        b2 = (x2 > 0)
    if ver=='med':
        # version median
        median_x1 = np.median(x1, axis=1, keepdims=True)
        median_x2 = np.median(x2, axis=1, keepdims=True)
        b1 = x1 > median_x1
        b2 = x2 > median_x2

    return np.count_nonzero(b1 != b2, axis=1) / dvecdims


def get_list_seq_speaker_by_sample(lab_dict, wav_lst, sample):
    # sample is expected to be a filepath
    label = lab_dict[sample]
    return [key for key, val in lab_dict.items() if (val == label and key in wav_lst)]

def get_list_wav_of_speaker_label(lab_dict):
    wav_lst_spk_dict = {}
    for key, val in lab_dict.items():
        if val in wav_lst_spk_dict:
            wav_lst_spk_dict[val].append(key)
        else:
            wav_lst_spk_dict[val] = [key]
        
    return wav_lst_spk_dict


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


def get_dictkey_by_value(my_dict, value):
    key_list = list(my_dict.keys())
    val_list = list(my_dict.values())
    return key_list[val_list.index(value)]


def access_random_chunk(signal, wlen, filepath):
    snt_len = signal.shape[0]
    snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
    snt_end = snt_beg + wlen

    channels = len(signal.shape)
    if channels == 2:
        print('WARNING: stereo to mono: ' + filepath)
        signal = signal[:, 0]
    return signal[snt_beg:snt_end]

def get_list_rnd(data_folder, wav_lst, N_snt, wlen, lab_dict, fact_amp):
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig = np.zeros([N_snt, wlen])

    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, N_snt)
    
    for i in range(N_snt):
        file_of_label = get_dictkey_by_value(lab_dict, i)
        list_file_of_speaker = (lab_dict, file_of_label)
        random_file = get_positive_file(wav_lst, list_file_of_speaker, file_of_label)

        [signal, fs] = sf.read(data_folder + random_file)

        # accesing to a random chunk
        sig[i, :] = access_random_chunk(signal, wlen, data_folder + random_file) * rand_amp_arr[i]

    sig_array = Variable(torch.from_numpy(sig).float().cuda().contiguous())

    return sig_array

def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict, wav_lst_spk_dict, fact_amp, start_idx):
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    anc_sig_batch = np.zeros([batch_size, wlen])
    pos_sig_batch = np.zeros([batch_size, wlen])
    neg_sig_batch = np.zeros([batch_size, wlen])
    lab_batch = np.zeros(batch_size)

    # snt_id_arr = np.random.randint(N_snt, size=batch_size)

    # Consider the case when batch_size > N_snt
    if (start_idx + batch_size > N_snt):
        wav_lst = wav_lst[start_idx:] + wav_lst[0:start_idx]
        start_idx = 0

    snt_arr = wav_lst[start_idx:start_idx + batch_size]
    start_idx = (start_idx + batch_size) % N_snt

    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

    for i in range(batch_size):  # should be len(snt_arr)
        # select a random sentence from the list
        # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        # signal=signal.astype(float)/32768

#         list_file_of_speaker = get_list_seq_speaker_by_sample(lab_dict, wav_lst, snt_arr[i])
#         pos_file = get_positive_file(wav_lst, list_file_of_speaker, snt_arr[i])
#         neg_file = get_negative_file(wav_lst, list_file_of_speaker)

        pos_file = get_positive_file(wav_lst, wav_lst_spk_dict[lab_dict[snt_arr[i]]], snt_arr[i])
        neg_file = get_negative_file(wav_lst, wav_lst_spk_dict[lab_dict[snt_arr[i]]])

        [anc_sig, anc_fs] = sf.read(data_folder + snt_arr[i])
        [pos_sig, pos_fs] = sf.read(data_folder + pos_file)
        [neg_sig, neg_fs] = sf.read(data_folder + neg_file)

        # accesing to a random chunk
        anc_sig_batch[i, :] = access_random_chunk(anc_sig, wlen, data_folder + snt_arr[i]) * rand_amp_arr[i]
        pos_sig_batch[i, :] = access_random_chunk(pos_sig, wlen, data_folder + pos_file) * rand_amp_arr[i]
        neg_sig_batch[i, :] = access_random_chunk(neg_sig, wlen, data_folder + neg_file) * rand_amp_arr[i]

        # get label of anchor
        label = lab_dict[snt_arr[i]]

        lab_batch[i] = label

    anc = Variable(torch.from_numpy(anc_sig_batch).float().cuda().contiguous())
    pos = Variable(torch.from_numpy(pos_sig_batch).float().cuda().contiguous())
    neg = Variable(torch.from_numpy(neg_sig_batch).float().cuda().contiguous())
    lab = Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())

    return anc, pos, neg, lab, wav_lst, start_idx


def compute_d_vector(signal, wav_te, wlen, wshift, Batch_dev, d_vector_dim, DNN1_net, CNN_net, avoid_small_en_fr=True):
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
        print('nan', wav_te)
        sys.exit(0)

    return d_vect_out.view(1, -1)


def visualize_distribution(intra, inter, label: str, path_export_graph: str, bin_range: float, bin_width: float, xmax: float, ymax: float):
    FONT_SIZE = 80
    plt.rcParams["figure.figsize"] = [70, 35]
    plt.rcParams["figure.autolayout"] = True
    plt.rc('font', size=FONT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE+10)     # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE+10)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=FONT_SIZE+10)
    
    fig, ax = plt.subplots()

    sns.set(style="darkgrid")
    sns.set(font_scale=8)
    sns.histplot(intra, binrange=(0, bin_range), binwidth=bin_width, ax=ax, kde=False, stat='probability',
                 label='intra', color='blue')
    sns.histplot(inter, binrange=(0, bin_range), binwidth=bin_width, ax=ax, kde=False, stat='probability',
                 label='inter', color='orange')

    plt.legend()
    plt.xlabel(label)
    plt.grid(True, linewidth=5)
    
    ax.set_xlim([0, xmax])
    ax.set_ylim([0, ymax])
    ax.tick_params(axis='both', which='major', pad=35)

    plt.savefig(path_export_graph)


# expect size of array is (n, dims) where n is the number of samples and dims is the length of each template.
# k must be lower than the number of files per speaker
def mean_each_kElements(k, array):
    new_arr = array
    
    if array.shape[0]%k!=0:
        new_arr = np.concatenate((array, array[-(k-array.shape[0]%k):]))
    
    grouped_arr = new_arr.reshape(-1, k, array.shape[1])
    return np.mean(grouped_arr, axis=1)


def dump_func(x1, x2, dims, threshhold):
    x1 = x1.reshape(-1, dims)
    x2 = x2.reshape(-1, dims)
    
    dot_product = x1*x2
    
    x1_broadcast = np.broadcast_to(x1, x2.shape)
    
    x1_plus = (dot_product < 0)
    x2_plus = (dot_product >= 0)
    
    mask = (dot_product >= 0)
    
    x1_plus[mask] = (np.abs(x1_broadcast[mask] - x2[mask]) < threshhold)
    x2_plus[mask] = 1
    
    res_x1 = np.concatenate((x1_broadcast > 0, x1_plus), axis=1)
    res_x2 = np.concatenate((x2 > 0, x2_plus), axis=1)
    
    return res_x1, res_x2