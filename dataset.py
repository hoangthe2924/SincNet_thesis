import torch
import numpy as np
import soundfile as sf
from torch.autograd import Variable
from torch.utils.data import Dataset
import utils

class CustomDataset(Dataset):
    def __init__(self, datafolder, lst_wav, dic_label, fact_amp, wlen, transform=None):
        self.datafolder = datafolder
        self.lst_wav = lst_wav
        self.dic_labels = dic_label
        self.fact_amp = fact_amp
        self.wlen = wlen
        self.dic_lbl_files = utils.get_list_wav_of_speaker_label(dic_label)
        
    def __len__(self):
        return len(self.lst_wav)

    def access_random_chunk(self, signal, wlen, filepath):
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
        snt_end = snt_beg + wlen

        channels = len(signal.shape)
        if channels == 2:
            print('WARNING: stereo to mono: ' + filepath)
            signal = signal[:, 0]
        return signal[snt_beg:snt_end]
    
    def __getitem__(self, item):
        rand_amp = np.random.uniform(1.0 - self.fact_amp, 1 + self.fact_amp)

        anc_file = self.lst_wav[item]
        label = self.dic_labels[anc_file]

        pos_file = utils.get_positive_file(self.lst_wav, self.dic_lbl_files[label], anc_file)
        neg_file = utils.get_negative_file(self.lst_wav, self.dic_lbl_files[label])

        [anc_sig, anc_fs] = sf.read(self.datafolder + anc_file)
        [pos_sig, pos_fs] = sf.read(self.datafolder + pos_file)
        [neg_sig, neg_fs] = sf.read(self.datafolder + neg_file)

        anc_sig_aug = self.access_random_chunk(anc_sig, self.wlen, anc_file)*rand_amp
        pos_sig_aug = self.access_random_chunk(pos_sig, self.wlen, pos_file)*rand_amp
        neg_sig_aug = self.access_random_chunk(neg_sig, self.wlen, neg_file)*rand_amp

        anc = torch.from_numpy(anc_sig_aug).float()
        pos = torch.from_numpy(pos_sig_aug).float()
        neg = torch.from_numpy(neg_sig_aug).float()

        return anc, pos, neg, label

        