import os, random, torch, librosa, numpy as np, torchaudio
from scipy.signal import fftconvolve
from python_speech_features import sigproc
from torch.utils.data import Dataset
from torchaudio.sox_effects import apply_effects_tensor

class WavDataset(Dataset):
    def __init__(self, wav_scp, utt2label=None,
                 fs=16000, preemph=0.97, channel=None,
                 is_aug=False, snr=None, noise_list=None):
        self.wav_scp = wav_scp
        self.utt2label = utt2label
        self.channel = channel
        
        self.fs = fs
        self.preemph = preemph

        self.is_aug = is_aug
        self.noise_list = noise_list
        self.snr = snr

    def __len__(self):
        return len(self.wav_scp)

    def _load_data(self, filename):
        signal, fs = librosa.load(filename, sr=self.fs)
        if fs != self.fs:
            signal, fs = librosa.load(filename, sr=self.fs)
        if len(signal.shape) == 2 and self.channel:
            channel = random.choice(self.channel) if type(self.channel) == list else self.channel
            return signal[:, channel]
        return signal
    
    def _norm_speech(self, signal):
        if np.std(signal) == 0:
            return signal
        signal = (signal - np.mean(signal)) / np.std(signal)
        return signal

    def _augmentation(self, signal, filename):
        signal = self._norm_speech(signal)
        
        noise_types = random.choice(['reverb', 'sox', 'noise'])
        
        if noise_types == 'sox':
            effect = random.choice(['tempo', 'vol'])
            if effect == 'tempo':
                effects = [['tempo', str(random.choice([0.9,1.1]))]]
                signal_sox, sample_rate = apply_effects_tensor(torch.tensor(signal).unsqueeze(dim=0), self.fs, effects, channels_first=True)
            elif effect == 'vol':
                effects = [['vol', str(random.random() * 15 + 5)]]
                signal_sox, sample_rate = apply_effects_tensor(torch.tensor(signal).unsqueeze(dim=0), self.fs, effects, channels_first=True)
            
            return self._truncate_speech(signal_sox.numpy()[0], len(signal))
        
        elif noise_types == 'reverb':
            rir = self._norm_speech(self._load_data(random.choice(self.noise_list[noise_types])))
            return fftconvolve(rir, signal)[0 : signal.shape[0]]
        
        else:
            noise_signal = np.zeros(signal.shape[0], dtype='float32')
            for noise_type in random.choice([['noise'], ['music'], ['babb', 'music'], ['babb'] * random.randint(3, 8)]):
                noise = self._load_data(random.choice(self.noise_list[noise_type]))
                noise = self._truncate_speech(noise, signal.shape[0])
                noise_signal = noise_signal + self._norm_speech(noise)
            snr = random.uniform(self.snr[0], self.snr[1])
            sigma_n = np.sqrt(10 ** (- snr / 10))
            return signal + self._norm_speech(noise_signal) * sigma_n

    def _truncate_speech(self, signal, tlen, offset=None):
        if tlen == None:
            return signal
        if signal.shape[0] <= tlen:
            signal = np.concatenate([signal] * (tlen // signal.shape[0] + 1), axis=0)
        if offset == None:
            offset = random.randint(0, signal.shape[0] - tlen)
        return np.array(signal[offset : offset + tlen])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            tlen = None
        elif len(idx) == 2:
            idx, tlen = idx
            tlen = int(tlen * self.fs)
        else:
            raise AssertionError("The idx should be int or a list with length of 2.")

        utt, filename = self.wav_scp[idx]
        signal = self._load_data(filename)
        
        offset = None if self.utt2label else 0
        signal = self._truncate_speech(signal, tlen, offset)
        
        if self.utt2label and self.is_aug and random.choice([0, 1, 1]):
            # only do data augmentation at training (with utt2label)
            # 2/3 data augmentation; 1/3 clean data
            signal = self._augmentation(signal, filename)
        
        signal = self._norm_speech(signal)
        signal = sigproc.preemphasis(signal, self.preemph)
        signal = torch.from_numpy(signal.astype('float32'))
        
        if self.utt2label:
            return signal, self.utt2label[utt]
        else:
            return signal, utt
        
class WavDataset_v2(Dataset):
    def __init__(self, wav_scp, utt2label=None, utt2domain=None,
                 fs=16000, preemph=0.97, channel=None,
                 is_aug=False, snr=None, noise_list=None):
        self.wav_scp = wav_scp
        self.utt2label = utt2label
        self.utt2domain = utt2domain
        self.channel = channel
        
        self.fs = fs
        self.preemph = preemph

        self.is_aug = is_aug
        self.noise_list = noise_list
        self.snr = snr

    def __len__(self):
        return len(self.wav_scp)

    def _load_data(self, filename):
        signal, fs = librosa.load(filename, sr=self.fs)
        if fs != self.fs:
            signal, fs = librosa.load(filename, sr=self.fs)
        if len(signal.shape) == 2 and self.channel:
            channel = random.choice(self.channel) if type(self.channel) == list else self.channel
            return signal[:, channel]
        return signal
    
    def _norm_speech(self, signal):
        if np.std(signal) == 0:
            return signal
        signal = (signal - np.mean(signal)) / np.std(signal)
        return signal

    def _augmentation(self, signal, filename):
        signal = self._norm_speech(signal)
        
        noise_types = random.choice(['reverb', 'sox', 'noise'])
        
        if noise_types == 'sox':
            effect = random.choice(['tempo', 'vol'])
            if effect == 'tempo':
                effects = [['tempo', str(random.choice([0.9,1.1]))]]
                signal_sox, sample_rate = apply_effects_tensor(torch.tensor(signal).unsqueeze(dim=0), self.fs, effects, channels_first=True)
            elif effect == 'vol':
                effects = [['vol', str(random.random() * 15 + 5)]]
                signal_sox, sample_rate = apply_effects_tensor(torch.tensor(signal).unsqueeze(dim=0), self.fs, effects, channels_first=True)
            
            return self._truncate_speech(signal_sox.numpy()[0], len(signal))
        
        elif noise_types == 'reverb':
            rir = self._norm_speech(self._load_data(random.choice(self.noise_list[noise_types])))
            return fftconvolve(rir, signal)[0 : signal.shape[0]]
        
        else:
            noise_signal = np.zeros(signal.shape[0], dtype='float32')
            for noise_type in random.choice([['noise'], ['music'], ['babb', 'music'], ['babb'] * random.randint(3, 8)]):
                noise = self._load_data(random.choice(self.noise_list[noise_type]))
                noise = self._truncate_speech(noise, signal.shape[0])
                noise_signal = noise_signal + self._norm_speech(noise)
            snr = random.uniform(self.snr[0], self.snr[1])
            sigma_n = np.sqrt(10 ** (- snr / 10))
            return signal + self._norm_speech(noise_signal) * sigma_n

    def _truncate_speech(self, signal, tlen, offset=None):
        if tlen == None:
            return signal
        if signal.shape[0] <= tlen:
            signal = np.concatenate([signal] * (tlen // signal.shape[0] + 1), axis=0)
        if offset == None:
            offset = random.randint(0, signal.shape[0] - tlen)
        return np.array(signal[offset : offset + tlen])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            tlen = None
        elif len(idx) == 2:
            idx, tlen = idx
            tlen = int(tlen * self.fs)
        else:
            raise AssertionError("The idx should be int or a list with length of 2.")

        utt, filename = self.wav_scp[idx]
        signal = self._load_data(filename)
        
        offset = None if self.utt2label else 0
        signal = self._truncate_speech(signal, tlen, offset)
        
        if self.utt2label and self.is_aug and random.choice([0, 1, 1]):
            # only do data augmentation at training (with utt2label)
            # 2/3 data augmentation; 1/3 clean data
            signal = self._augmentation(signal, filename)
        
        signal = self._norm_speech(signal)
        signal = sigproc.preemphasis(signal, self.preemph)
        signal = torch.from_numpy(signal.astype('float32'))
        
        if self.utt2label:
            return signal, self.utt2label[utt], self.utt2domain[utt]
        else:
            return signal, utt