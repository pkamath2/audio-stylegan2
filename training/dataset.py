import sys  
import os
import numpy as np
import librosa
import json
import torch

from tifresi.utils import load_signal
from tifresi.utils import preprocess_signal
from tifresi.stft import GaussTF, GaussTruncTF
from tifresi.transforms import log_spectrogram
from tifresi.transforms import inv_log_spectrogram

sys.path.insert(0, '../')
from util import util

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, config_file, use_labels  = False):

        self.name = 'AudioDataset'
        self._data_dir = data_dir
        self._config_file = config_file
        self._use_labels = use_labels #TODO: Need to integrate conditioning

        self._config = util.get_config(self._config_file)

        self._raw_labels = None
        self._all_fnames = sorted({os.path.relpath(os.path.join(root, fname), start=self._data_dir) for root, _dirs, files in os.walk(self._data_dir) for fname in files})
        self._idx = np.arange(len(self._all_fnames))
        self._stft_channels = self._config['stft_channels']#512 
        self._n_frames = self._config['n_frames']#256
        self._hop_size = self._config['hop_size']#128
        self._sample_rate = self._config['sample_rate']#16000
        self._raw_shape = [len(self._all_fnames)] + list(self._load_raw_image(0).shape)
    

    def _zeropad(self, signal, audio_length):
        if len(signal) < audio_length:
            return np.append(
                signal, 
                np.zeros(audio_length - len(signal))
            )
        else:
            signal = signal[0:audio_length]
            return signal

    
    def _pghi_stft(self, x, use_truncated_window):
        if use_truncated_window:
            stft_system = GaussTruncTF(hop_size=self._hop_size, stft_channels=self._stft_channels)
        else:
            stft_system = GaussTF(hop_size=self._hop_size, stft_channels=self._stft_channels)
        Y = stft_system.spectrogram(x)
        log_Y= log_spectrogram(Y)
        return np.expand_dims(log_Y,axis=0)

    
    def _load_raw_image(self, idx):

        fname = self._all_fnames[idx]
        with open(os.path.join(self._data_dir, fname), 'rb')as f:
            y, sr = load_signal(f, sr=16000)
            y = preprocess_signal(y)
            y = self._zeropad(y, self._n_frames * self._hop_size ) #Check if you need this. Also ensure that spectrograms are eventually square.
            y = self._pghi_stft(y, use_truncated_window=True)
            y = util.renormalize(y, (-50, 0), (0, 255)) # rescale to 0-255 like RGB images
            y = y.astype(np.uint8)

            # Shape here is 1 X 257 X 256
            ### WHOA! What!!!! - Did this to get the Spectrogram to 1X256 X 256   
            y = y[:,:256,:] 
            # Shape here is 1 X 256 X 256

        return y

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
    
    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def get_label(self, idx):
        label = self._get_raw_labels()[self._idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def __getitem__(self, idx):
        image = self._load_raw_image(self._idx[idx])
        return image.copy(), self.get_label(idx)

    def __len__(self):
        return self._idx.size

    @property
    def has_labels(self):
        return self._use_labels
    
    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def resolution(self):
        return self._raw_shape[2] #_raw_shape is N (num of files) X 1 X 256 X 256 (Last two are Spectrogram Shape)
    