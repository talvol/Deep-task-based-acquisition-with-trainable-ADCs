import numpy as np
import h5py
import torch
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform
import matplotlib.pyplot as plt
import pandas as pd


epsilon = 1e-7  # To avoid division by 0 / log of small numbers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plotTime(data, label):
    sample_1 = data[1]
    sample_2 = data[9601]

    Fs = 1e6  # Sampling frequency in Hz
    t = np.arange(len(sample_1)) / Fs * 1000  # Convert to milliseconds

    # Plot the selected samples
    plt.figure(figsize=(10, 7))

    plt.subplot(1, 2, 1)
    plt.plot(t, np.real(sample_1))
    plt.xlabel("Time (ms)", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t, np.real(sample_2))
    plt.xlabel("Time (ms)", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('Radio.png', format='png')
    plt.show()


class LoadDataset:
    def __init__(self):
        self.dataset_name = 'data'
        self.labelset_name = 'label'

    def load_iq_samples(self, file_path_data, file_path_labels):
        # Load the labels
        label_df = pd.read_csv(file_path_labels)

        # Strip and lowercase column names to avoid issues
        label_df.columns = label_df.columns.str.strip().str.lower()

        # Ensure 'label' is correctly referenced
        if 'label' not in label_df.columns:
            raise KeyError("Column 'label' not found in the CSV file")

        # Extract labels
        labels = label_df[self.labelset_name].values.astype(int)

        # Load the IQ samples directly from the numpy file
        data = np.load(file_path_data)
        return data, labels


class ChannelIndSpectrogram():
    def __init__(self):
        pass

    def _normalization(self, data):
        ''' Normalize the signal.'''
        epsilon = 1e-8
        sig_amplitude = data.abs()
        rms = torch.sqrt(torch.mean(sig_amplitude ** 2, dim=1, keepdim=True))
        rms = torch.clamp(rms, min=epsilon)
        s_norm = data / rms
        return s_norm

    def _spec_crop(self, x):
        '''Crop the generated channel independent spectrogram.'''
        num_row = x.shape[1]  # Adjust to handle batch dimension
        # Crop more of the spectrogram, e.g., from 10% to 90%
        x_cropped = x[:, round(num_row * 0.1):round(num_row * 0.9), :]
        return x_cropped

    def _gen_channel_ind_spectrogram_batch(self, sig_batch, win_len=512, overlap=256):
        '''
        _gen_channel_ind_spectrogram_batch converts the IQ samples to channel
        independent spectrograms according to set window and overlap length.

        INPUT:
            SIG_BATCH is a batch of complex IQ samples.

            WIN_LEN is the window length used in STFT.

            OVERLAP is the overlap length used in STFT.

        RETURN:
            CHAN_IND_SPEC_AMP is the generated channel independent spectrogram.
        '''
        assert torch.isfinite(sig_batch).all(), "Input contains NaN or Inf"

        # Use a Hann window for the STFT
        window = torch.hann_window(win_len).to(sig_batch.device).float()

        spec_batch = torch.stft(sig_batch, n_fft=win_len, hop_length=win_len - overlap, win_length=win_len,
                                window=window, return_complex=True, pad_mode='constant', center=False)

        assert not torch.isnan(spec_batch).any(), "NaN in STFT output"
        assert not torch.isinf(spec_batch).any(), "Inf in STFT output"

        # FFT shift to adjust the central frequency
        spec_batch = torch.fft.fftshift(spec_batch, dim=-2)

        # Generate channel independent spectrogram
        epsilon = 1e-8  # Small value to avoid division by zero
        chan_ind_spec_batch = spec_batch[:, :, 1:] / (spec_batch[:, :, :-1] + epsilon)

        assert not torch.isnan(chan_ind_spec_batch).any(), "NaN after division in spectrogram"
        assert not torch.isinf(chan_ind_spec_batch).any(), "Inf after division in spectrogram"

        # Take the logarithm of the magnitude squared
        chan_ind_spec_amp_batch = torch.log10(torch.abs(chan_ind_spec_batch) ** 2 + epsilon)

        assert not torch.isnan(chan_ind_spec_amp_batch).any(), "NaN after log10 in spectrogram"
        assert not torch.isinf(chan_ind_spec_amp_batch).any(), "Inf after log10 in spectrogram"

        return chan_ind_spec_amp_batch

    def channel_ind_spectrogram(self, data):
        '''
        channel_ind_spectrogram converts a batch of IQ samples to channel independent
        spectrograms.

        INPUT:
            DATA is a batch of IQ samples.

        RETURN:
            DATA_CHANNEL_IND_SPEC is channel independent spectrograms for the batch.
        '''
        # Normalize the IQ samples.
        data = self._normalization(data)

        # Convert each packet (IQ samples) to a channel independent spectrogram in batch.
        chan_ind_spec_amp_batch = self._gen_channel_ind_spectrogram_batch(data)
        chan_ind_spec_amp_batch = self._spec_crop(chan_ind_spec_amp_batch)

        # Dynamically calculate num_row after cropping
        num_sample = data.shape[0]
        num_row = chan_ind_spec_amp_batch.shape[1]
        num_column = chan_ind_spec_amp_batch.shape[2]

        # Initialize the output tensor with the correct shape
        data_channel_ind_spec = torch.zeros([num_sample, 1, num_row, num_column], device=data.device)

        # Reshape and assign to output tensor, ensuring the shape is [batch_size, channels, height, width]
        data_channel_ind_spec[:, 0, :, :] = chan_ind_spec_amp_batch
        return data_channel_ind_spec


# def plotCIS_single_signals(data_channel_ind_spec_1, data_channel_ind_spec_2, cmap='inferno'):
#     # Plot the spectrograms for the two signals
#     fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
#
#     # Experiment with different color scale values to enhance visibility
#     axes[0].imshow(data_channel_ind_spec_1.detach().cpu(), aspect='auto', cmap=cmap, vmin=-2, vmax=2)
#     axes[0].axis('off')
#
#     axes[1].imshow(data_channel_ind_spec_2.detach().cpu(), aspect='auto', cmap=cmap, vmin=-2, vmax=2)
#     axes[1].axis('off')
#
#     plt.tight_layout()
#     plt.savefig('CIS_single_signals_adjusted.png', format='png')
#     plt.show()
#
# def plotTime_with_CIS(data):
#     sample_1 = torch.tensor(data[1], dtype=torch.complex64)
#     sample_2 = torch.tensor(data[9601], dtype=torch.complex64)
#
#     # Instantiate your modified CIS class
#     cis = ChannelIndSpectrogram2()
#
#     # Generate CIS for both samples
#     cis_1 = cis.channel_ind_spectrogram(sample_1)
#     cis_2 = cis.channel_ind_spectrogram(sample_2)
#
#     # Plot the generated spectrograms
#     plotCIS_single_signals(cis_1, cis_2)

