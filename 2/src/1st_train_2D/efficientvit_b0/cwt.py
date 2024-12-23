import numpy as np

import torch
import torch.nn as nn

class CWT(nn.Module):
    def __init__(
        self,
        wavelet_width,
        fs,
        lower_freq,
        upper_freq,
        n_scales,
        size_factor=1.0,
        border_crop=0,
        stride=1
    ):
        super().__init__()

        self.initial_wavelet_width = wavelet_width
        self.fs = fs
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq
        self.size_factor = size_factor
        self.n_scales = n_scales
        self.wavelet_width = wavelet_width
        self.border_crop = border_crop
        self.stride = stride
        wavelet_bank_real, wavelet_bank_imag = self._build_wavelet_kernel()
        self.wavelet_bank_real = nn.Parameter(wavelet_bank_real, requires_grad=False)
        self.wavelet_bank_imag = nn.Parameter(wavelet_bank_imag, requires_grad=False)

        self.kernel_size = self.wavelet_bank_real.size(3)

    def _build_wavelet_kernel(self):
        s_0 = 1 / self.upper_freq
        s_n = 1 / self.lower_freq

        base = np.power(s_n / s_0, 1 / (self.n_scales - 1))
        scales = s_0 * np.power(base, np.arange(self.n_scales))

        frequencies = 1 / scales
        truncation_size = scales.max() * np.sqrt(4.5 * self.initial_wavelet_width) * self.fs
        one_side = int(self.size_factor * truncation_size)
        kernel_size = 2 * one_side + 1

        k_array = np.arange(kernel_size, dtype=np.float32) - one_side
        t_array = k_array / self.fs

        wavelet_bank_real = []
        wavelet_bank_imag = []

        for scale in scales:
            norm_constant = np.sqrt(np.pi * self.wavelet_width) * scale * self.fs / 2.0
            scaled_t = t_array / scale
            exp_term = np.exp(-(scaled_t ** 2) / self.wavelet_width)
            kernel_base = exp_term / norm_constant
            kernel_real = kernel_base * np.cos(2 * np.pi * scaled_t)
            kernel_imag = kernel_base * np.sin(2 * np.pi * scaled_t)
            wavelet_bank_real.append(kernel_real)
            wavelet_bank_imag.append(kernel_imag)

        wavelet_bank_real = np.stack(wavelet_bank_real, axis=0)
        wavelet_bank_imag = np.stack(wavelet_bank_imag, axis=0)

        wavelet_bank_real = torch.from_numpy(wavelet_bank_real).unsqueeze(1).unsqueeze(2)
        wavelet_bank_imag = torch.from_numpy(wavelet_bank_imag).unsqueeze(1).unsqueeze(2)
        return wavelet_bank_real, wavelet_bank_imag

    def forward(self, x):
        border_crop = self.border_crop // self.stride
        start = border_crop
        end = (-border_crop) if border_crop > 0 else None

        # x [n_batch, n_channels, time_len]
        out_reals = []
        out_imags = []

        in_width = x.size(2)
        out_width = int(np.ceil(in_width / self.stride))
        pad_along_width = np.max((out_width - 1) * self.stride + self.kernel_size - in_width, 0)
        padding = pad_along_width // 2 + 1

        for i in range(x.size(1)):
            # [n_batch, 1, 1, time_len]
            x_ = x[:, i, :].unsqueeze(1).unsqueeze(2)
            out_real = nn.functional.conv2d(x_, self.wavelet_bank_real, stride=(1, self.stride), padding=(0, padding))
            out_imag = nn.functional.conv2d(x_, self.wavelet_bank_imag, stride=(1, self.stride), padding=(0, padding))
            out_real = out_real.transpose(2, 1)
            out_imag = out_imag.transpose(2, 1)
            out_reals.append(out_real)
            out_imags.append(out_imag)

        out_real = torch.cat(out_reals, axis=1)
        out_imag = torch.cat(out_imags, axis=1)

        out_real = out_real[:, :, :, start:end]
        out_imag = out_imag[:, :, :, start:end]

        scalograms = torch.sqrt(out_real ** 2 + out_imag ** 2)
        return scalograms