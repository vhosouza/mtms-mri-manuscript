"""
processing_tools.py

Author: Victor H. Souza
Date: December 16, 2023

Copyright (c) 2023 Victor H. Souza

Description:
Miscellaneous functions for processing E-field data and signals.

Description of the tools and methods for recording the data are described in the manuscript:

 A multi-channel TMS system enabling accurate stimulus orientation control during concurrent ultra-high-field MRI
 for preclinical applications. Souza et al., bioRxiv 2023.08.10.552401; doi: https://doi.org/10.1101/2023.08.10.552401

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, peak_widths


def lowpass_filter(data, low_cut, fs, order):
    # Get the filter coefficients
    b, a = butter(order, low_cut, btype='low', fs=fs)
    return filtfilt(b, a, data)


def read_efield(filepath):
    pos = np.zeros([1000, 3])
    ef = np.zeros([1000, 3])
    efn = np.zeros([1000, 3])
    enor = []
    k = 0

    content = [s.rstrip() for s in open(filepath)]
    for data in content:
        line = [s for s in data.split()]
        pos[k, ...] = float(line[0]), float(line[1]), float(line[2])
        ef[k, ...] = -float(line[3]), -float(line[4]), -float(line[5])
        # Normalize each vector of E-field, all will have norm 1 to
        # better scaling in quiver plot
        enor.append((np.sqrt(ef[k, 0] ** 2 + ef[k, 1] ** 2 + ef[k, 2] ** 2)).tolist())
        efn[k, ...] = ef[k, ...] / enor[k]
        k += 1

    # Sort all arrays according to crescent Z coordinate
    ids = np.argsort(pos[..., 2])
    ids_v = np.sort(pos[..., 2])
    pos_s = pos[ids[::1], ...]
    ef_s = ef[ids[::1], ...]
    efn_s = efn[ids[::1], ...]

    # Slice second part of coordinates array removing vectors in steps of stp
    ini = 200
    stp = 1

    pos_up = pos_s[:ini, ...]
    pos_down = pos_s[ini::stp, ...]
    ef_up = ef_s[:ini, ...]
    ef_down = ef_s[ini::stp, ...]
    efn_up = efn_s[:ini, ...]
    efn_down = efn_s[ini::stp, ...]

    return np.vstack((pos_up, pos_down)), np.vstack((ef_up, ef_down)), np.vstack((efn_up, efn_down))


def fwhm_efield(x_raw, y_raw, n=20, id_vis=True):
    # spatial sampling frequency to convert the FWHM from index to millimeter
    sampling_freq = x_raw[1] - x_raw[0]

    # make a polynomial fit
    p_mm = np.poly1d(np.polyfit(x_raw, y_raw, n))
    y_fit = p_mm(x_raw)
    # normalize data so that maximum is 1 and minimum as kept the same
    y_fit = y_fit.min() + ((y_fit - y_fit.min())*(1-y_fit.min()))/(y_fit.max() - y_fit.min())

    # use the absolute to provide the correct FWHM, otherwise the baseline is considered the negative values and not zero
    # using just the absolute e-field provides the correct FWHM, as already checked
    y_fit_abs = np.abs(y_fit)
    peaks, _ = find_peaks(y_fit_abs, height=0.5)
    half_widths = peak_widths(y_fit_abs, peaks, rel_height=1-1/np.sqrt(2))

    t1_ind = half_widths[2].round().astype(int)
    t2_ind = half_widths[3].round().astype(int)

    # the default width is given in the indices scale as the function peak_widths does not know what is the x scale
    # xfwhm = half_widths[0]
    # xfwhm = np.abs(x_raw[t1_ind] - x_raw[t2_ind])[0]
    xfwhm = half_widths[0][0]*sampling_freq
    yfwhm = half_widths[1]

    if id_vis:
        fig_fwhm, ax_fwhm = plt.subplots(ncols=1, figsize=(6, 6))
        ax_fwhm.plot(y_raw, 'kx')
        ax_fwhm.plot(y_fit)
        ax_fwhm.plot(peaks, y_fit[peaks], "o", markersize=10)
        ax_fwhm.hlines(*half_widths[1:], color="C2")

        # plot the absolute e-field
        # ax.plot(y_efield_abs, 'r--')
        # ax.plot(peaks_abs, y_efield_abs[peaks_abs], "ko")
        # ax.hlines(*half_widths_abs[1:], color="c")

        fig_fwhm.tight_layout()
        plt.show()

    return xfwhm, yfwhm, t1_ind, t2_ind