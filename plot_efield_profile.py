"""
plot_efield_profile.py

Author: Victor H. Souza
Date: December 16, 2023

Copyright (c) 2023 Victor H. Souza

Description:
This script reads and analyzes electric field and current waveforms from measurements of the 2-coil mTMS transducer.

Description of the tools and methods for recording the data are described in the manuscript:

 A multi-channel TMS system enabling accurate stimulus orientation control during concurrent ultra-high-field MRI
 for preclinical applications. Souza et al., bioRxiv 2023.08.10.552401; doi: https://doi.org/10.1101/2023.08.10.552401

The script is designed to:
1. Read measurement data from CSV files.
2. Compute the full-width at half-maximum from the efield profile
3. Plot the electric field profiles.

Usage:
- Create a .env file in the current working directory
- Ensure that the relative file paths are correctly entered.
- Create a Python environment with the required libraries specified in the environment.yml file

"""

import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import processing_tools as pt
import settings as stg

# redefining matplotlib default parameters
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['font.size'] = 8
mpl.rcParams['font.weight'] = 'light'
mpl.rcParams['axes.linewidth'] = 1.0
leg_font_size = 7

mm2in = lambda mm: (mm/2.54)*0.1
deg_sign = u'\xb0'
en_dash = u'\u2013'
line_list = ['--', '-']

# %
# show and save the plot
id_show, id_save = True, False

# figure dimensions as width and height
fig_ratio, fig_res, fig_format = 1.6, 600, 'png'
wid = 60
hei = wid*fig_ratio
fig_dim = (mm2in(wid), mm2in(hei))

directions = ('parallel', 'perpendicular')

# path for data files
data_dir = stg.DIR_EFIELD_CURRENT
plot_dir = stg.DIR_SAVE_PLOT

# %%
color_bot = '#0072BD'  # blue
color_top = '#D95319'  # orange

xticks_list = [-100, -50, 0, 50, 100]
yticks_list = [0, 0.5, 1.0]

fig, axs = plt.subplots(ncols=1, nrows=2, figsize=fig_dim)

# n to plot: 0 is parallel and 1 is perpendicular
for n, ax in enumerate(axs):
    filename = os.path.join(data_dir, 'efield_profile_20Vm_{}.csv'.format(directions[n]))
    data = pd.read_csv(filename)

    samp_freq = 1e3 / (data.x_mm[1] - data.x_mm[0])
    cutoff = 50
    data['efield_top_filt'] = pt.lowpass_filter(data.efield_top.values, cutoff, samp_freq, order=2)
    data['efield_bottom_filt'] = pt.lowpass_filter(data.efield_bottom.values, cutoff, samp_freq, order=2)

    # rescale from minimum norm to 1 because the lowpass filter might reduce the normalized peak to below 1
    y_filt = data['efield_top_filt']
    data['efield_top_filt'] = y_filt.min() + ((y_filt - y_filt.min()) * (1 - y_filt.min())) / (
                y_filt.max() - y_filt.min())
    y_filt = data['efield_bottom_filt']
    data['efield_bottom_filt'] = y_filt.min() + ((y_filt - y_filt.min()) * (1 - y_filt.min())) / (
                y_filt.max() - y_filt.min())

    pl_top = ax.plot(data.x_mm, data.efield_top_filt, linestyle=line_list[n], color=color_top,
                     linewidth=1.5, zorder=2, label='Top')
    pl_bot = ax.plot(data.x_mm, data.efield_bottom_filt, linestyle=line_list[n], color=color_bot,
                     linewidth=1.5, zorder=2, label='Bottom')

    y_lim_perp = (-0.30, 1.06)
    ax.set_ylim(y_lim_perp)
    ax.set_yticks(yticks_list)
    ax.set_xticks(xticks_list)
    ax.set_yticklabels(['0', ' ', '1'], rotation=0)
    ax.tick_params(width=1.5, length=3., direction='in', top=False, right=False)
    ax.xaxis.grid(True, zorder=1, color=3*[0.9], linewidth=1.)

    if n:
        ax.set_ylabel('Normalized E-field', labelpad=2)
        ax.set_xlabel('Distance (mm)', labelpad=2)

        # plot legend
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles, labels, loc='upper right', facecolor='w', framealpha=1.0,
                        fontsize=leg_font_size, handlelength=1., ncol=1, handletextpad=0.5,
                        bbox_to_anchor=(1., 0.98))
    else:
        ax.set_xticklabels(5*[' '], rotation=0)

    ax.set_title('{}'.format(directions[n].capitalize()), pad=2, fontsize=8)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # compute the focality as FWHM
    bottom_focality = pt.fwhm_efield(data.x_mm.values, data.efield_bottom.values, id_vis=False)
    top_focality = pt.fwhm_efield(data.x_mm.values, data.efield_top.values, id_vis=False)

    print('Bottom coil {} focality as FWHM {:.2f} mm'.format(directions[n].capitalize(), bottom_focality[0]))
    print('Top coil {} focality as FWHM {:.2f} mm'.format(directions[n].capitalize(), top_focality[0]))

# show and/or save figure
if id_show:
    fig.tight_layout()
    plt.show()

# %
if id_save:
    filename_png = os.path.join(plot_dir, 'efield_profile.{}'.format(fig_format))
    fig.savefig(filename_png, dpi=fig_res, bbox_inches='tight', pad_inches=0.01, transparent=False)
