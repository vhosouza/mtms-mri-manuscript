"""
plot_efield_current_waveform.py

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
2. Process the data to filter and scale electric field and current waveforms.
3. Plot the electric field and current waveforms for visualization.

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
minus_sign = u'\u2212'
line_list = ['-', '-']

# %%
# show and save the plot
id_show, id_save = True, False

# figure dimensions as width and height
fig_ratio, fig_res, fig_format = 1.6, 600, 'png'
wid = 60
hei = wid*fig_ratio
fig_dim = (mm2in(wid), mm2in(hei))

# path for data files
data_dir = stg.DIR_EFIELD_CURRENT
plot_dir = stg.DIR_SAVE_PLOT
filename_bottom = os.path.join(data_dir, 'current_waveform_nautilus_20Vm_monophasic_bottom.csv')
filename_top = os.path.join(data_dir, 'current_waveform_nautilus_20Vm_monophasic_top.csv')

data_bottom = pd.read_csv(filename_bottom, delimiter=',', skiprows=lambda x: x in [1, 2])
data_top = pd.read_csv(filename_top, delimiter=',', skiprows=lambda x: x in [1, 2])

data_bottom.rename(columns={'x-axis': 'time', '1': 'current', '2': 'efield', '4': 'trigger'}, inplace=True)
data_top.rename(columns={'x-axis': 'time', '1': 'current', '2': 'efield', '4': 'trigger'}, inplace=True)

samp_freq = 1/(data_bottom.time[1] - data_bottom.time[0])
cutoff = 1e6
rogowski_scale = (1/.5e-3)/1e3  # convert directly to kA
time_micros = 1e6

# scale the e-field to V/m based on the first 60 us
# there is 100 us delay from the mtms gate out to the pulse
# measurement was done with 20 V/m
epoch_id = [int(n*samp_freq/time_micros) for n in [112, 162]]
efield_epoch = data_bottom.efield.values[epoch_id[0]:epoch_id[1]]
efield_vm_scale_bot = 20/efield_epoch.mean()
efield_epoch = data_top.efield.values[epoch_id[0]:epoch_id[1]]
efield_vm_scale_top = 20/efield_epoch.mean()

# scale measurements:
# time -> microseconds
# efield -> Volts/meters
# current -> kiloAmperes

data_bottom.time = data_bottom.time*time_micros
data_bottom['efield_vm'] = data_bottom.efield*efield_vm_scale_bot
data_bottom.current = data_bottom.current*rogowski_scale

data_top.time = data_top.time*time_micros
data_top['efield_vm'] = data_top.efield*efield_vm_scale_top
data_top.current = data_top.current*rogowski_scale

# %
# apply low pass fielter to measurements for smoother visualization
data_bottom['current_filt'] = pt.lowpass_filter(data_bottom.current.values, cutoff, samp_freq, order=2)
data_bottom['efield_filt'] = pt.lowpass_filter(data_bottom.efield_vm.values, cutoff, samp_freq, order=2)
data_top['current_filt'] = pt.lowpass_filter(data_top.current.values, cutoff, samp_freq, order=2)
data_top['efield_filt'] = pt.lowpass_filter(data_top.efield_vm.values, cutoff, samp_freq, order=2)

# %
color_bot = '#0072BD'  # blue
color_top = '#D95319'  # orange

x_bot = data_bottom.time.values
x_top = data_top.time.values

fig, axs = plt.subplots(ncols=1, nrows=2, figsize=fig_dim)
xticks_list = [10, 70, 100, 140]

# n to plot: 0 is current and 1 is efield
for n, ax in enumerate(axs):

    if n:
        yticks_list = [-20, 0, 20]
        yticks_labels = ['{}20'.format(minus_sign), '0', ' 20']
        xticks_labels = ['0', '60', '90', '130']
        y_bot = data_bottom.efield_filt.values
        y_top = data_top.efield_filt.values

        ax.set_ylabel('E-field (V/m)', labelpad=2)
        ax.set_xlabel('Time (µs)', labelpad=2)
        ax.set_ylim([-65, 50])

    else:
        yticks_list = [0, 0.5, 1.]
        yticks_labels = yticks_list
        xticks_labels = 4*[' ']
        # current on bottom coil was negative
        y_bot = -data_bottom.current_filt.values
        y_top = data_top.current_filt.values

        ax.set_ylabel('Current (kA)', labelpad=2)

    pl_top = ax.plot(x_top, y_top, linestyle=line_list[n], color=color_top,
                 linewidth=1.5, zorder=2, label='Top')
    pl_bot = ax.plot(x_bot, y_bot, linestyle=line_list[n], color=color_bot,
                 linewidth=1.5, zorder=2, label='Bottom')

    # customize plot aesthetics
    ax.set_xticks(xticks_list)
    ax.set_xticklabels(xticks_labels, rotation=0)
    ax.set_xlim([0, 160])

    ax.set_yticks(yticks_list)
    ax.set_yticklabels(yticks_labels, rotation=0)

    ax.tick_params(width=1.5, length=3., direction='in', top=False, right=False)
    ax.xaxis.grid(True, zorder=1, color=3*[0.9], linewidth=1.)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plot legend only in second subplot
    if n:
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles, labels, loc='upper right', facecolor='w', framealpha=1,
                        fontsize=leg_font_size, handlelength=.7, ncol=1, handletextpad=0.5)


# show and/or save figure
if id_show:
    fig.tight_layout()
    plt.show()

# %
if id_save:
    filename_png = os.path.join(plot_dir, 'efield_current_waveform.{}'.format(fig_format))
    fig.savefig(filename_png, dpi=fig_res, bbox_inches='tight', pad_inches=0.01, transparent=False)
