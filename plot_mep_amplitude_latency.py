"""
plot_mep_amplitude_latency.py

Author: Victor H. Souza
Date: December 16, 2023

Copyright (c) 2023 Victor H. Souza

Description:
This script reads and analyzes the MEP amplitude and latency from the EMG recordings with rats.
Plot: MEP x orientation x stimulation side
Two brain hemispheres (left, right) and two paws (hands) (left, right)
Eight orientation: 0 - 315 in steps of 45degs.
The original order is: A (0deg) -> RA (45deg) -> R (90deg)
Matlab data columns: 1. pulse_number latency(coord), 2. latency(ms), 3. y-coord(nothing important)
                     4. input(always 1), 5. peak-to-peak

Description of the tools and methods for recording the data are described in the manuscript:

 A multi-channel TMS system enabling accurate stimulus orientation control during concurrent ultra-high-field MRI
 for preclinical applications. Souza et al., bioRxiv 2023.08.10.552401; doi: https://doi.org/10.1101/2023.08.10.552401

The script is designed to:
1. Read measurement data from CSV files.
2. Compute the MEP amplitude and latencies
3. Plot the summarized data

Usage:
- Create a .env file in the current working directory
- Ensure that the relative file paths are correctly entered.
- Create a Python environment with the required libraries specified in the environment.yml file

"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import settings as stg


sns.set_theme(style="ticks")
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

# redefining matplotlib default parameters
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'light'
mpl.rcParams['axes.linewidth'] = 1.
mpl.rcParams["mathtext.fontset"] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'

mm2in = lambda mm: (mm/2.54)*0.1
deg_sign = u'\xb0'
en_dash = u'\u2013'
minus_sign = u'\u2212'
micro_sign = u'\u00b5'

# %
# show and save the plot
id_show, id_save = True, True

# figure dimensions as width and height
fig_ratio, fig_res, fig_format = 1.3, 600, 'png'
fig_dim = (mm2in(1.*200), mm2in(60))
fontsize = 10

orientations = np.linspace(np.pi/2, -(3/2)*np.pi, 9)[:-1]

filename_png = os.path.join(stg.DIR_SAVE_PLOT, 'mep_{}.{}')
filename = os.path.join(stg.DIR_MEP, 'mep_amplitude_latency.csv')

df_all = pd.read_csv(filename)

mask = df_all['orientation'] > 180
df_all.loc[mask, 'orientation'] = df_all.loc[mask, 'orientation'] - 360
df_all.sort_values(by=['brain', 'paw', 'orientation'], inplace=True)
df_all['orientation_rad'] = np.deg2rad(df_all['orientation'])

# create column with contralateral and ipsilateral labels
df_all['mep_side'] = df_all['paw'].copy(deep=True)
brain_unique, paw_unique = df_all['brain'].unique(), df_all['paw'].unique()
combination = np.array(np.meshgrid(paw_unique, brain_unique)).reshape(4, 2)
combination[1, :] = combination[1, ::-1]

for n in combination:
    if n[0] == n[1]:
        label = 'ipsilateral'
    else:
        label = 'contralateral'
    mask = (df_all['brain'] == n[0]) & (df_all['paw'] == n[1])
    df_all.loc[mask, 'mep_side'] = label

# %%
measure = ['amplitude', 'latency']
measure_units = ['{}V'.format(micro_sign), 'ms']
x_axis_label = [" ", " "]
n_orientations = df_all.orientation.unique().size

for id_measure in range(len(measure)):
    df_plot = df_all.copy(deep=True)
    if id_measure:
        df_plot = df_plot[df_plot[measure[id_measure]] != 0]

    g = sns.catplot(x="orientation", y=measure[id_measure], col="brain", hue="mep_side", dodge=False,
                    data=df_plot, kind="point", height=fig_dim[1], aspect=fig_ratio, estimator=np.median,
                    sharey=True, sharex=True, legend=True, palette=[3*[0.], 3*[0.6]],
                    err_kws={'linewidth': 1.7}, markers=['o', 'D'], markersize=4, linewidth=1.5)

    if id_measure:
        g.set(xticks=[n for n in range(n_orientations)], xlim=[-0.5, n_orientations], ylim=[0, 16],
              yticks=[0, 5, 10, 15], yticklabels=[0, 5, 10, ' 15'],
              xticklabels=['{}135'.format(minus_sign), '{}90'.format(minus_sign), '{}45'.format(minus_sign),
                           0, 45, 90, 135])
    else:
        g.set(xticks=[n for n in range(n_orientations)], yticks=[0, 50, 100],
              xlim=[-0.5, n_orientations], ylim=[-1, 110],
              xticklabels=['{}135'.format(minus_sign), '{}90'.format(minus_sign), '{}45'.format(minus_sign),
                           0, 45, 90, 135])

    g.figure.text(0.25, 0.065, 'Orientation ({})'.format(deg_sign), ha='center', va='center', fontsize=fontsize)

    g.set_axis_labels(x_axis_label[id_measure], "MEP {} ({})".format(measure[id_measure], measure_units[id_measure]),
                      fontsize=fontsize)

    # Customize legend
    g._legend.set_title('MEP side')
    plt.setp(g._legend.get_title(), fontsize=fontsize)
    plt.setp(g._legend.get_texts(), fontsize=fontsize)

    axes = g.axes.flatten()
    for ax, st in zip(axes, g.axes_dict):
        sns.despine(ax=ax, offset=10, trim=True)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_title("")
        ax.tick_params(length=3.)

        for clcs in ax.collections:
            clcs.set_edgecolor([1., 1., 1.])
            clcs.set_linewidth(0.7)

    # show and/or save figure
    if id_show:
        g.tight_layout()
        plt.show()

    if id_save:
        g.savefig(filename_png.format(measure[id_measure], fig_format),
                  dpi=fig_res, bbox_inches='tight', pad_inches=0.05, transparent=False)
