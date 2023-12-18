"""
plot_efield_map.py

Author: Victor H. Souza
Date: December 16, 2023

Copyright (c) 2023 Victor H. Souza

Description:
Plot 2D projection of 3D electric field vectors measured with TMS characterizer.

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

import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import settings as stg
import processing_tools as pt

# redefining matplotlib default parameters
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.weight'] = 'light'
mpl.rcParams['axes.linewidth'] = 1.5

mm2in = lambda mm: (mm / 2.54) * 0.1
deg_sign = u'\xb0'
en_dash = u'\u2013'

# %%
# show the plot
id_show, id_save = True, True

# directions = (0, 45, 90)
# figure dimensions as width and height
fig_ratio, fig_res, fig_format = 3., 600, 'png'
wid = 60
hei = wid*fig_ratio
fig_dim = (mm2in(wid), mm2in(hei))

# directories: where the data file is located and where to save the plots
data_dir = stg.DIR_EFIELD_CURRENT
plot_dir = stg.DIR_SAVE_PLOT
filename_list = ('0', '45', '90')


# %
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=fig_dim)

# n to plot: 0, 1, 2 is 0, 45, 90-degree maps
for n, ax in enumerate(axs):

    # Load field values
    filepath = os.path.join(data_dir, 'efield_map_20Vm_{}deg.txt'.format(filename_list[n]))
    pos, ef, efn = pt.read_efield(filepath)
    scale_pos = 1000  # Scale coordinates from m to mm
    pos *= scale_pos
    ax_lim = 71

    enor = np.linalg.norm(ef, axis=1).tolist()
    enor = [en - min(enor) for en in enor]
    enor = [en / max(enor) for en in enor]

    # Creates extra circles and lines in plot
    in_circle = plt.Circle((0, 0), 70, color=[col/255. for col in 3*[127.]], zorder=1)

    # Normalize the colormap values according to E-field norm
    norm = Normalize()
    norm.autoscale(enor)
    colormap = mpl.colormaps['cividis']

    z = np.array(enor)
    x, y = pos[:, 0], pos[:, 1]

    # create triangles based on point coordinates
    triang = tri.Triangulation(x, y)
    # the cubic interpolation causes a peak in amplitude, better to use the linear
    interp_lin = tri.LinearTriInterpolator(triang, z)
    # refine the triangulation by a subdivision of 2
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(z, triinterpolator=interp_lin,
                                                 subdiv=2)
    ax.tricontourf(tri_refi, z_test_refi, cmap=colormap, levels=9, zorder=3)

    fc = colormap(norm(enor))

    if n == 2:
        axins = inset_axes(ax, width="70%", height="8%", loc='lower center',
                           bbox_to_anchor=(0., -0.2, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cb = fig.colorbar(cm.ScalarMappable(norm=norm,  cmap=mpl.colormaps['cividis'].resampled(8)),
                          orientation="horizontal", cax=axins, aspect=5)
        cb.set_ticks((0., 0.5, 1.))
        cb.set_ticklabels(('0', '', '1'))
        cb.ax.tick_params(direction='in', length=1, width=.5, labelsize=10, pad=1)
        cb.ax.get_xaxis().labelpad = 5
        cb.ax.set_xlabel('Normalized E-field', rotation=0, fontsize=10)
        cb.outline.set_linewidth(1.)

    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_xticks(())
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_yticks(())

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.add_artist(in_circle)

    ax.set_aspect('equal', 'box')

    ax.title.set_text('{}{}'.format(filename_list[n], deg_sign))

# show and/or save figure
if id_show:
    # fig.tight_layout()
    plt.show()

# %
if id_save:
    filename_png = os.path.join(plot_dir, 'efield_map.{}'.format(fig_format))
    fig.savefig(filename_png, dpi=fig_res, bbox_inches='tight', pad_inches=0.01, transparent=False)
