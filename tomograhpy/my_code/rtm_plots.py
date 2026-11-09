"""
RTM image figures (project journal format), shared by RTM_orientation.py and plot_RTM_images.py.
"""

import numpy as np
import matplotlib.pyplot as plt

from my_code.plot_style import save_paper_figure, plot_sources_receivers, DOUBLE_COL, MM

# panel order of the RTM image figures: (a) homogeneous, (b) reconstructed, (c) true model
RTM_PANEL_ORDER = ['homogeneous', 'reconstructed', 'true']
RTM_TITLES = {'homogeneous': r'Homogeneous model (45$^\circ$)',
              'reconstructed': 'Reconstructed (FWI) model', 'true': 'True model'}


def save_rtm_image_figure(images, x_img, y_img, src, rec, defects, extent, save_path):
    """
    One panel per migration model in RTM_PANEL_ORDER (models missing from `images` are skipped),
    each normalised by its 99th percentile, shared colour bar.

    images  : {model name: (ny, nx) image}
    src, rec: (n, 2) source / receiver positions [m]
    defects : (n, 2) scatterer positions [m], drawn as open circles
    extent  : (x0, x1, y0, y1) [m]
    """
    names = [n for n in RTM_PANEL_ORDER if n in images]
    x0, x1, y0, y1 = extent
    fig, axes = plt.subplots(1, len(names), figsize=(DOUBLE_COL, 2.7), sharey=True,
                             layout='constrained')
    axes = np.atleast_1d(axes)
    for k, (ax, name) in enumerate(zip(axes, names)):
        img = images[name] / (np.percentile(np.abs(images[name]), 99.0) or 1.0)
        pc = ax.pcolormesh(x_img * MM, y_img * MM, img, cmap='RdBu', vmin=-1, vmax=1,
                           shading='auto', rasterized=True)
        ax.scatter(defects[:, 0] * MM, defects[:, 1] * MM, s=30, facecolors='none',
                   edgecolors='k', linewidths=0.7, zorder=6)
        plot_sources_receivers(ax, src, rec, legend=(k == 0))
        ax.set_xlim(x0 * MM, x1 * MM); ax.set_ylim(y0 * MM, y1 * MM); ax.set_aspect('equal')
        ax.set_xlabel('$x$ (mm)')
        ax.set_title(f'({"abcdef"[k]}) {RTM_TITLES[name]}', loc='left')
    axes[0].set_ylabel('$z$ (mm)')
    fig.colorbar(pc, ax=axes.tolist(), shrink=0.8, label='Normalised RTM amplitude')
    save_paper_figure(fig, save_path)
    plt.close(fig)
