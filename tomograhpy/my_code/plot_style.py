"""
Journal (scientific-paper) figure format shared by every script and notebook.

Usage
-----
    from my_code.plot_style import (apply_paper_style, save_paper_figure, plot_orientation_field,
                                    SINGLE_COL, DOUBLE_COL, MM, C_OBS, C_SYN, C_TRUE, GRID_KW)
    apply_paper_style()                                    # once, before the first figure
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    ...
    save_paper_figure(fig, IMAGE_DIR / "name")             # -> name.png (+ name.pdf if PDF is on)

Rules (see CLAUDE.md, "Figures"):
  * single-panel paper figures PAPER_FIGSIZE (SINGLE_COL x 3.0 in) so all share one size;
    multi-panel width DOUBLE_COL (7.0 in); never pass dpi to plt.subplots
  * 8 pt text, 7 pt legends, inward ticks, no top/right spines (PAPER_RC)
  * axis labels "quantity (unit)" with math symbols: '$x$ (mm)', r'Time ($\\mu$s)', r'$\\theta$ (deg)'
  * lengths in mm (MM = 1e3), vertical axis labelled z
  * no figure titles for paper figures (caption instead); multi-panel figures get
    left-aligned panel labels '(a) ...', '(b) ...'
  * dense colour fields rasterized=True, colour bars via make_axes_locatable
  * save with save_paper_figure: 300 dpi PNG (cropped to the content); PDF (embedded fonts) only when turned on
    (SAVE_PDF = True, or pdf=True for one figure)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from my_code.utilities import elemental_nodal_to_nodal_field


PAPER_RC = {
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,   # embed TrueType fonts (journal requirement)
    'ps.fonttype': 42,
    'savefig.dpi': 300,
    'figure.dpi': 150,
}

SAVE_PDF = False                    # save_paper_figure writes a PDF too only when True

SINGLE_COL, DOUBLE_COL = 3.5, 7.0   # journal column widths [in]
PAPER_FIGSIZE = (SINGLE_COL, 3.0)  # every single-panel paper figure [in]
MM = 1e3                            # m -> mm for plotting

C_OBS, C_SYN, C_TRUE = 'red', '#0072B2', '#009E73'   # observed / synthetic / true-model curves
C_SRC, C_REC = 'red', '#009E73'                      # source / receiver markers
GRID_KW = dict(linewidth=0.4, color='0.85')


def apply_paper_style():
    """Set the journal rcParams for every figure created afterwards."""
    plt.rcParams.update(PAPER_RC)


def save_paper_figure(fig, save_path, dpi=None, pdf=None):
    """
    Save `fig` as <save_path>.png, plus <save_path>.pdf (vector) when PDF output is on.

    pdf : True / False for this figure; None (default) follows the module switch
          SAVE_PDF (off unless turned on, e.g. `plot_style.SAVE_PDF = True` for a
          final paper run).

    Only a trailing .pdf / .png is treated as an extension, so names containing
    dots (receiver ids such as 'XX.R1') are kept whole.
    """
    save_path = str(save_path)
    if save_path.lower().endswith(('.pdf', '.png')):
        save_path = save_path[:-4]
    exts = ['.png'] + (['.pdf'] if (SAVE_PDF if pdf is None else pdf) else [])
    for ext in exts:
        fig.savefig(save_path + ext, bbox_inches='tight', **({'dpi': dpi} if dpi else {}))
    print(f"Saved {save_path}" + " / ".join(exts))


def add_colorbar(fig, ax, mappable, label):
    """Colour bar of the same height as `ax` (5 % wide strip on the right)."""
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.08)
    return fig.colorbar(mappable, cax=cax, label=label)


def plot_sources_receivers(ax, src_locations, rec_locations, legend=True):
    """Source (red down-triangle) and receiver (green up-triangle) markers, positions in m."""
    ax.scatter(src_locations[:, 0] * MM, src_locations[:, 1] * MM, marker='v', color=C_SRC,
               edgecolors='k', linewidths=0.4, s=30, zorder=5, label='Source')
    ax.scatter(rec_locations[:, 0] * MM, rec_locations[:, 1] * MM, marker='^', color=C_REC,
               edgecolors='k', linewidths=0.4, s=24, zorder=5, label='Receiver')
    if legend:
        ax.legend(loc='upper right', framealpha=0.85, edgecolor='none', handletextpad=0.3)


def orientation_from_stiffness(mesh):
    """
    Recover the nodal grain orientation from the meshed 2D stiffness tensor.

    Uses the ClockwiseAngle convention (symmetry axis clockwise from +z).
    Under a rotation by theta, C22 - C11 = D cos(2 theta) and
    2 (C13 + C23) = D sin(2 theta), so theta follows from atan2 without the
    unoriented reference tensor (works for inverted models too). Tensor
    components, not angles, are averaged to the nodes, so it is wrap-safe.

    Returns
    -------
    theta_deg : (n_nodes,) orientation in degrees, in [0, 180)
    two_theta : (n_nodes,) doubled angle in radians (for smooth interpolation)
    """
    C = {
        k: elemental_nodal_to_nodal_field(mesh.element_nodal_fields[k], mesh.connectivity)
        for k in ['C11', 'C22', 'C13', 'C23']
    }
    two_theta = np.arctan2(2.0 * (C['C13'] + C['C23']), C['C22'] - C['C11'])
    theta_deg = np.rad2deg(0.5 * two_theta) % 180.0
    return theta_deg, two_theta


def plot_orientation_field(mesh, src_locations, rec_locations, extent, save_path=None,
                           show_glyphs=True, n_glyphs=15, title=None):
    """
    Simulation configuration: orientation field of `mesh` with sources and receivers.

    extent      : (x0, x1, y0, y1) of the plotted (physical) domain [m]
    show_glyphs : dashed line segments along the symmetry axis on an
                  n_glyphs x n_glyphs grid (the local orientation angle)
    save_path   : saved as <save_path>.png (+ .pdf when PDF output is on)
    title       : optional; paper figures leave it to the caption

    Returns (fig, ax); the caller shows (notebook) or closes (script) the figure.
    """
    x0, x1, y0, y1 = extent
    theta_deg, two_theta = orientation_from_stiffness(mesh)
    print(f"Orientation range in model: {theta_deg.min():.2f} - {theta_deg.max():.2f} deg")

    tri = mtri.Triangulation(mesh.points[:, 0] * MM, mesh.points[:, 1] * MM)

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    tp = ax.tripcolor(tri, theta_deg, cmap='viridis', vmin=0.0, vmax=90.0,
                      shading='gouraud', rasterized=True)
    add_colorbar(fig, ax, tp, r'$\theta$ (deg, clockwise from $+z$)')

    if show_glyphs:
        # Interpolate cos/sin of 2*theta on a regular grid, then halve the angle;
        # dashed segments along the symmetry axis (clockwise from +z: (sin t, cos t))
        xg, yg = np.meshgrid(np.linspace(x0 * MM, x1 * MM, n_glyphs)[1:-1],
                             np.linspace(y0 * MM, y1 * MM, n_glyphs)[1:-1])
        c2 = mtri.LinearTriInterpolator(tri, np.cos(two_theta))(xg, yg)
        s2 = mtri.LinearTriInterpolator(tri, np.sin(two_theta))(xg, yg)
        theta_g = 0.5 * np.arctan2(s2, c2)
        half = 0.4 * (x1 - x0) * MM / (n_glyphs - 1)   # half segment length [mm]
        dx, dy = half * np.sin(theta_g), half * np.cos(theta_g)
        segs = np.stack([np.stack([xg - dx, yg - dy], -1),
                         np.stack([xg + dx, yg + dy], -1)], axis=-2).reshape(-1, 2, 2)
        ax.add_collection(LineCollection(segs, colors='k', linewidths=0.7,
                                         linestyles=(0, (2.0, 1.5)), zorder=3))

    plot_sources_receivers(ax, src_locations, rec_locations)
    ax.set_xlabel('$x$ (mm)')
    ax.set_ylabel('$z$ (mm)')
    if title:
        ax.set_title(title)
    ax.set_xlim(x0 * MM, x1 * MM)
    ax.set_ylim(y0 * MM, y1 * MM)
    ax.set_aspect('equal')
    fig.tight_layout(pad=0.3)
    if save_path is not None:
        save_paper_figure(fig, save_path)
    return fig, ax
