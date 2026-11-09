"""
TFM with the averaged P velocity of the FWI-reconstructed model vs RTM.

Uses the data saved by RTM_orientation.py (no new simulations):
  data/RTM_orientation_multilayer/rtm_receiver_traces.npz  defect-scattered residual traces
  data/RTM_orientation_multilayer/rtm_image_<model>.npz    RTM images
  data/FWI_orientation_multilayer/                         reconstructed theta(z) (FWI output)

TFM (delay-and-sum, isotropic, single velocity):
  * VP_avg: vertical qP phase velocity of the reconstructed model at every depth (largest
    eigenvalue of the Christoffel matrix for n = z), background 45 deg outside the inverted
    region, harmonic mean over the 10 mm depth (= the average that gives the vertical travel
    time through the stack)
  * I(x) = | sum_r A_r(t0 + (|x - s| + |x - r|) / VP_avg) |, A_r = analytic signal (Hilbert
    transform) of the vertical-displacement residual u_z at receiver r, t0 = Ricker delay

Figure (paper_figures/TFM_RTM_comparison):
  (a) TFM, VP_avg of the reconstructed model   (b) RTM, homogeneous 45 deg   (c) RTM, reconstructed

Run from the tomograhpy folder:
    python TFM_RTM_comparison.py
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file-only backend
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from my_code.plot_style import (apply_paper_style, save_paper_figure, plot_sources_receivers,
                                DOUBLE_COL, MM)
from material_ela_constants.Elastic_Material import Austenite


# =============================================================================
# Paths and settings (must match RTM_orientation.py)
# =============================================================================

if os.environ.get('SCRATCH', '').startswith('/scratch'):
    _BASE = os.environ['SCRATCH'] + '/Salvus_project/elastic_model/anisotropic'
    DATA_DIR_WIN, IMAGE_DIR_WIN = _BASE + '/data', _BASE + '/image'
else:
    DATA_DIR_WIN  = '/mnt/f/Salvus_project/elastic_model/anisotropic/data'
    IMAGE_DIR_WIN = '/mnt/f/Salvus_project/elastic_model/anisotropic/image'

FWI_DATA_DIR  = Path(DATA_DIR_WIN) / "FWI_orientation_multilayer"
RTM_DATA_DIR  = Path(DATA_DIR_WIN) / "RTM_orientation_multilayer"
PAPER_FIG_DIR = Path(IMAGE_DIR_WIN) / "RTM_orientation_multilayer" / "paper_figures"

x0, x1, y0, y1       = 0.0, 0.01, 0.0, 0.01                     # physical domain [m]
BACKGROUND_ANGLE_DEG = 45.0
TIME_SHIFT           = 1.0e-6                                   # Ricker delay 1 / f_c of RTM_orientation.py [s]
DEFECT_CENTERS       = np.column_stack([np.linspace(20, 80, 4), np.linspace(20, 75, 4)]) * 1e-4
RTM_IMAGE_KEY        = 'image'                                  # or 'image_illumination_normalised'

matl = Austenite()


def stiffness_from_theta(theta):
    """Rotated 2D stiffness (Salvus Voigt names) for orientation theta [rad], as the FWI scripts."""
    Cb11, Cb13, Cb33, Cb55 = matl.C11, matl.C13, matl.C33, matl.C44
    a = Cb11 - 2.0 * Cb13 + Cb33 - 4.0 * Cb55
    b = 3.0 * Cb11 + 2.0 * Cb13 + 3.0 * Cb33 + 4.0 * Cb55
    g = Cb11 - Cb33
    s2, c2, s4, c4 = np.sin(2 * theta), np.cos(2 * theta), np.sin(4 * theta), np.cos(4 * theta)
    return {'C22': b / 8.0 - g / 2.0 * c2 + a / 8.0 * c4,          # zz
            'C23': -g / 4.0 * s2 + a / 8.0 * s4,                    # zz-xz
            'C33': (Cb11 - 2.0 * Cb13 + Cb33 + 4.0 * Cb55) / 8.0 - a / 8.0 * c4}   # xz-xz


def vertical_qp_velocity(theta_deg):
    """qP phase velocity for propagation along z: Christoffel [[C33, C23], [C23, C22]] / rho."""
    C = stiffness_from_theta(np.deg2rad(theta_deg))
    tr, det = C['C33'] + C['C22'], C['C33'] * C['C22'] - C['C23'] ** 2
    return np.sqrt(0.5 * (tr + np.sqrt(tr ** 2 - 4.0 * det)) / matl.RHO)


# =============================================================================
# Averaged VP of the reconstructed model
# =============================================================================

models = np.load(FWI_DATA_DIR / 'model_history_theta_rad_multiscale.npy')
xy = np.load(FWI_DATA_DIR / 'roi_node_xy.npy')
inside = (xy[:, 0] >= x0 - 1e-9) & (xy[:, 0] <= x1 + 1e-9)
z_lv, row = np.unique(np.round(xy[inside, 1], 10), return_inverse=True)
theta_rec = np.bincount(row, weights=np.rad2deg(models[-1][inside])) / np.bincount(row)

z = np.linspace(y0, y1, 4001)
theta_z = np.where((z >= z_lv[0]) & (z <= z_lv[-1]), np.interp(z, z_lv, theta_rec), BACKGROUND_ANGLE_DEG)
vp_z = vertical_qp_velocity(theta_z)
VP_AVG = 1.0 / np.mean(1.0 / vp_z)                              # harmonic mean over depth
print(f"Reconstructed model: vertical qP {vp_z.min():.0f}-{vp_z.max():.0f} m/s, "
      f"arithmetic mean {vp_z.mean():.0f} m/s, harmonic mean (used) VP_avg = {VP_AVG:.0f} m/s")


# =============================================================================
# TFM: delay-and-sum of the analytic u_z residual, single velocity VP_avg
# =============================================================================

tr = np.load(RTM_DATA_DIR / 'rtm_receiver_traces.npz')
t, src, rx = tr['time'], tr['source_loc'], tr['receiver_locs']
analytic = hilbert(tr['residual'][:, 1, :], axis=-1)            # u_z (Salvus component 'Y')

rtm = {name: np.load(RTM_DATA_DIR / f'rtm_image_{name}.npz') for name in ('homogeneous', 'reconstructed')}
x_img, y_img = rtm['reconstructed']['x'], rtm['reconstructed']['y']
X, Y = np.meshgrid(x_img, y_img)                                 # (ny, nx), as the RTM images
d_src = np.hypot(X - src[0], Y - src[1])
tfm = np.zeros(X.shape, dtype=complex)
for r, a in zip(rx, analytic):
    tau = TIME_SHIFT + (d_src + np.hypot(X - r[0], Y - r[1])) / VP_AVG
    tfm += np.interp(tau, t, a.real, left=0, right=0) + 1j * np.interp(tau, t, a.imag, left=0, right=0)
tfm = np.abs(tfm)
np.savez_compressed(RTM_DATA_DIR / 'tfm_image_reconstructed_vp_avg.npz', image=tfm, x=x_img, y=y_img,
                    vp_avg=VP_AVG)


# =============================================================================
# Figure: (a) TFM, (b) RTM homogeneous, (c) RTM reconstructed
# =============================================================================

apply_paper_style()
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.7), sharey=True, layout='constrained')
panels = [
    (tfm, rf'(a) TFM, $\bar{{V}}_P$ = {VP_AVG:.0f} m/s', 'magma', (0, 1)),
    (rtm['homogeneous'][RTM_IMAGE_KEY], r'(b) RTM, homogeneous (45$^\circ$)', 'RdBu', (-1, 1)),
    (rtm['reconstructed'][RTM_IMAGE_KEY], '(c) RTM, reconstructed', 'RdBu', (-1, 1)),
]
for k, (ax, (img, title, cmap, lim)) in enumerate(zip(axes, panels)):
    img = img / (np.percentile(np.abs(img), 99.0) or 1.0)
    pc = ax.pcolormesh(x_img * MM, y_img * MM, img, cmap=cmap, vmin=lim[0], vmax=lim[1],
                       shading='auto', rasterized=True)
    ax.scatter(DEFECT_CENTERS[:, 0] * MM, DEFECT_CENTERS[:, 1] * MM, s=30, facecolors='none',
               edgecolors='w' if cmap == 'magma' else 'k', linewidths=0.7, zorder=6)
    plot_sources_receivers(ax, src[None, :], rx, legend=(k == 1))
    ax.set_xlim(x0 * MM, x1 * MM); ax.set_ylim(y0 * MM, y1 * MM); ax.set_aspect('equal')
    ax.set_xlabel('$x$ (mm)')
    ax.set_title(title, loc='left')
    if k == 0:
        fig.colorbar(pc, ax=ax, location='bottom', shrink=0.8, label='Normalised TFM amplitude')
    if k == 2:
        fig.colorbar(pc, ax=axes[1:].tolist(), location='bottom', shrink=0.5,
                     label='Normalised RTM amplitude')
axes[0].set_ylabel('$z$ (mm)')
save_paper_figure(fig, PAPER_FIG_DIR / 'TFM_RTM_comparison')
plt.close(fig)
