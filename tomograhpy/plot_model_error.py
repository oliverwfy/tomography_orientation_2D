"""
Model-error figures of a saved orientation FWI run (re-plot without re-running the FWI).

The FWI scripts save these figures themselves when they finish; this script redraws them from
the saved histories (model_history_theta_rad_multiscale.npy, roi_node_xy.npy,
misfit_history_alpha_*.npy). See my_code/model_error.py for the metrics and figures.

Run from the tomograhpy folder:
    python plot_model_error.py
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file-only backend

from my_code.model_error import save_model_error_figures


# =============================================================================
# Run and true model (must match the FWI run that wrote DATA_DIR)
# =============================================================================

if os.environ.get('SCRATCH', '').startswith('/scratch'):
    _BASE = os.environ['SCRATCH'] + '/Salvus_project/elastic_model/anisotropic'   # Isambard AI
else:
    _BASE = '/mnt/f/Salvus_project/elastic_model/anisotropic'                    # WSL / Windows

DATA_DIR  = Path(_BASE, 'data', 'FWI_orientation_multilayer')
IMAGE_DIR = Path(_BASE, 'image', 'FWI_orientation_multilayer', 'paper_figures')

CENTRAL_FREQUENCY  = 2e6                                     # alpha = 1 centre frequency of that run [Hz]
x0, x1             = 0.0, 0.01                               # physical domain [m]
BACKGROUND_ANGLE   = 45.0                                    # [deg]
LAYER_INTERFACES_Y = [0.009, 0.007, 0.006, 0.0045, 0.002, 0.001]   # top to bottom [m]
LAYER_ANGLES_DEG   = [70.0, 30.0, 60.0, 20.0, 40.0]          # layers 1-5, top to bottom


# stages: alpha from the per-stage files; the multiscale history drops row 0 of every
# later stage (= the previous stage's last model)
files = sorted(DATA_DIR.glob('misfit_history_alpha_*.npy'), key=lambda f: float(f.stem.split('_')[-1]))
alphas = [float(f.stem.split('_')[-1]) for f in files]
counts = [len(np.load(f)) - (i > 0) for i, f in enumerate(files)]

save_model_error_figures(
    np.load(DATA_DIR / 'model_history_theta_rad_multiscale.npy'), np.load(DATA_DIR / 'roi_node_xy.npy'),
    alphas, counts, CENTRAL_FREQUENCY, LAYER_INTERFACES_Y, LAYER_ANGLES_DEG, BACKGROUND_ANGLE,
    (x0, x1), IMAGE_DIR,
)
