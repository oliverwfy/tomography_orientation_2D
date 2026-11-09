"""
Redraw the RTM image figures from the arrays saved by RTM_orientation.py (no new simulations):
RTM_images and RTM_images_illumination_normalised, (a) homogeneous (b) reconstructed (c) true.

Run from the tomograhpy folder:
    python plot_RTM_images.py
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file-only backend

from my_code.plot_style import apply_paper_style
from my_code.rtm_plots import save_rtm_image_figure, RTM_PANEL_ORDER

if os.environ.get('SCRATCH', '').startswith('/scratch'):
    _BASE = os.environ['SCRATCH'] + '/Salvus_project/elastic_model/anisotropic'
    DATA_DIR_WIN, IMAGE_DIR_WIN = _BASE + '/data', _BASE + '/image'
else:
    DATA_DIR_WIN  = '/mnt/f/Salvus_project/elastic_model/anisotropic/data'
    IMAGE_DIR_WIN = '/mnt/f/Salvus_project/elastic_model/anisotropic/image'

RTM_DATA_DIR  = Path(DATA_DIR_WIN) / "RTM_orientation_multilayer"
PAPER_FIG_DIR = Path(IMAGE_DIR_WIN) / "RTM_orientation_multilayer" / "paper_figures"
EXTENT        = (0.0, 0.01, 0.0, 0.01)                                        # physical domain [m]
DEFECTS       = np.column_stack([np.linspace(20, 80, 4), np.linspace(20, 75, 4)]) * 1e-4   # as RTM_orientation.py

apply_paper_style()
saved = {n: np.load(RTM_DATA_DIR / f"rtm_image_{n}.npz") for n in RTM_PANEL_ORDER
         if (RTM_DATA_DIR / f"rtm_image_{n}.npz").exists()}
traces = np.load(RTM_DATA_DIR / "rtm_receiver_traces.npz")
x_img, y_img = next(iter(saved.values()))['x'], next(iter(saved.values()))['y']
for tag, key in [("RTM_images", "image"), ("RTM_images_illumination_normalised", "image_illumination_normalised")]:
    save_rtm_image_figure({n: d[key] for n, d in saved.items()}, x_img, y_img,
                          traces['source_loc'][None, :], traces['receiver_locs'], DEFECTS, EXTENT,
                          PAPER_FIG_DIR / tag)
