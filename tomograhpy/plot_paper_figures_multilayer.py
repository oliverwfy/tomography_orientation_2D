"""
All paper figures of a saved multilayer orientation FWI run (re-plot without re-running the FWI).

Redraws, into <image>/FWI_orientation_multilayer/paper_figures/, the figures that
FWI_orientation_multilayer.py saves when it finishes:

  * FWI_orientation_configuration       : true orientation field, source and receivers
                                          (the mesh is rebuilt; no simulation is run)
  * FWI_orientation_misfit_vs_iteration : misfit per accepted iteration, one band per stage
  * model_error_convergence / _profile / _layers : see my_code/model_error.py

Run from the tomograhpy folder:
    python plot_paper_figures_multilayer.py
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file-only backend
import matplotlib.pyplot as plt

import salvus.mesh
import salvus.mesh.layered_meshing as lm
import salvus.namespace as sn
from salvus.material._details import material as md

from my_code.model_error import save_model_error_figures
from my_code.plot_style import (apply_paper_style, save_paper_figure, plot_orientation_field,
                                SINGLE_COL, C_SYN, GRID_KW)
from material_ela_constants.Elastic_Material import Austenite


# =============================================================================
# Paths (same layout as FWI_orientation_multilayer.py)
# =============================================================================

if os.environ.get('SCRATCH', '').startswith('/scratch'):
    _BASE = os.environ['SCRATCH'] + '/Salvus_project/elastic_model/anisotropic'   # Isambard AI
else:
    _BASE = '/mnt/f/Salvus_project/elastic_model/anisotropic'                    # WSL / Windows

DATA_DIR      = Path(_BASE, 'data', 'FWI_orientation_multilayer')
PAPER_FIG_DIR = Path(_BASE, 'image', 'FWI_orientation_multilayer', 'paper_figures')


# =============================================================================
# Run and true model (must match the FWI run that wrote DATA_DIR)
# =============================================================================

CENTRAL_FREQUENCY    = 2e6                                     # alpha = 1 centre frequency [Hz]
x0, x1               = 0.0, 0.01                               # physical domain [m]
y0, y1               = 0.0, 0.01
BACKGROUND_ANGLE_DEG = 45.0                                    # [deg]
LAYER_INTERFACES_Y   = [0.009, 0.007, 0.006, 0.0045, 0.002, 0.001]   # top to bottom [m]
LAYER_ANGLES_DEG     = [70.0, 30.0, 60.0, 20.0, 40.0]          # layers 1-5, top to bottom

tx_locations = np.array([[0.005, 0.01]])                                   # source (x, y) [m]
r_loc_all    = np.array([[x, 0.0] for x in np.linspace(0.001, 0.009, 9)])  # receivers (x, y) [m]

# Mesh settings of the FWI run
elements_per_wavelength = 4
model_order             = 4
number_of_wavelengths   = 2                 # absorbing layer thickness (in wavelengths)
free_surfaces           = ['y0', 'y1']

apply_paper_style()
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Simulation configuration (true model rebuilt with the run's mesh settings)
# =============================================================================

def oriented_material(material_unoriented, angle_in_degrees):
    """Rotate a hexagonal material and convert it to the 2D solver form (C11..C33)."""
    orientation = sn.material.orientation.ClockwiseAngle.from_params(
        angle_in_degrees=angle_in_degrees
    )
    return md.to_solver_form(material_unoriented.with_orientation(orientation), ndim=2)


matl    = Austenite()
VP_0deg = np.sqrt(matl.C33 / matl.RHO)      # P velocity along symmetry axis [m/s]
material_unoriented = sn.material.elastic.hexagonal.TensorComponents.from_params(
    **matl.VTI_parameters()
)
matl_background = oriented_material(material_unoriented, BACKGROUND_ANGLE_DEG)
_layer_stack = [matl_background]
for _y_if, _matl in zip(LAYER_INTERFACES_Y,
                        [oriented_material(material_unoriented, a) for a in LAYER_ANGLES_DEG]
                        + [matl_background]):
    _layer_stack += [sn.layered_meshing.interface.Hyperplane.at(_y_if), _matl]

reference_frequency = CENTRAL_FREQUENCY * 2
mesh = lm.mesh_from_domain(
    domain=sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1),
    model=sn.layered_meshing.MeshingProtocol(
        lm.LayeredModel(_layer_stack),
        ab=salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
            free_surface=free_surfaces, number_of_wavelengths=number_of_wavelengths,
            reference_velocity=VP_0deg, reference_frequency=reference_frequency,
        ),
    ),
    mesh_resolution=sn.MeshResolution(
        reference_frequency=reference_frequency,
        elements_per_wavelength=elements_per_wavelength, model_order=model_order,
    ),
)

fig, _ = plot_orientation_field(mesh, tx_locations, r_loc_all, extent=(x0, x1, y0, y1),
                                save_path=PAPER_FIG_DIR / "FWI_orientation_configuration")
plt.close(fig)


# =============================================================================
# Stages: alpha from the per-stage files; the multiscale history drops row 0 of
# every later stage (= the previous stage's last model)
# =============================================================================

files  = sorted(DATA_DIR.glob('misfit_history_alpha_*.npy'), key=lambda f: float(f.stem.split('_')[-1]))
alphas = [float(f.stem.split('_')[-1]) for f in files]
counts = [len(np.load(f)) - (i > 0) for i, f in enumerate(files)]
stage_boundaries = np.concatenate([[0], np.cumsum(counts)])

misfit_history = np.load(DATA_DIR / 'misfit_history_multiscale.npy')
assert len(misfit_history) == stage_boundaries[-1], (len(misfit_history), counts)


# =============================================================================
# Misfit chi vs iteration (log scale, one band per frequency stage)
# =============================================================================

fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
for _k, _alpha in enumerate(alphas):
    _lo, _hi = stage_boundaries[_k] - 0.5, stage_boundaries[_k + 1] - 0.5
    if _k % 2 == 1:
        ax.axvspan(_lo, _hi, color='0.93', zorder=0, linewidth=0)
    ax.text(0.5 * (_lo + _hi), 1.0, f"{_alpha * CENTRAL_FREQUENCY / 1e6:.1f} MHz",
            transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
ax.semilogy(np.arange(len(misfit_history)), misfit_history, color=C_SYN, marker='o',
            linewidth=1.0, markersize=3, markeredgecolor='black', markeredgewidth=0.4, zorder=3)
ax.set_xlabel("FWI iteration")
ax.set_ylabel(r"$\chi$")
ax.yaxis.grid(True, which='major', **GRID_KW)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
save_paper_figure(fig, PAPER_FIG_DIR / "FWI_orientation_misfit_vs_iteration", dpi=600)
plt.close(fig)


# =============================================================================
# Model error vs the true model (RMS / relative theta error, profile, layers)
# =============================================================================

save_model_error_figures(
    np.load(DATA_DIR / 'model_history_theta_rad_multiscale.npy'), np.load(DATA_DIR / 'roi_node_xy.npy'),
    alphas, counts, CENTRAL_FREQUENCY, LAYER_INTERFACES_Y, LAYER_ANGLES_DEG, BACKGROUND_ANGLE_DEG,
    (x0, x1), PAPER_FIG_DIR,
)
