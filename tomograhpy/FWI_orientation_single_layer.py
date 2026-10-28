"""
Elastic FWI for the grain orientation theta(y) of a single anisotropic layer.

Same workflow as acoustic_FWI_velocity_underwater/FWI_single_layer.py, with
ln(VP) replaced by the grain orientation theta [rad]:

  * gradient   : Salvus built-in adjoint gradient K_C (compute_gradients /
                 sum_gradients), mapped to K_theta = K_C :: dC/dtheta
  * model      : theta at the ROI nodes; layered prior theta(y) enforced by
                 x-averaging, accepted profiles Gaussian-smoothed along y
  * optimizer  : L-BFGS + trust region (L-inf ball in theta, radians)
  * mask       : gradient zeroed outside the first-Fresnel-zone ellipses
  * multiscale : frequency stages alpha * CENTRAL_FREQUENCY, warm-started;
                 simulation end time end_time / alpha per stage

Unlike the acoustic case the gradient is NOT mirror-symmetrised about x = 5 mm:
a tilted anisotropic medium maps theta -> -theta under that reflection.

Self-contained: paths, true model, mesh, events and the starting model
(mesh_init) are all set up below. Run from the tomograhpy folder:
    python FWI_orientation_single_layer.py

Units: SI throughout (m, s, Pa, kg/m^3); plots use mm.
"""

import copy
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file-only backend: figures are saved, never shown (no X server needed)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d

import salvus.mesh
import salvus.mesh.layered_meshing as lm
import salvus.namespace as sn
from salvus.material._details import material as md

from my_code.utilities import elemental_nodal_to_nodal_field
from my_code.model_error import save_model_error_figures
from my_code.plot_style import (apply_paper_style, save_paper_figure, plot_orientation_field,
                                C_OBS, C_SYN, C_TRUE, GRID_KW, MM as mm)
from material_ela_constants.Elastic_Material import Austenite


# =============================================================================
# Paths
# =============================================================================

_ON_ISAMBARD = os.environ.get('SCRATCH', '').startswith('/scratch')

if _ON_ISAMBARD:
    # Isambard AI HPC — store all large files on scratch (Lustre)
    _BASE           = os.environ['SCRATCH'] + '/Salvus_project/elastic_model/anisotropic'
    PROJECT_DIR_WIN = _BASE + '/Project'
    DATA_DIR_WIN    = _BASE + '/data'
    IMAGE_DIR_WIN   = _BASE + '/image'
else:
    # WSL / Windows
    PROJECT_DIR_WIN = '/mnt/f/Salvus_project/elastic_model/anisotropic/Project'
    DATA_DIR_WIN    = '/mnt/f/Salvus_project/elastic_model/anisotropic/data'
    IMAGE_DIR_WIN   = '/mnt/f/Salvus_project/elastic_model/anisotropic/image'

PROJECT_NAME = "tomography_solid_orientation_singlelayer"   # folder inside PROJECT_DIR_WIN

FWI_IMAGE_DIR = Path(IMAGE_DIR_WIN) / "FWI_orientation_singlelayer"
FWI_DATA_DIR  = Path(DATA_DIR_WIN) / "FWI_orientation_singlelayer"
PAPER_FIG_DIR = FWI_IMAGE_DIR / "paper_figures"

# Start every run from scratch: delete this script's previous data, images and Salvus project
for _d in (FWI_IMAGE_DIR, FWI_DATA_DIR, Path(PROJECT_DIR_WIN, PROJECT_NAME)):
    if _d.exists():
        print(f"Deleting previous results: {_d}")
        shutil.rmtree(_d)


# =============================================================================
# Solver / compute site
# =============================================================================

SITE_NAME = "isambard_oliver" if _ON_ISAMBARD else "oliver_wsl"   # Salvus site to run simulations on
RANKS     = 2              # MPI ranks per simulation job (the 2D mesh is only ~7k elements)


# =============================================================================
# Source wavelet and time axis
# =============================================================================

# Stage centre frequency = alpha * CENTRAL_FREQUENCY; the mesh is built for alpha = 1
CENTRAL_FREQUENCY      = 2e6                         # Ricker centre frequency at alpha = 1 [Hz]
# dt = 1 / (250 f_c) must stay below the mesh CFL limit (~4.25 ns for the 1 MHz mesh;
# the mesh scales with CENTRAL_FREQUENCY, so the margin is kept)
sampling_rate_in_hertz = CENTRAL_FREQUENCY * 250
time_step_in_seconds   = 1 / sampling_rate_in_hertz  # solver/output time step [s]
# Source and receiver on the free surfaces, 10 mm apart: direct qP ~1.75 us, direct qSV
# ~4.12 us (vertical group velocities 5.8 / 2.4 km/s at 45 deg, 5.5 / 2.5 km/s at 60 deg).
# 5 us keeps the whole qSV pulse at 2 MHz and stops before the first qP free-surface
# multiple (~3 x 1.75 = 5.25 us).
end_time               = 5e-6                        # simulation end time at alpha = 1 [s]

amplitude_ratio = 1e10              # source amplitude (fy of VectorPoint2D)


# =============================================================================
# Domain geometry
# =============================================================================

x0, x1 = 0.0, 0.01                 # horizontal extent [m] (10 mm)
y0, y1 = 0.0, 0.01                 # vertical extent [m]   (10 mm)

LAYER_TOP_Y    = 0.0065            # upper interface of the oriented layer [m]
LAYER_BOTTOM_Y = 0.0035            # lower interface of the oriented layer [m]


# =============================================================================
# Material and orientations
# =============================================================================

matl    = Austenite()                         # hexagonal austenite (RHO, Cij)
VP_0deg = np.sqrt(matl.C33 / matl.RHO)        # P velocity along symmetry axis [m/s]

# Grain orientation angles passed to ClockwiseAngle (symmetry axis measured
# clockwise from +z).
BACKGROUND_ANGLE_DEG = 45.0
LAYER_ANGLE_DEG      = 60.0

STIFFNESS_PARAMETERS = {'C11', 'C12', 'C13', 'C22', 'C23', 'C33'}   # Salvus 2D Voigt names


# =============================================================================
# Mesh resolution and absorbing boundaries
# =============================================================================

elements_per_wavelength = 3                       # elements per minimum wavelength
model_order             = 4                       # spectral-element polynomial order
ab_reference_velocity   = VP_0deg                 # absorbing layer thickness measured in P wavelengths [m/s]
reference_frequency     = CENTRAL_FREQUENCY * 2   # max frequency resolved by the mesh [Hz]
number_of_wavelengths   = 3                       # absorbing layer thickness (in wavelengths)
free_surfaces           = ['y0', 'y1']            # sides without absorbing layer


# =============================================================================
# Model / kernel helpers
# =============================================================================

def oriented_material(material_unoriented, angle_in_degrees):
    """Rotate a hexagonal material and convert it to the 2D solver form (C11..C33)."""
    orientation = sn.material.orientation.ClockwiseAngle.from_params(
        angle_in_degrees=angle_in_degrees
    )
    return md.to_solver_form(material_unoriented.with_orientation(orientation), ndim=2)


def theta_from_mesh(mesh):
    """
    Element-nodal grain orientation theta [rad] of a mesh (ClockwiseAngle convention).

    Under a rotation by theta, C22 - C11 = D cos(2 theta) and
    2 (C13 + C23) = D sin(2 theta), so theta follows from atan2 without the
    unoriented reference tensor (works for inverted models too).
    """
    f = mesh.element_nodal_fields
    return 0.5 * np.arctan2(2.0 * (f['C13'] + f['C23']), f['C22'] - f['C11'])


def dC_dtheta(theta, matl):
    """
    Derivative of the rotated 2D stiffness tensor w.r.t. theta [Pa/rad].

    Uses the unrotated (theta = 0) hexagonal constants Cbar of `matl` in the
    x-z plane (Cbar11 = C11, Cbar13 = C13, Cbar33 = C33, Cbar55 = C44):
        alpha = Cbar11 - 2 Cbar13 + Cbar33 - 4 Cbar55
        gamma = Cbar11 - Cbar33
    Returned keys are Salvus 2D Voigt names (1 = xx, 2 = zz, 3 = xz), i.e.
    paper C11, C13, C15, C33, C35, C55 -> Salvus C11, C12, C13, C22, C23, C33.
    """
    alpha = matl.C11 - 2.0 * matl.C13 + matl.C33 - 4.0 * matl.C44
    gamma = matl.C11 - matl.C33
    s2, c2 = np.sin(2.0 * theta), np.cos(2.0 * theta)
    s4, c4 = np.sin(4.0 * theta), np.cos(4.0 * theta)
    return {
        'C11': -0.5 * (alpha * s4 + 2.0 * gamma * s2),   # dC11/dtheta
        'C12':  0.5 * alpha * s4,                        # dC13/dtheta
        'C13': -0.5 * (gamma * c2 + alpha * c4),         # dC15/dtheta
        'C22': -0.5 * (alpha * s4 - 2.0 * gamma * s2),   # dC33/dtheta
        'C23': -0.5 * (gamma * c2 - alpha * c4),         # dC35/dtheta
        'C33':  0.5 * alpha * s4,                        # dC55/dtheta
    }


def orientation_kernel(gradient_mesh, model_mesh, matl):
    """
    Sensitivity kernel w.r.t. grain orientation, K_theta = K_C :: dC/dtheta.

    gradient_mesh : Salvus gradient (element-nodal fields C11 ... C33). These
                    are already pointwise kernels K_C (the mass matrix is
                    stored separately as 'FemMassMatrix'), so the chain rule
                    is applied node by node.
    model_mesh    : model the gradient was computed in (gives theta(x))

    Returns the element-nodal K_theta array [per rad], same shape as the fields.
    """
    dC = dC_dtheta(theta_from_mesh(model_mesh), matl)
    return sum(gradient_mesh.element_nodal_fields[k] * dC[k] for k in dC)


# =============================================================================
# Salvus helpers
# =============================================================================

def forward_simulation(simulation_name, events):
    """Launch a forward simulation (receiver data only) in project `p` and block until it finishes."""
    _n_events = 1 if isinstance(events, str) else len(events)
    print(f"  [forward_simulation] launching '{simulation_name}' for {_n_events} event(s) ...")

    t_start = time.perf_counter()
    p.simulations.launch(
        ranks_per_job=RANKS,
        site_name=SITE_NAME,
        events=events,
        simulation_configuration=simulation_name,
        delete_conflicting_previous_results=True,
    )

    # Wait for THIS simulation only (an untargeted query also polls stale jobs,
    # e.g. from interrupted runs, and can fail on them). query(block=True)
    # occasionally raises while it polls a running job; keep waiting instead.
    while True:
        try:
            p.simulations.query(simulation_configuration=simulation_name, events=events,
                                block=True, verbosity=-1)
            break
        except (AssertionError, ValueError) as e:
            print(f"  [forward_simulation] query retry after {type(e).__name__}: {e}")
            time.sleep(2)
    elapsed = time.perf_counter() - t_start
    print(f"  [forward_simulation] '{simulation_name}' finished in "
          f"{elapsed:.1f} s ({elapsed / 60:.2f} min) wall-clock.")


def extract_data(simulation, events, receiver_field='displacement'):
    """Receiver waveforms (xarray) of a finished simulation for the first event in `events`."""
    ed = p.waveforms.get(data_name=simulation, events=events)[0]
    return ed.get_waveform_data_xarray(receiver_field=receiver_field)


# =============================================================================
# Project, true model and mesh
# =============================================================================

domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)

p = sn.Project.from_domain(
    path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True
)

# Unrotated hexagonal material built from the austenite constants
material_unoriented = sn.material.elastic.hexagonal.TensorComponents.from_params(
    **matl.VTI_parameters()
)

matl_background = oriented_material(material_unoriented, BACKGROUND_ANGLE_DEG)
matl_layer      = oriented_material(material_unoriented, LAYER_ANGLE_DEG)

# Layers listed top to bottom: background / oriented layer / background
m_true = lm.LayeredModel(
    [
        matl_background,
        sn.layered_meshing.interface.Hyperplane.at(LAYER_TOP_Y),
        matl_layer,
        sn.layered_meshing.interface.Hyperplane.at(LAYER_BOTTOM_Y),
        matl_background,
    ]
)

ab_params = salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
    free_surface=free_surfaces,
    number_of_wavelengths=number_of_wavelengths,
    reference_velocity=ab_reference_velocity,
    reference_frequency=reference_frequency,
)

mesh_res = sn.MeshResolution(
    reference_frequency=reference_frequency,
    elements_per_wavelength=elements_per_wavelength,
    model_order=model_order,
)

# True (layered) model
mesh = lm.mesh_from_domain(
    domain=domain,
    model=sn.layered_meshing.MeshingProtocol(m_true, ab=ab_params),
    mesh_resolution=mesh_res,
)

# Starting model: same mesh, homogeneous background orientation everywhere
mesh_init = mesh.copy()
for k in STIFFNESS_PARAMETERS:
    mesh_init.element_nodal_fields[k][:] = getattr(matl_background, k).p


# =============================================================================
# Sources, receivers and events
# =============================================================================

tx_locations = np.array([[0.005, 0.01]])   # source (x, y) [m], one row per source
r_loc_all    = np.array([[0.005, 0.0]])   # receiver (x, y) [m], shared by all events

receivers = [
    sn.simple_config.receiver.cartesian.Point2D(
        x=r[0], y=r[1], station_code=f"R{j+1}", fields=["displacement"]
    )
    for j, r in enumerate(r_loc_all)
]

# One event per source, each recorded by all receivers
event_ls = []
for i, s in enumerate(tx_locations):
    event_name = f"src_{i+1}"
    source = sn.simple_config.source.cartesian.VectorPoint2D(
        x=s[0], y=s[1], fx=0, fy=amplitude_ratio
    )
    p.add_to_project(
        sn.Event(event_name=event_name, sources=source, receivers=receivers)
    )
    event_ls.append(event_name)
print(f"{len(event_ls)} event(s) registered: {event_ls}")


# =============================================================================
# Figure style (project journal format, my_code/plot_style.py) and the
# simulation configuration: true orientation field with source and receivers
# =============================================================================

apply_paper_style()

for _d in (FWI_IMAGE_DIR, FWI_DATA_DIR, PAPER_FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

fig, _ = plot_orientation_field(mesh, tx_locations, r_loc_all, extent=(x0, x1, y0, y1),
                                save_path=PAPER_FIG_DIR / "FWI_orientation_configuration")
plt.close(fig)


# =============================================================================
# FWI settings
# =============================================================================

# Multiscale continuation: centre frequency = alpha * CENTRAL_FREQUENCY, low -> high.
# alpha <= 1 keeps every stage within the mesh resolution (reference_frequency = 2 f_c).
alpha_ls           = [0.4, 0.6, 0.8, 1.0]       # f_c = alpha * CENTRAL_FREQUENCY
# Simulation end time per stage: end_time is the alpha = 1 value, scaled with the period
end_time_ls        = [end_time / alpha for alpha in alpha_ls]   # 12.5, 8.33, 6.25, 5 us
MAX_ITER_PER_STAGE = 10                        # max accepted L-BFGS iterations per stage

# Inverted region = the middle layer only; the background stays at its known orientation.
# x spans the whole mesh (incl. absorbing layers): the layer continues there, and the
# x-averaged theta(y) prior would otherwise be diluted by fixed background nodes.
ROI = (mesh_init.points[:, 0].min(), mesh_init.points[:, 0].max(),
       LAYER_BOTTOM_Y, LAYER_TOP_Y)             # (x0, x1, y0, y1) [m]
THETA_MIN, THETA_MAX = np.deg2rad(0.0), np.deg2rad(90.0)   # orientation bounds [rad]
MODEL_SMOOTH_SIGMA   = 1.0                      # Gaussian smoothing of theta(y) [y-levels]

# Trust region in theta [rad] (L-inf ball)
DELTA_TR_INIT = np.deg2rad(5.0)
DELTA_TR_MAX  = np.deg2rad(20.0)

DIAG_EVENT = event_ls[len(event_ls) // 2]       # source shown in the diagnostic figures
print(f"DIAG_EVENT = {DIAG_EVENT!r}")


# =============================================================================
# Orientation -> stiffness
# =============================================================================

def stiffness_from_theta(theta, matl):
    """
    Rotated 2D stiffness for orientation theta [rad] (ClockwiseAngle convention).

    Closed forms whose theta-derivatives are exactly dC_dtheta():
        C11 = beta/8 + gamma/2 cos2t + alpha/8 cos4t
        C13 = (Cb11 + 6 Cb13 + Cb33 - 4 Cb55)/8 - alpha/8 cos4t
        C15 = -gamma/4 sin2t - alpha/8 sin4t
        C33 = beta/8 - gamma/2 cos2t + alpha/8 cos4t
        C35 = -gamma/4 sin2t + alpha/8 sin4t
        C55 = (Cb11 - 2 Cb13 + Cb33 + 4 Cb55)/8 - alpha/8 cos4t
    Keys are Salvus 2D Voigt names (C11, C12, C13, C22, C23, C33).
    """
    Cb11, Cb13, Cb33, Cb55 = matl.C11, matl.C13, matl.C33, matl.C44
    a = Cb11 - 2.0 * Cb13 + Cb33 - 4.0 * Cb55
    b = 3.0 * Cb11 + 2.0 * Cb13 + 3.0 * Cb33 + 4.0 * Cb55
    g = Cb11 - Cb33
    s2, c2 = np.sin(2.0 * theta), np.cos(2.0 * theta)
    s4, c4 = np.sin(4.0 * theta), np.cos(4.0 * theta)
    return {
        'C11': b / 8.0 + g / 2.0 * c2 + a / 8.0 * c4,
        'C12': (Cb11 + 6.0 * Cb13 + Cb33 - 4.0 * Cb55) / 8.0 - a / 8.0 * c4,
        'C13': -g / 4.0 * s2 - a / 8.0 * s4,
        'C22': b / 8.0 - g / 2.0 * c2 + a / 8.0 * c4,
        'C23': -g / 4.0 * s2 + a / 8.0 * s4,
        'C33': (Cb11 - 2.0 * Cb13 + Cb33 + 4.0 * Cb55) / 8.0 - a / 8.0 * c4,
    }


# consistency check against Salvus' own rotation (the true-model layer material)
_C_check = stiffness_from_theta(np.deg2rad(LAYER_ANGLE_DEG), matl)
for _k in STIFFNESS_PARAMETERS:
    assert np.isclose(_C_check[_k], getattr(matl_layer, _k).p, rtol=1e-9, atol=1e-3), _k


# =============================================================================
# Geometry-only setup (frequency independent)
# =============================================================================

# Starting model = homogeneous background orientation (mesh_init); same mesh as the true model
base_mesh = copy.deepcopy(mesh_init)

_x, _y = base_mesh.points[:, 0], base_mesh.points[:, 1]
x0_roi, x1_roi, y0_roi, y1_roi = ROI
_tol = 1e-9   # [m] keeps the interface nodes inside the ROI
roi_mask = ((_x >= x0_roi - _tol) & (_x <= x1_roi + _tol) &
            (_y >= y0_roi - _tol) & (_y <= y1_roi + _tol))
_x_roi, _y_roi = _x[roi_mask], _y[roi_mask]

# Elements inside the layer: only these take the inverted theta. The mesh
# conforms to the interfaces, so theta stays discontinuous there — background
# elements sharing an interface node keep the background orientation.
_cen_y = base_mesh.get_element_centroid()[:, 1]
_roi_elems = (_cen_y > y0_roi) & (_cen_y < y1_roi)
_conn_roi  = base_mesh.connectivity[_roi_elems]
# Share of each node's mass that belongs to layer elements (1 inside, ~1/2 on the
# interfaces, 0 outside): only the layer elements take the inverted theta
_n_elem_per_node  = np.bincount(base_mesh.connectivity.ravel(), minlength=base_mesh.npoint)
_n_layer_per_node = np.bincount(_conn_roi.ravel(), minlength=base_mesh.npoint)
_layer_frac = _n_layer_per_node / np.maximum(_n_elem_per_node, 1)
_theta_bg_elemental = theta_from_mesh(base_mesh)   # element-nodal background theta
np.save(FWI_DATA_DIR / "roi_node_xy.npy", np.column_stack([_x_roi, _y_roi]))

# Nodal theta of the starting model (background value everywhere)
_theta_bg_nodal = elemental_nodal_to_nodal_field(theta_from_mesh(base_mesh), base_mesh.connectivity)
THETA_HOMO = float(_theta_bg_nodal.mean())
print(f"THETA_HOMO (from mesh_init) = {np.rad2deg(THETA_HOMO):.2f} deg")

# True theta(y), for plotting only
_theta_true_nodal = elemental_nodal_to_nodal_field(theta_from_mesh(mesh), mesh.connectivity)

# y-levels for the layered prior (coordinates are O(1e-3) m)
_y_round_full = np.round(_y, decimals=10)
_y_round_roi  = np.round(_y_roi, decimals=10)
_y_levels     = np.unique(_y_round_full)

# Full list of acquired (source, receiver) pairs
sr_pairs = [(tuple(s), tuple(r)) for s in tx_locations for r in r_loc_all]

_fwi_iter   = [0]
_fwi_ls     = [0]
_fwi_call   = [0]   # unique simulation names: Salvus caches results per configuration name
_diag_cache = {}    # filled by objective() for DIAG_EVENT, read by the optimizer for plotting
_W_roi      = [None]   # per-node level mass (L-BFGS preconditioner), set by objective()


def _average_theta_in_x(theta_nodal):
    """Layered prior theta(y): every node gets the mean theta of its y-level."""
    out = theta_nodal.copy()
    for yv in _y_levels:
        mask = _y_round_full == yv
        out[mask] = theta_nodal[mask].mean()
    return out


def _theta_profile(theta_nodal):
    """theta(y) profile (one value per y-level) of a nodal field."""
    return np.array([theta_nodal[_y_round_full == yv].mean() for yv in _y_levels])


def build_mesh_from_model(m_flat):
    """
    Decode theta (ROI) -> layered theta(y) -> stiffness, attached to a copy of the mesh.

    Returns
    -------
    mesh_it     : mesh with the rotated C11 ... C33 fields
    theta_nodal : (n_nodes,)      full (x-averaged) theta field [rad]
    theta_roi   : (n_roi_nodes,)  theta restricted to the ROI [rad]
    """
    theta_nodal = _theta_bg_nodal.copy()
    theta_nodal[roi_mask] = np.clip(m_flat, THETA_MIN, THETA_MAX)
    theta_nodal = _average_theta_in_x(theta_nodal)

    # only the layer elements change; the background elements keep their theta
    theta_en = _theta_bg_elemental.copy()
    theta_en[_roi_elems] = theta_nodal[_conn_roi]

    mesh_it = copy.deepcopy(base_mesh)
    C = stiffness_from_theta(theta_en, matl)
    for k in STIFFNESS_PARAMETERS:
        mesh_it.attach_field(k, C[k])
    return mesh_it, theta_nodal, theta_nodal[roi_mask]


def _ellipse_mask(xs, ys, pairs, margin):
    """Union of first-Fresnel-zone ellipses a = (|s - r| + margin) / 2 over all pairs."""
    mask = np.zeros_like(xs, dtype=bool)
    for s, r in pairs:
        d_sr = np.hypot(s[0] - r[0], s[1] - r[1])
        a = (d_sr + margin) / 2
        mask |= (np.hypot(xs - s[0], ys - s[1]) + np.hypot(xs - r[0], ys - r[1])) <= 2 * a
    return mask


# =============================================================================
# Diagnostics
# =============================================================================

def _kernel_axes(ax, K, xs, ys, title, xlim, ylim, label):
    """tripcolor of a nodal kernel with a symmetric (99th-percentile) colour scale."""
    vmax = np.percentile(np.abs(K), 99.0) or 1.0
    tp = ax.tripcolor(xs * mm, ys * mm, K, cmap='RdBu', shading='gouraud',
                      vmin=-vmax, vmax=vmax, rasterized=True)
    ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$z$ (mm)')
    ax.set_title(title)
    ax.set_xlim(*[v * mm for v in xlim]); ax.set_ylim(*[v * mm for v in ylim])
    ax.set_aspect('equal')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.08)
    plt.colorbar(tp, cax=cax, label=label)


def plot_fwi_diagnostics(it, u_obs, u_syn, f_adj, K_theta, K_roi, save=True):
    """Waveforms + adjoint source, full K_theta and masked ROI K_theta for iteration `it`."""
    t_us = u_obs.time.values * 1e6

    # ── Figure 1: observed vs synthetic and adjoint source, X and Y ────────
    fig1, axs = plt.subplots(2, 2, figsize=(7.0, 4.0), layout='constrained', sharex=True)
    for col, comp in enumerate(['X', 'Y']):
        axs[0, col].plot(t_us, u_syn.sel(component=comp).values.squeeze(), color=C_SYN, lw=0.9,
                         label=rf'$u^{{({it})}}_{comp.lower()}$')
        axs[0, col].plot(t_us, u_obs.sel(component=comp).values.squeeze(), color=C_OBS, lw=0.9,
                         ls='--', label=rf'$u_{{{comp.lower()},\mathrm{{obs}}}}$')
        axs[0, col].legend(frameon=False)
        axs[0, col].set_ylabel(f'$u_{comp.lower()}$ (m)')
        axs[0, col].set_title(f'({"ab"[col]}) Observed vs. synthetic, {comp} (iteration {it})')
        axs[1, col].plot(t_us, f_adj[col], color='0.3', lw=0.9)
        axs[1, col].axhline(0, color='k', lw=0.4, ls='--')
        axs[1, col].set_xlabel(r'Time ($\mu$s)')
        axs[1, col].set_ylabel(rf'$f^\dagger_{comp.lower()}$')
        axs[1, col].set_title(rf'({"cd"[col]}) Adjoint source $f^\dagger = d - u$, {comp}')
        for ax in axs[:, col]:
            ax.grid(True, **GRID_KW)
    if save:
        plt.savefig(FWI_IMAGE_DIR / f'iter_{it:03d}_waveforms_adjoint_source_alpha_{alpha}.png')
    plt.close(fig1)

    # ── Figure 2: full K_theta ──────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(3.5, 2.9))
    _kernel_axes(ax, K_theta, _x, _y, rf'$K_\theta$ (iteration {it})',
                 (x0, x1), (y0, y1), r'$K_\theta$')
    for (s, r) in sr_pairs:
        ax.scatter(s[0] * mm, s[1] * mm, marker='v', color='red', s=20, zorder=5)
        ax.scatter(r[0] * mm, r[1] * mm, marker='^', color='lime', s=12, zorder=5)
    fig2.subplots_adjust(left=0.13, right=0.80, bottom=0.14, top=0.92)
    if save:
        plt.savefig(FWI_IMAGE_DIR / f'iter_{it:03d}_kernel_theta_full_alpha_{alpha}.png')
    plt.close(fig2)

    # ── Figure 3: K_theta in the ROI, masked to the Fresnel zone ───────────
    fig3, ax = plt.subplots(figsize=(3.5, 2.9))
    _kernel_axes(ax, np.where(_ellipse_roi, K_roi, 0.0), _x_roi, _y_roi,
                 rf'$K_\theta$ in ROI (iteration {it})',
                 (x0, x1), (y0_roi, y1_roi), r'$K_\theta$')
    fig3.subplots_adjust(left=0.13, right=0.80, bottom=0.14, top=0.92)
    if save:
        plt.savefig(FWI_IMAGE_DIR / f'iter_{it:03d}_kernel_theta_roi_alpha_{alpha}.png')
    plt.close(fig3)


def plot_theta_model(theta_nodal, it, ls):
    """theta field and theta(y) profile (vs. the true profile) of one objective() call."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), layout='constrained')
    tp = axes[0].tripcolor(_x * mm, _y * mm, np.rad2deg(theta_nodal), cmap='viridis',
                           vmin=0.0, vmax=90.0, shading='gouraud', rasterized=True)
    fig.colorbar(tp, ax=axes[0], label=r'$\theta$ (deg)', shrink=0.9)
    axes[0].set_xlabel('$x$ (mm)'); axes[0].set_ylabel('$z$ (mm)')
    axes[0].set_title(rf'(a) $\theta$ field (iteration {it}, ls {ls})')
    axes[0].set_xlim(x0 * mm, x1 * mm); axes[0].set_ylim(y0 * mm, y1 * mm)
    axes[0].set_aspect('equal')
    axes[1].plot(np.rad2deg(_theta_profile(_theta_true_nodal)), _y_levels * mm,
                 color=C_TRUE, lw=1.0, ls='--', label='True')
    axes[1].plot(np.rad2deg(_theta_profile(theta_nodal)), _y_levels * mm,
                 color=C_SYN, marker='o', markersize=2, lw=1.0, label='Current')
    axes[1].axhspan(y0_roi * mm, y1_roi * mm, color='0.93', zorder=0, lw=0)
    axes[1].set_xlabel(r'$\theta$ (deg)'); axes[1].set_ylabel('$z$ (mm)')
    axes[1].set_title(rf'(b) $\theta(z)$ profile (iteration {it}, ls {ls})')
    axes[1].set_ylim(y0 * mm, y1 * mm)
    axes[1].legend(frameon=False)
    axes[1].grid(True, **GRID_KW)
    plt.savefig(FWI_IMAGE_DIR / f'iter_{it:03d}_ls_{ls}_theta_field_alpha_{alpha}.png')
    plt.close(fig)


# =============================================================================
# Objective: misfit and K_theta from Salvus' built-in adjoint gradient
# =============================================================================

def _wait_for_result(call, name, poll_seconds=5, max_errors=60):
    """
    Repeat a non-blocking Salvus action until it returns a result.

    compute_gradients / compute_misfits launch the next missing step (forward,
    then adjoint) and return None until the result exists. Transient job-store
    errors while polling (AssertionError / ValueError) are retried up to
    `max_errors` times in a row before being raised.
    """
    n_errors = 0
    while True:
        try:
            result = call()
            n_errors = 0
        except (AssertionError, ValueError) as e:
            n_errors += 1
            if n_errors > max_errors:
                raise
            print(f"  [{name}] retry {n_errors}/{max_errors} after {type(e).__name__}: {e}")
            result = None
        if result is not None:
            return result
        time.sleep(poll_seconds)

def objective(m_flat):
    """
    Multi-source FWI objective — model parameter is theta [rad] at the ROI nodes.

    Salvus runs forward + adjoint for all events and returns K_C summed over
    events; K_theta = K_C :: dC/dtheta is evaluated at the current theta(x).

    Returns
    -------
    misfit : float                           total L2 misfit, summed over sources
    grad   : np.ndarray, (n_roi_nodes,)      K_theta at the ROI nodes, zeroed outside the Fresnel zone
    """
    it, ls = _fwi_iter[0], _fwi_ls[0]
    _fwi_ls[0] += 1
    _fwi_call[0] += 1
    sim_name = f'fwi_theta_a{stage_idx}_{it:03d}_ls{ls}_c{_fwi_call[0]:03d}'
    print(f"\n=== objective() call: alpha={alpha} iter={it} ls={ls} sim_name={sim_name!r} ===")

    # ── 1. Decode theta, enforce theta(y), build the mesh ──────────────────
    mesh_it, theta_nodal, theta_roi = build_mesh_from_model(m_flat)
    print(f'  [obj] theta(y) in ROI: mean={np.rad2deg(theta_roi.mean()):.2f} deg  '
          f'range=[{np.rad2deg(theta_roi.min()):.2f}, {np.rad2deg(theta_roi.max()):.2f}] deg')
    plot_theta_model(theta_nodal, it, ls)

    p.add_to_project(
        sn.UnstructuredMeshSimulationConfiguration(
            unstructured_mesh=mesh_it,
            name=sim_name,
            event_configuration=event_config_fwd,
        ),
        overwrite=True,
    )

    # ── 2. Forward + adjoint (Salvus built-in), all sources ────────────────
    t_start = time.perf_counter()
    _wait_for_result(lambda: p.actions.inversion.compute_gradients(
        simulation_configuration=sim_name,
        misfit_configuration=misfit_name,
        events=event_ls,
        site_name=SITE_NAME,
        ranks_per_job=RANKS,
        gradient_parameters=STIFFNESS_PARAMETERS,
    ), "compute_gradients")
    misfits = _wait_for_result(lambda: p.actions.inversion.compute_misfits(
        simulation_configuration=sim_name,
        misfit_configuration=misfit_name,
        events=event_ls,
        site_name=SITE_NAME,
        ranks_per_job=RANKS,
    ), "compute_misfits")
    print(f'  [obj] forward + adjoint done in {time.perf_counter() - t_start:.1f} s')
    misfit_total = float(sum(misfits.values()))

    gradient_C = p.actions.inversion.sum_gradients(
        simulation_configuration=sim_name,
        misfit_configuration=misfit_name,
        events=event_ls,
        gradient_parameters=STIFFNESS_PARAMETERS,
    )

    # ── 3. K_theta = K_C :: dC/dtheta at the current theta(x) ──────────────
    K_theta_en = orientation_kernel(gradient_C, mesh_it, matl)
    K_theta = elemental_nodal_to_nodal_field(K_theta_en, mesh_it.connectivity)   # full field, for plots
    # gradient w.r.t. the layer theta: average over the layer elements only, so
    # interface nodes do not pick up the background elements' dC/dtheta
    K_theta_layer = np.zeros_like(K_theta)
    _k = elemental_nodal_to_nodal_field(K_theta_en[_roi_elems], _conn_roi)
    K_theta_layer[:len(_k)] = _k

    # ── Diagnostics for DIAG_EVENT: synthetic and adjoint source ───────────
    u_syn = extract_data(sim_name, DIAG_EVENT, receiver_field='displacement')
    event_misfit = sn.EventMisfit(
        observed_event=p.waveforms.get(data_name=obs_simulation, events=DIAG_EVENT)[0],
        synthetic_event=p.waveforms.get(data_name=sim_name, events=DIAG_EVENT)[0],
        misfit_function="L2",
        receiver_field="displacement",
    )
    adj = event_misfit.get_misfit_and_adjoint_source_for_receiver(event_misfit.receiver_name_list[0])
    # Salvus stores -(u - d) * dt; plot the continuous adjoint force d - u
    f_adj = adj['adjoint_sources'].values.astype(float) / (1.0 / sampling_rate_in_hertz)
    _diag_cache.update(u_syn=u_syn, f_adj=f_adj, K_theta=K_theta, K_roi=K_theta[roi_mask])

    # ── 4. Fresnel-zone mask, then project onto the layered theta(y) space ─
    # K_theta is a pointwise kernel density [misfit / rad / m^2]; the discrete
    # gradient is d chi / d theta_i = K_theta(x_i) * M_i with the lumped nodal
    # mass M_i (FemMassMatrix: assembled quadrature weight * |J|, no density),
    # restricted to the layer elements' share. Without it pred = -g.dm is
    # ~1/M ~ 1e9 too large, rho ~ 0 and every step is rejected.
    # The model is theta(y) (x-averaged), so the gradient must be too: sum
    # K_theta * M over each y-level inside the Fresnel zone and spread it over
    # the level's nodes, so that -g.dm is the first-order misfit change of a
    # uniform level update dm.
    # The level masses follow the GLL weights (ratio ~3.6 between neighbouring
    # node rows), so g itself zigzags in y; W (level mass per node) is kept for
    # the L-BFGS preconditioner W^-1, whose direction W^-1 g is the smooth
    # mass-weighted mean of K_theta per level.
    M_nodal = np.zeros(base_mesh.npoint)
    M_nodal[base_mesh.connectivity] = gradient_C.element_nodal_fields['FemMassMatrix']
    M_roi = (M_nodal * _layer_frac)[roi_mask]
    grad_nodes = K_theta_layer[roi_mask] * M_roi
    grad = np.zeros_like(grad_nodes)
    W = np.zeros_like(grad_nodes)
    for yv in np.unique(_y_round_roi):
        level = _y_round_roi == yv
        sensed = level & _ellipse_roi
        W[level] = M_roi[level].sum() / level.sum()
        if sensed.any():
            grad[level] = grad_nodes[sensed].sum() / level.sum()
    _W_roi[0] = np.maximum(W, W.max() * 1e-6)   # guard against zero-mass levels

    label = 'grad' if ls == 0 else f'ls_{ls}'
    print(f"  iter {it:3d} {label:6s} | misfit = {misfit_total:.6e}"
          f" | |g| max = {np.abs(grad).max():.3e}")
    return misfit_total, grad


# =============================================================================
# L-BFGS + trust region
# =============================================================================

def run_lbfgs_trust_region(max_iter=10, max_tries=10, M=10,
                           delta_tr_init=DELTA_TR_INIT, delta_max=DELTA_TR_MAX,
                           eta1=0.0001, eta2=0.75,
                           gamma1=0.5, gamma2=0.9, sigma=2.0,
                           theta_init=None):
    """
    L-BFGS + trust region FWI optimizer for theta [rad] (see the acoustic script).

    Steps and radii are in radians (L-inf ball). After each accepted step m is
    projected onto the layered theta(y) space and the profile is Gaussian-
    smoothed (sigma = MODEL_SMOOTH_SIGMA y-levels). `max_iter` counts accepted
    steps; the gradient is cached across rejected steps.

    `theta_init` (ROI theta, (n_roi_nodes,)) warm-starts a frequency stage;
    defaults to the homogeneous starting theta.

    Returns
    -------
    misfit_history    : list[float],              length n_accepted + 1
    model_history     : (n_accepted + 1, n_roi_nodes)     theta [rad]
    synthetic_history : (n_accepted + 1, 2, n_time)       DIAG_EVENT synthetic (X, Y)
    adjoint_history   : (n_accepted + 1, 2, n_time)       DIAG_EVENT adjoint source (X, Y)
    """
    print(f"Starting L-BFGS trust-region optimization: "
          f"max_iter={max_iter}, max_tries={max_tries}, M={M}")

    def _mesh_theta_roi(theta_roi_vals):
        """ROI theta after the full-mesh x-averaging applied in objective()."""
        _t = _theta_bg_nodal.copy()
        _t[roi_mask] = theta_roi_vals
        return np.clip(_average_theta_in_x(_t)[roi_mask], THETA_MIN, THETA_MAX)

    def _diag_plots(it):
        plot_fwi_diagnostics(it=it, u_obs=u_obs_dict[DIAG_EVENT], u_syn=_diag_cache['u_syn'],
                             f_adj=_diag_cache['f_adj'], K_theta=_diag_cache['K_theta'],
                             K_roi=_diag_cache['K_roi'])

    delta_tr = delta_tr_init
    m = (np.full(roi_mask.sum(), THETA_HOMO) if theta_init is None
         else np.clip(theta_init, THETA_MIN, THETA_MAX))
    s_list, y_list, rho_list = [], [], []
    prev_grad = None
    prev_s    = None

    _fwi_iter[0] = 0
    _fwi_ls[0]   = 0

    # Starting model = iteration-0 baseline (seeds all histories together)
    print("  evaluating objective() for the starting model (iteration 0 baseline) ...")
    misfit0, grad0 = objective(m)

    misfit_history    = [float(misfit0)]
    model_history     = [_mesh_theta_roi(m)]
    synthetic_history = [np.array(_diag_cache['u_syn'])]
    adjoint_history   = [np.array(_diag_cache['f_adj'])]
    cached_misfit, cached_grad = misfit0, grad0
    _diag_plots(0)

    k     = 0   # accepted iteration counter
    tries = 0   # total attempt counter

    while k < max_iter and tries < max_tries:
        tries += 1
        _fwi_iter[0] = k
        _fwi_ls[0]   = 0

        print(f"\n--- L-BFGS attempt {tries} (accepted so far: {k}/{max_iter}) ---")
        if cached_misfit is not None:
            misfit, grad = cached_misfit, cached_grad
            print("  [cache] reusing gradient from previous step")
        else:
            misfit, grad = objective(m)

        # ── Update L-BFGS memory ────────────────────────────────────────────
        if prev_grad is not None and prev_s is not None:
            y_k = grad - prev_grad
            sy  = np.dot(prev_s, y_k)
            if sy > 0:
                s_list.append(prev_s.copy())
                y_list.append(y_k.copy())
                rho_list.append(1.0 / sy)
                if len(s_list) > M:
                    s_list.pop(0); y_list.pop(0); rho_list.pop(0)

        # ── L-BFGS two-loop recursion, H0 = gamma W^-1 (inverse level mass) ─
        W_inv = 1.0 / _W_roi[0]
        q, alphas = grad.copy(), []
        for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a = rho * np.dot(s, q); q -= a * y; alphas.append(a)
        gamma_h = (np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], W_inv * y_list[-1])
                   if s_list else 1.0)
        r = gamma_h * W_inv * q
        for s, y, rho, a in zip(s_list, y_list, rho_list, reversed(alphas)):
            r += s * (a - rho * np.dot(y, r))
        d = -r

        # ── Constrain step to the trust region (L-inf ball, radians) ────────
        d_max = np.abs(d).max()
        if d_max < 1e-30:
            print(f"iter {k} | direction vanished — stopped"); break
        step        = delta_tr * (d / d_max)
        m_new       = np.clip(m + step, THETA_MIN, THETA_MAX)
        actual_step = m_new - m

        # ── Evaluate proposed model ─────────────────────────────────────────
        _fwi_ls[0] = 1
        print("  evaluating objective() for the trial step ...")
        misfit_new, grad_new = objective(m_new)

        # ── rho = actual / predicted reduction ──────────────────────────────
        actual_red    = misfit - misfit_new
        predicted_red = -np.dot(grad, actual_step)
        rho_k = actual_red / predicted_red if abs(predicted_red) > 1e-30 else 0.0
        step_norm = np.abs(actual_step).max()

        # ── Accept / reject ─────────────────────────────────────────────────
        if rho_k >= eta1:
            prev_grad = grad.copy()

            # Project onto theta(y), then Gaussian-smooth the depth profile
            _t_acc      = np.clip(m_new, THETA_MIN, THETA_MAX)
            _yv_unique  = np.unique(_y_round_roi)
            _t_profile  = np.array([_t_acc[_y_round_roi == _yv].mean() for _yv in _yv_unique])
            _t_profile  = gaussian_filter1d(_t_profile, sigma=MODEL_SMOOTH_SIGMA)
            for _yv, _t_smooth in zip(_yv_unique, _t_profile):
                _t_acc[_y_round_roi == _yv] = _t_smooth
            m_proj = np.clip(_t_acc, THETA_MIN, THETA_MAX)

            prev_s        = m_proj - m
            m             = m_proj
            cached_misfit = misfit_new
            cached_grad   = grad_new
            misfit_history.append(misfit_new)
            model_history.append(_mesh_theta_roi(m))
            synthetic_history.append(np.array(_diag_cache['u_syn']))
            adjoint_history.append(np.array(_diag_cache['f_adj']))
            status = "accepted"
            k += 1
            _diag_plots(k)
        else:
            cached_misfit = misfit
            cached_grad   = grad
            status = "REJECTED"

        # ── Update trust-region radius (Nocedal & Wright §4.1) ──────────────
        if rho_k < eta1:
            delta_tr = min(delta_tr / sigma, step_norm)
        elif rho_k > eta2 and step_norm >= gamma2 * delta_tr:
            delta_tr = min(sigma * delta_tr, delta_max)
        elif step_norm < gamma1 * delta_tr:
            delta_tr = max(delta_tr / sigma, sigma * step_norm)

        theta_mean = np.rad2deg(_mesh_theta_roi(m)[_ellipse_roi].mean())
        print(f"Iteration {k}: n_events={len(event_ls)}  theta_mean={theta_mean:.2f} deg  "
              f"chi={misfit:.6e}  ||g||={np.linalg.norm(grad):.3e}")
        print(f"  pred={predicted_red:.3e}  ared={actual_red:.3e}  rho={rho_k:.3f}  "
              f"||dm||_inf={np.rad2deg(step_norm):.3f} deg  "
              f"delta_tr={np.rad2deg(delta_tr):.3f} deg  ({status})")

    print(f"L-BFGS finished: {k} accepted iteration(s), {tries} total attempt(s).")
    assert (len(misfit_history) == len(model_history) == len(synthetic_history)
            == len(adjoint_history))
    return (misfit_history, np.array(model_history),
            np.array(synthetic_history), np.array(adjoint_history))


# =============================================================================
# Multi-frequency (multiscale) FWI driver
# =============================================================================

all_misfit_history    = []
all_model_history     = []
all_synthetic_history = []
all_adjoint_history   = []
all_observed_history  = []   # DIAG_EVENT observed traces, one per stage
stage_boundaries      = [0]  # cumulative accepted-iteration index at the start of each stage
stage_summary         = []   # (alpha, f_c, n_accepted)

theta_warm_start = None      # None -> first stage starts from the homogeneous theta

for stage_idx, (alpha, end_time_stage) in enumerate(zip(alpha_ls, end_time_ls)):
    f_c = alpha * CENTRAL_FREQUENCY
    # Ricker peaks at t = 0 -> start >= 1.5 periods early (no switch-on ringing).
    # Snap start/end to the dt grid: a fractional step count makes Salvus round
    # observed and synthetic traces to different lengths (off-by-one sample).
    start_time_stage = -np.ceil(1.5 / f_c / time_step_in_seconds) * time_step_in_seconds
    end_time_stage   = np.round(end_time_stage / time_step_in_seconds) * time_step_in_seconds
    print(f"\n{'='*70}\nFrequency stage {stage_idx + 1}/{len(alpha_ls)}: "
          f"alpha={alpha}  f_c={f_c / 1e6:.2f} MHz  "
          f"t=[{start_time_stage * 1e6:.3f}, {end_time_stage * 1e6:.3f}] us\n{'='*70}")

    event_config_fwd = sn.EventConfiguration(
        wavelet=sn.simple_config.stf.Ricker(center_frequency=f_c),
        waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
            start_time_in_seconds=start_time_stage,
            end_time_in_seconds=end_time_stage,
            time_step_in_seconds=time_step_in_seconds,
        ),
    )

    # Observed data = true (layered) model at this stage's frequency
    obs_simulation = f'true_model_a{stage_idx}'
    misfit_name    = f'L2_theta_a{stage_idx}'
    p.add_to_project(
        sn.UnstructuredMeshSimulationConfiguration(
            unstructured_mesh=mesh, name=obs_simulation, event_configuration=event_config_fwd,
        ),
        overwrite=True,
    )
    p.add_to_project(
        sn.MisfitConfiguration(
            name=misfit_name,
            observed_data=f"SYNTHETIC_DATA:{obs_simulation}",
            misfit_function="L2",
            receiver_field="displacement",
        ),
        overwrite=True,
    )

    t0 = datetime.now()
    forward_simulation(obs_simulation, event_ls)
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"Stage {stage_idx} observed data done in {int(elapsed // 60)}m {elapsed % 60:.1f}s")

    u_obs_dict = {ev: extract_data(obs_simulation, ev, receiver_field='displacement')
                  for ev in event_ls}

    # Fresnel-zone mask: margin = half a P wavelength at this stage's frequency
    fresnel_margin = 0.5 * VP_0deg / f_c
    _ellipse_roi = _ellipse_mask(_x_roi, _y_roi, sr_pairs, fresnel_margin)
    print(f"Stage {stage_idx} ellipse mask: {_ellipse_roi.sum()} / {roi_mask.sum()} "
          f"ROI nodes inside the Fresnel zone (margin {fresnel_margin * mm:.2f} mm)")

    (misfit_history_stage, model_history_stage,
     synthetic_history_stage, adjoint_history_stage) = run_lbfgs_trust_region(
        max_iter=MAX_ITER_PER_STAGE, theta_init=theta_warm_start,
    )
    theta_warm_start = model_history_stage[-1]

    # Row 0 of later stages duplicates the previous stage's last model -> keep stage 0's only
    _kept = slice(None) if stage_idx == 0 else slice(1, None)
    all_misfit_history.extend(misfit_history_stage[_kept])
    all_model_history.append(model_history_stage[_kept])
    all_synthetic_history.append(synthetic_history_stage[_kept])
    all_adjoint_history.append(adjoint_history_stage[_kept])
    all_observed_history.append(np.array(u_obs_dict[DIAG_EVENT]))
    stage_boundaries.append(stage_boundaries[-1] + len(misfit_history_stage[_kept]))
    stage_summary.append((alpha, f_c, len(misfit_history_stage[_kept])))

    np.save(FWI_DATA_DIR / f"misfit_history_alpha_{alpha}.npy", np.array(misfit_history_stage))
    np.save(FWI_DATA_DIR / f"model_history_theta_rad_alpha_{alpha}.npy", model_history_stage)
    np.save(FWI_DATA_DIR / f"synthetic_signal_history_alpha_{alpha}.npy", synthetic_history_stage)
    np.save(FWI_DATA_DIR / f"adjoint_signal_history_alpha_{alpha}.npy", adjoint_history_stage)
    np.save(FWI_DATA_DIR / f"observed_signal_alpha_{alpha}.npy", np.array(u_obs_dict[DIAG_EVENT]))
    print(f"Stage {stage_idx} (alpha={alpha}, f_c={f_c / 1e6:.2f} MHz): "
          f"{len(misfit_history_stage)} model(s) saved.")

print(f"\nMulti-frequency FWI finished: {stage_summary}")


# =============================================================================
# Summary figures and results
# =============================================================================

# ── Misfit chi vs iteration (log scale, one band per frequency stage) ──
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for _k, (_alpha, _f_c, _n) in enumerate(stage_summary):
    _lo, _hi = stage_boundaries[_k] - 0.5, stage_boundaries[_k + 1] - 0.5
    if _k % 2 == 1:
        ax.axvspan(_lo, _hi, color='0.93', zorder=0, linewidth=0)
    ax.text(0.5 * (_lo + _hi), 1.0, f"{_f_c / 1e6:.1f} MHz", transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=7)
ax.semilogy(np.arange(len(all_misfit_history)), all_misfit_history, color=C_SYN, marker='o',
            linewidth=1.0, markersize=3, markeredgecolor='black', markeredgewidth=0.4, zorder=3)
ax.set_xlabel("FWI iteration")
ax.set_ylabel(r"$\chi$")
ax.yaxis.grid(True, which='major', **GRID_KW)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
plt.savefig(FWI_IMAGE_DIR / "misfit_history_multiscale.png")
save_paper_figure(fig, PAPER_FIG_DIR / "FWI_orientation_misfit_vs_iteration", dpi=600)
plt.close(fig)

_all_model_history = np.concatenate(all_model_history, axis=0)

# ── Combined histories ──────────────────────────────────────────────────────
np.save(FWI_DATA_DIR / "misfit_history_multiscale.npy", np.array(all_misfit_history))
np.save(FWI_DATA_DIR / "model_history_theta_rad_multiscale.npy", _all_model_history)
# Trace length differs per stage (start/end time depend on f_c) -> one npz key per stage
_stage_keys = [f"alpha_{_alpha}" for _alpha in alpha_ls]
np.savez(FWI_DATA_DIR / "synthetic_signal_history_multiscale.npz",
         **dict(zip(_stage_keys, all_synthetic_history)))
np.savez(FWI_DATA_DIR / "adjoint_signal_history_multiscale.npz",
         **dict(zip(_stage_keys, all_adjoint_history)))
np.savez(FWI_DATA_DIR / "observed_signal_history_multiscale.npz",
         **dict(zip(_stage_keys, all_observed_history)))
print(f"Saved misfit/model/synthetic/adjoint histories ({len(all_misfit_history)} entries) "
      f"and observed traces ({len(all_observed_history)} stage(s)) to {FWI_DATA_DIR}")


# ── Model error vs the true model (RMS / relative theta error, profile, layers) ──
save_model_error_figures(
    _all_model_history, np.column_stack([_x_roi, _y_roi]), alpha_ls, np.diff(stage_boundaries),
    CENTRAL_FREQUENCY, [LAYER_TOP_Y, LAYER_BOTTOM_Y], [LAYER_ANGLE_DEG], BACKGROUND_ANGLE_DEG, (x0, x1),
    PAPER_FIG_DIR,
)
