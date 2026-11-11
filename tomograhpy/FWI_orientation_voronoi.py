"""
Elastic FWI for the grain orientation theta(x, z) of a Voronoi polycrystal of austenite grains,
multi-source acquisition with transducers on all four sides of a finite specimen.

Forward configuration as forward_2D_voronoi.py:
  * 10-grain Voronoi map (seed 300, PNG segmented by VoronoiGrainIndexer), grain orientations
    ~ N(45, 15) deg (seed 0), set per GLL node on a regular (non-conforming) box mesh
  * finite specimen: free surfaces on all 4 sides, no absorbing layer
  * 36 transducers, 9 per side at 1 ... 9 mm. Every transducer in FIRING_SIDES x FIRING_INDICES
    is a separate event: it transmits (force normal to its side) and the 27 transducers on the
    other 3 sides receive

FWI as FWI_orientation_multilayer_multisource.py, without the layered prior:
  * gradient   : Salvus built-in adjoint gradient K_C (compute_gradients / sum_gradients),
                 summed over the events, mapped to K_theta = K_C :: dC/dtheta
  * model      : theta at every mesh node (the whole specimen, grain boundaries NOT assumed
                 known); no x-averaging, theta(x, z) is free in 2D
  * optimizer  : L-BFGS + trust region (L-inf ball in theta, radians)
  * mask       : gradient zeroed outside the first-Fresnel-zone ellipses
  * H0         : 2D Gaussian smoothing of the update (on a regular grid, sigma ~ shear wavelength)
                 and a 2D illumination preconditioner 1/E(x, z)
  * multiscale : frequency stages alpha * CENTRAL_FREQUENCY, warm-started;
                 simulation end time end_time / alpha per stage

Checkpoint / restart: the FWI state is saved after every L-BFGS attempt (data/.../checkpoint.pkl);
running the script again resumes from it. run_FWI_orientation_voronoi.sh resubmits itself before
the 24 h limit. FWI_FRESH_START=1 discards the checkpoint and starts a new run.

Self-contained: paths, Voronoi map, true model, mesh, events and the starting model (mesh_init,
homogeneous THETA_MEAN_DEG) are all set up below. Run from the tomograhpy folder:
    python FWI_orientation_voronoi.py                      # resumes if a checkpoint exists
    FWI_FRESH_START=1 python FWI_orientation_voronoi.py    # new run

Units: SI throughout (m, s, Pa, kg/m^3); plots use mm.
"""

import copy
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file-only backend: figures are saved, never shown (no X server needed)
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi

import salvus.mesh.layered_meshing as lm
import salvus.namespace as sn
from salvus.material._details import material as md

from my_code.utilities import VoronoiGrainIndexer, elemental_nodal_to_nodal_field
from my_code.model_error import save_grain_model_error_figures
from my_code.plot_style import (apply_paper_style, save_paper_figure, plot_orientation_field,
                                add_colorbar, C_OBS, C_SYN, GRID_KW, MM as mm)
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

PROJECT_NAME = "tomography_solid_orientation_voronoi_fwi"   # folder inside PROJECT_DIR_WIN

FWI_IMAGE_DIR = Path(IMAGE_DIR_WIN) / "FWI_orientation_voronoi"
FWI_DATA_DIR  = Path(DATA_DIR_WIN) / "FWI_orientation_voronoi"
PAPER_FIG_DIR = FWI_IMAGE_DIR / "paper_figures"
VORONOI_DATA  = Path(DATA_DIR_WIN) / "voronoi"   # grain map PNG, shared with forward_2D_voronoi.py

# Checkpoint / restart (the GPU allocation is limited to 24 h): the FWI state is saved to
# CHECKPOINT after every L-BFGS attempt and every finished frequency stage, and a new run of this
# script resumes from it (same Salvus project, finished simulations are reused). A run starts from
# scratch, deleting this script's previous data, images and Salvus project, when there is no
# checkpoint or when FWI_FRESH_START=1 is set in the environment.
CHECKPOINT  = FWI_DATA_DIR / "checkpoint.pkl"
FRESH_START = os.environ.get('FWI_FRESH_START', '0') == '1' or not CHECKPOINT.exists()
# Unique per run: simulations interrupted by the end of an allocation are not reused, the
# resumed run launches them again under a new name
RUN_TAG     = os.environ.get('SLURM_JOB_ID', datetime.now().strftime('%Y%m%d%H%M%S'))

ckpt = None
if FRESH_START:
    for _d in (FWI_IMAGE_DIR, FWI_DATA_DIR, Path(PROJECT_DIR_WIN, PROJECT_NAME)):
        if _d.exists():
            print(f"Deleting previous results: {_d}")
            shutil.rmtree(_d)
else:
    with open(CHECKPOINT, 'rb') as _fh:
        ckpt = pickle.load(_fh)
    if ckpt['finished']:
        print(f"FWI already finished ({CHECKPOINT}); set FWI_FRESH_START=1 to start a new run")
        sys.exit(0)
    print(f"Resuming from {CHECKPOINT}: stage {ckpt['next_stage']}, "
          f"{'in progress' if ckpt['opt'] is not None else 'not started'}")


# =============================================================================
# Solver / compute site
# =============================================================================

SITE_NAME = "isambard_oliver" if _ON_ISAMBARD else "oliver_wsl"   # Salvus site to run simulations on
RANKS     = 1              # MPI ranks per simulation job (1 GPU; the mesh is only ~1.2k elements)

# Multi-GPU node: one Salvus site per GPU (isambard_oliver_gpu<g>, CUDA_VISIBLE_DEVICES = g, see
# ~/.config/SalvusFlow/1.0/salvus-flow-config.toml). The events are split into one group per GPU
# and the groups run concurrently, one simulation per GPU at a time.
N_GPUS     = int(os.environ.get('SLURM_GPUS_ON_NODE', '1')) if _ON_ISAMBARD else 1
SITE_NAMES = [f"{SITE_NAME}_gpu{g}" for g in range(N_GPUS)] if N_GPUS > 1 else [SITE_NAME]


# =============================================================================
# Voronoi grains and their orientations (as forward_2D_voronoi.py)
# =============================================================================

N_GRAIN       = 10         # number of grains (Voronoi seeds)
VORONOI_SEED  = 300        # seed of the grain geometry (Voronoi seed points and colours)
OUTPUT_PX     = 512        # square PNG size [px]
VORONOI_PNG   = VORONOI_DATA / f"voronoi_{N_GRAIN}_seed{VORONOI_SEED}.png"

THETA_MEAN_DEG = 45.0      # grain orientations ~ N(mean, std) [deg]; also the starting model
THETA_STD_DEG  = 15.0
THETA_SEED     = 0         # seed of the orientation draw (independent of the geometry)


# =============================================================================
# Source wavelet and time axis
# =============================================================================

# Stage centre frequency = alpha * CENTRAL_FREQUENCY; the mesh is built for alpha = 1
CENTRAL_FREQUENCY = 1e6                 # Ricker centre frequency at alpha = 1 [Hz]
end_time          = 6e-6                # simulation end time at alpha = 1 [s], qSV ~4.1 us across 10 mm
amplitude_ratio   = 1e10                # transducer force [N]


# =============================================================================
# Domain geometry, material, mesh resolution
# =============================================================================

x0, x1 = 0.0, 0.01                 # horizontal extent [m] (10 mm)
y0, y1 = 0.0, 0.01                 # vertical extent [m]   (10 mm)

matl    = Austenite()                         # hexagonal austenite (RHO, Cij)
VP_0deg = np.sqrt(matl.C33 / matl.RHO)        # P velocity along symmetry axis [m/s]
VS_0deg = np.sqrt(matl.C44 / matl.RHO)        # S velocity along symmetry axis [m/s]

STIFFNESS_PARAMETERS = {'C11', 'C12', 'C13', 'C22', 'C23', 'C33'}   # Salvus 2D Voigt names

elements_per_wavelength = 4                       # elements per minimum wavelength
model_order             = 4                       # spectral-element polynomial order
reference_frequency     = CENTRAL_FREQUENCY * 2   # max frequency resolved by the mesh [Hz]
# finite specimen: all 4 sides free -> no absorbing layer, the whole mesh is the physical domain


# =============================================================================
# Transducers (as forward_2D_voronoi.py, every listed transducer transmits in turn)
# =============================================================================

# 36 transducers, 9 per side at 1 ... 9 mm, ordered bottom, top, left, right; station code
# T01 ... T36 in that order. Each firing transducer transmits a force normal to its side (unit
# normal SIDE_NORMALS, pointing into the specimen); the 27 on the other 3 sides receive.
SIDES         = ['bottom', 'top', 'left', 'right']
FIRING_SIDES  = SIDES                   # sides whose transducers transmit
FIRING_INDICES = range(9)               # transducers (0-8 along the side) that transmit
_line = np.linspace(0.001, 0.009, 9)
_side_xy = {'bottom': np.column_stack([_line, np.full(9, y0)]), 'top':   np.column_stack([_line, np.full(9, y1)]),
            'left':   np.column_stack([np.full(9, x0), _line]), 'right': np.column_stack([np.full(9, x1), _line])}
SIDE_NORMALS = {'bottom': (0.0, 1.0), 'top': (0.0, -1.0), 'left': (1.0, 0.0), 'right': (-1.0, 0.0)}
trd_loc  = np.vstack([_side_xy[s] for s in SIDES])                         # all 36 transducers [m]
trd_side = np.repeat(SIDES, 9)
trd_code = [f"T{j + 1:02d}" for j in range(len(trd_loc))]


# =============================================================================
# Voronoi map helpers (voronoi_generator.ipynb, as forward_2D_voronoi.py)
# =============================================================================

def voronoi_finite_polygons_2d(vor, radius=None):
    """Close the infinite Voronoi regions at a far radius (voronoi_generator.ipynb)."""
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions, new_vertices = [], vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max()               # (ndarray.ptp is gone in numpy 2)
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        verts = vor.regions[region]
        if verts and all(v >= 0 for v in verts):
            new_regions.append(verts)
            continue
        new_region = [v for v in verts if v >= 0]
        for p2, v1, v2 in all_ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]; t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            new_region.append(len(new_vertices))
            new_vertices.append((vor.vertices[v2] + direction * radius).tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        ang = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_regions.append(np.array(new_region)[np.argsort(ang)].tolist())
    return new_regions, np.asarray(new_vertices)


def generate_voronoi_png(path, n_grain, seed, output_px):
    """
    Voronoi grain map of voronoi_generator.ipynb: n_grain seeds in the unit square, one random flat
    colour per grain, output_px x output_px PNG. Anti-aliasing off so every pixel carries exactly
    one grain colour (no blended edge colours for the colour segmentation).
    """
    rng = np.random.RandomState(seed)                            # = np.random.seed(seed) of the notebook
    points = rng.rand(n_grain, 2)
    regions, vertices = voronoi_finite_polygons_2d(Voronoi(points))
    colors = rng.rand(n_grain, 3)
    inch = 6.0
    fig = plt.figure(figsize=(inch, inch), dpi=output_px / inch)
    ax = fig.add_axes([0, 0, 1, 1])
    for i, region in enumerate(regions):
        ax.fill(*zip(*vertices[region]), facecolor=colors[i], edgecolor='none', antialiased=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    fig.savefig(path, dpi=output_px / inch)                      # explicit dpi: exact pixel size
    plt.close(fig)
    print(f"Generated Voronoi grain map {path} ({n_grain} grains, seed {seed}, {output_px} px)")


def grain_ids_of_points(indexer, xz):
    """Grain id of every point (x, z) [m] (any array shape (..., 2)), clipped to the domain."""
    xn = np.clip((xz[..., 0] - x0) / (x1 - x0), 0.0, 1.0)
    zn = np.clip((xz[..., 1] - y0) / (y1 - y0), 0.0, 1.0)
    col = np.rint(xn * (indexer.W - 1)).astype(int)
    row = np.rint((1.0 - zn) * (indexer.H - 1)).astype(int)    # image rows run top -> bottom
    gid = indexer.seg_labels[row, col]
    for idx in zip(*np.nonzero(gid < 0)):                       # unlabelled pixels: neighbour vote
        gid[idx] = indexer.point_to_grain_id(float(xn[idx]), float(zn[idx]))
    return gid


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


# =============================================================================
# Salvus helpers
# =============================================================================

def event_groups(events):
    """(site, events) pairs: the events dealt round-robin over SITE_NAMES (one group per GPU)."""
    return [(site, list(events[g::len(SITE_NAMES)])) for g, site in enumerate(SITE_NAMES)
            if len(events[g::len(SITE_NAMES)])]


def forward_simulation(simulation_name, events):
    """Launch a forward simulation (receiver data only) in project `p` and block until it finishes."""
    _n_events = 1 if isinstance(events, str) else len(events)
    print(f"  [forward_simulation] launching '{simulation_name}' for {_n_events} event(s) "
          f"on {len(SITE_NAMES)} site(s) ...")

    t_start = time.perf_counter()
    for site, evs in event_groups([events] if isinstance(events, str) else events):
        p.simulations.launch(
            ranks_per_job=RANKS,
            site_name=site,
            events=evs,
            simulation_configuration=simulation_name,
            delete_conflicting_previous_results=True,
        )

    # Wait for THIS simulation only; query(block=True) occasionally raises while it
    # polls a running job; keep waiting instead.
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
# Voronoi grain map and orientations
# =============================================================================

VORONOI_DATA.mkdir(parents=True, exist_ok=True)
if not VORONOI_PNG.exists():
    generate_voronoi_png(VORONOI_PNG, N_GRAIN, VORONOI_SEED, OUTPUT_PX)
indexer = VoronoiGrainIndexer(str(VORONOI_PNG), min_percent=0.1, dilate_iters=1)
indexer.process()
n_grains_found = len(indexer.kept_colors_sorted)
assert n_grains_found == N_GRAIN, f"expected {N_GRAIN} grains, segmentation found {n_grains_found}"

rng = np.random.default_rng(THETA_SEED)
theta_grain_deg = rng.normal(THETA_MEAN_DEG, THETA_STD_DEG, size=n_grains_found)   # one per grain id
for g, t in enumerate(theta_grain_deg):
    print(f"  grain {g} ({indexer.areas_df['Percentage (%)'][g]:5.1f} % area): theta = {t:6.2f} deg")


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
matl_mean = oriented_material(material_unoriented, THETA_MEAN_DEG)   # starting model, mesh sizing

mesh_res = sn.MeshResolution(
    reference_frequency=reference_frequency,
    elements_per_wavelength=elements_per_wavelength,
    model_order=model_order,
)

# Starting model: regular box mesh, homogeneous THETA_MEAN_DEG (no absorbing layer: all sides free)
mesh_init = lm.mesh_from_domain(
    domain=domain,
    model=sn.layered_meshing.MeshingProtocol(lm.LayeredModel([matl_mean]), ab=None),
    mesh_resolution=mesh_res,
)

# consistency of the closed-form rotation with Salvus' own rotation
_C_check = stiffness_from_theta(np.deg2rad(THETA_MEAN_DEG), matl)
for _k in STIFFNESS_PARAMETERS:
    assert np.isclose(_C_check[_k], getattr(matl_mean, _k).p, rtol=1e-9, atol=1e-3), _k

# True model: every mesh node takes the theta of the grain it lies in (grain boundaries resolved
# at the GLL-node spacing, not by element edges); shared nodes get one value -> theta continuous
_x, _y = mesh_init.points[:, 0], mesh_init.points[:, 1]
grain_node = grain_ids_of_points(indexer, mesh_init.points)                 # (n_nodes,)
_theta_true_nodal = np.deg2rad(theta_grain_deg[grain_node])                 # (n_nodes,) [rad]
mesh = mesh_init.copy()
_C_true = stiffness_from_theta(_theta_true_nodal[mesh.connectivity], matl)
for _k in STIFFNESS_PARAMETERS:
    mesh.element_nodal_fields[_k][:] = _C_true[_k]
print(f"Mesh: {mesh.nelem} elements, {mesh.npoint} nodes; nodes per grain: "
      f"{np.bincount(grain_node, minlength=n_grains_found).tolist()}")

# Time step: 90 % of the CFL limit. The fastest quasi-P speed is bounded by the largest
# eigenvalue of the 2D Voigt tensor, which a rotation leaves (nearly) unchanged -> valid for all theta
_f = mesh.element_nodal_fields
_C_voigt = np.stack([np.stack([_f['C11'], _f['C12'], _f['C13']], -1),
                     np.stack([_f['C12'], _f['C22'], _f['C23']], -1),
                     np.stack([_f['C13'], _f['C23'], _f['C33']], -1)], -2)
_v_max = np.sqrt(np.linalg.eigvalsh(_C_voigt)[..., -1] / _f['RHO']).max(axis=1)
_dt_cfl = mesh.compute_time_step(max_velocity=_v_max)[0]
time_step_in_seconds = np.floor(0.9 * _dt_cfl * 1e11) / 1e11
print(f"CFL time step {_dt_cfl * 1e9:.3f} ns, using {time_step_in_seconds * 1e9:.2f} ns")


# =============================================================================
# Events: one per firing transducer, received by the transducers on the other 3 sides
# =============================================================================

event_ls, tx_of_event, rx_of_event = [], {}, {}
for side in FIRING_SIDES:
    for i in FIRING_INDICES:
        j_tx = SIDES.index(side) * 9 + i                     # index into trd_loc
        rx = np.flatnonzero(trd_side != side)                 # 27 receiving transducers
        receivers = [
            sn.simple_config.receiver.cartesian.Point2D(
                x=trd_loc[j, 0], y=trd_loc[j, 1], station_code=trd_code[j], fields=["displacement"]
            )
            for j in rx
        ]
        nx, nz = SIDE_NORMALS[side]                           # force normal to the side
        event_name = f"{side}_{i + 1}"                        # e.g. 'top_5'
        ev = sn.Event(event_name=event_name, receivers=receivers,
                      sources=sn.simple_config.source.cartesian.VectorPoint2D(
                          x=trd_loc[j_tx, 0], y=trd_loc[j_tx, 1],
                          fx=nx * amplitude_ratio, fy=nz * amplitude_ratio))
        p.events.update(ev) if p.events.exists(event_name) else p.add_to_project(ev)   # exists on resume
        event_ls.append(event_name)
        tx_of_event[event_name], rx_of_event[event_name] = j_tx, rx
print(f"{len(event_ls)} event(s) registered: {event_ls}")
print(f"{N_GPUS} GPU(s), sites: {SITE_NAMES}")

tx_locations = trd_loc[[tx_of_event[ev] for ev in event_ls]]   # firing transducers [m]


# =============================================================================
# Figure style (project journal format, my_code/plot_style.py) and the
# simulation configuration: true orientation field with the transducers
# =============================================================================

apply_paper_style()

for _d in (FWI_IMAGE_DIR, FWI_DATA_DIR, PAPER_FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

fig, _ = plot_orientation_field(mesh, tx_locations, trd_loc, extent=(x0, x1, y0, y1),
                                transducers=True,
                                save_path=PAPER_FIG_DIR / "FWI_orientation_configuration")
plt.close(fig)


# =============================================================================
# FWI settings
# =============================================================================

# Multiscale continuation: centre frequency = alpha * CENTRAL_FREQUENCY, low -> high.
# alpha <= 1 keeps every stage within the mesh resolution (reference_frequency = 2 f_c).
alpha_ls           = [0.2, 0.4, 0.6, 0.8, 1.0]  # f_c = alpha * CENTRAL_FREQUENCY (0.2 ... 1 MHz)
# Simulation end time per stage: end_time is the alpha = 1 value, scaled with the period
end_time_ls        = [end_time / alpha for alpha in alpha_ls]   # 30, 15, 10, 7.5, 6 us
MAX_ITER_PER_STAGE = 10                        # max accepted L-BFGS iterations per stage

# Orientation bounds [rad], kept off 0 and 90 deg: there dC/dtheta of every normal-stress term
# (C11, C13, C33, C55) vanishes and only the weak shear couplings C15, C35 remain, so K_theta
# nearly vanishes and a node that reaches 0 or 90 deg cannot move back. The true model
# (26 - 65 deg) lies inside.
THETA_MIN, THETA_MAX = np.deg2rad(5.0), np.deg2rad(85.0)

# 2D smoothing operators (update smoothing and illumination) act on a regular grid of spacing
# SMOOTH_GRID_H: nodes -> grid by bilinear scattering (B^T), Gaussian filter, grid -> nodes (B).
SMOOTH_GRID_H = 0.05e-3                         # grid spacing [m] (~ the finest GLL node spacing)

# Gaussian smoothing of the UPDATE in x and z, built into the L-BFGS initial inverse Hessian.
# sigma = UPDATE_SMOOTH_WAVELENGTHS * shear wavelength of the stage (scaled with the resolution).
UPDATE_SMOOTH_WAVELENGTHS = 0.1                 # sigma / lambda_S

# Illumination preconditioner: K_theta is far stronger near the transducers than inside, so the
# L-BFGS direction is divided by a map E(x, z) of |K_theta| (mass-weighted, Gaussian-smoothed),
# fixed per stage from the stage's first gradient. The water level keeps poorly illuminated
# regions from being over-amplified.
PRECOND_ILLUMINATION = True
PRECOND_WATER_LEVEL  = 0.05                     # floor of E, fraction of max E
PRECOND_SMOOTH       = 0.5e-3                   # Gaussian smoothing length of E [m]

# Trust region in theta [rad] (L-inf ball)
DELTA_TR_INIT = np.deg2rad(5.0)
DELTA_TR_MAX  = np.deg2rad(20.0)

DIAG_EVENT    = "top_5" if "top_5" in event_ls else event_ls[len(event_ls) // 2]   # shown in the diagnostics
_j_diag = SIDES.index('bottom') * 9 + 4                  # bottom-middle transducer (single traces)
if _j_diag not in rx_of_event[DIAG_EVENT]:
    _j_diag = rx_of_event[DIAG_EVENT][len(rx_of_event[DIAG_EVENT]) // 2]
DIAG_STATION  = trd_code[_j_diag]
DIAG_RECEIVER = f"XX.{DIAG_STATION}."                     # its receiver_id in the waveform xarray
print(f"DIAG_EVENT = {DIAG_EVENT!r}, DIAG_RECEIVER = {DIAG_RECEIVER!r}")


# =============================================================================
# Geometry-only setup (frequency independent)
# =============================================================================

base_mesh = copy.deepcopy(mesh_init)
_theta_init_nodal = elemental_nodal_to_nodal_field(theta_from_mesh(base_mesh), base_mesh.connectivity)
THETA_HOMO = float(_theta_init_nodal.mean())
print(f"THETA_HOMO (from mesh_init) = {np.rad2deg(THETA_HOMO):.2f} deg")
np.save(FWI_DATA_DIR / "node_xy.npy", np.column_stack([_x, _y]))
np.savez_compressed(FWI_DATA_DIR / "voronoi_true_model.npz", grain_node=grain_node,
                    theta_true_nodal=_theta_true_nodal, theta_grain_deg=theta_grain_deg,
                    n_grain=N_GRAIN, voronoi_seed=VORONOI_SEED, theta_seed=THETA_SEED)

# Bilinear node <-> grid operator B (n_nodes x n_grid), grid values as (nz, nx) images
_gx = np.linspace(x0, x1, int(round((x1 - x0) / SMOOTH_GRID_H)) + 1)
_gz = np.linspace(y0, y1, int(round((y1 - y0) / SMOOTH_GRID_H)) + 1)
_hx, _hz = _gx[1] - _gx[0], _gz[1] - _gz[0]
_ix = np.clip(((_x - x0) / _hx).astype(int), 0, len(_gx) - 2)
_iz = np.clip(((_y - y0) / _hz).astype(int), 0, len(_gz) - 2)
_fx, _fz = (_x - _gx[_ix]) / _hx, (_y - _gz[_iz]) / _hz
_B = sps.csr_matrix(
    (np.stack([(1 - _fx) * (1 - _fz), _fx * (1 - _fz), (1 - _fx) * _fz, _fx * _fz], 1).ravel(),
     (np.repeat(np.arange(len(_x)), 4),
      np.stack([_iz * len(_gx) + _ix, _iz * len(_gx) + _ix + 1,
                (_iz + 1) * len(_gx) + _ix, (_iz + 1) * len(_gx) + _ix + 1], 1).ravel())),
    shape=(len(_x), len(_gx) * len(_gz)))
_GRID_SHAPE = (len(_gz), len(_gx))


def _grid_smooth(img, sigma):
    """Gaussian filter of a grid image, sigma [m]; zero outside (a symmetric operator)."""
    return gaussian_filter(img, sigma=sigma / SMOOTH_GRID_H, mode='constant')


# Full list of acquired (source, receiver) pairs
sr_pairs = [(tuple(trd_loc[tx_of_event[ev]]), tuple(trd_loc[j])) for ev in event_ls for j in rx_of_event[ev]]

_fwi_iter   = [0]
_fwi_ls     = [0]
_fwi_call   = [0]   # unique simulation names: Salvus caches results per configuration name
_diag_cache = {}    # filled by objective() for DIAG_EVENT, read by the optimizer for plotting
_P_nodes    = [None]   # per-node illumination preconditioner 1/E(x, z), set once per stage by objective()
_M_nodal    = [None]   # lumped nodal mass (area per node, no density), set by objective()


def save_checkpoint(next_stage, opt_state=None, finished=False):
    """
    Save the FWI state atomically (temporary file + rename, so a run killed while writing
    leaves the previous checkpoint intact).

    next_stage : frequency stage to run on restart (the one in progress if opt_state is given)
    opt_state  : L-BFGS + trust-region state of that stage (run_lbfgs_trust_region), None if
                 the stage has not started
    Also saved: the histories of the finished stages, the warm start, the observed-data names
    of every stage and the objective() call counter.
    """
    state = dict(next_stage=next_stage, opt=opt_state, finished=finished, fwi_call=_fwi_call[0],
                 M_nodal=_M_nodal[0], obs_names=obs_names, theta_warm_start=theta_warm_start,
                 all_misfit_history=all_misfit_history, all_model_history=all_model_history,
                 all_synthetic_history=all_synthetic_history, all_adjoint_history=all_adjoint_history,
                 all_observed_history=all_observed_history, stage_boundaries=stage_boundaries,
                 stage_summary=stage_summary)
    tmp = CHECKPOINT.with_suffix('.tmp')
    with open(tmp, 'wb') as fh:
        pickle.dump(state, fh)
    os.replace(tmp, CHECKPOINT)


def build_mesh_from_model(m_flat):
    """
    Decode theta (every mesh node) -> stiffness, attached to a copy of the mesh.

    Returns
    -------
    mesh_it     : mesh with the rotated C11 ... C33 fields
    theta_nodal : (n_nodes,) theta field [rad], clipped to the bounds
    """
    theta_nodal = np.clip(m_flat, THETA_MIN, THETA_MAX)
    mesh_it = copy.deepcopy(base_mesh)
    C = stiffness_from_theta(theta_nodal[base_mesh.connectivity], matl)
    for k in STIFFNESS_PARAMETERS:
        mesh_it.attach_field(k, C[k])
    return mesh_it, theta_nodal


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

def _field_axes(fig, ax, f, title, cmap, vmin, vmax, label):
    """tripcolor of a nodal field on the specimen, colour bar of the same height."""
    tp = ax.tripcolor(_x * mm, _y * mm, f, cmap=cmap, shading='gouraud',
                      vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$z$ (mm)')
    ax.set_title(title, loc='left')
    ax.set_xlim(x0 * mm, x1 * mm); ax.set_ylim(y0 * mm, y1 * mm)
    ax.set_aspect('equal')
    add_colorbar(fig, ax, tp, label)


def plot_fwi_diagnostics(it, u_obs, u_syn, f_adj, K_theta, save=True):
    """
    Waveforms + adjoint source (DIAG_RECEIVER), receiver gathers per receiving side and
    K_theta (full, and masked to the Fresnel zone) for iteration `it`, event DIAG_EVENT.
    """
    t_us = u_obs.time.values * 1e6

    # ── Figure 1: observed vs synthetic and adjoint source, X and Y ────────
    fig1, axs = plt.subplots(2, 2, figsize=(7.0, 4.0), layout='constrained', sharex=True)
    for col, comp in enumerate(['X', 'Y']):
        name = 'xz'[col]
        axs[0, col].plot(t_us, u_syn.sel(receiver_id=DIAG_RECEIVER, component=comp).values.squeeze(),
                         color=C_SYN, lw=0.9, label=rf'$u^{{({it})}}_{name}$')
        axs[0, col].plot(t_us, u_obs.sel(receiver_id=DIAG_RECEIVER, component=comp).values.squeeze(),
                         color=C_OBS, lw=0.9, ls='--', label=rf'$u_{{{name},\mathrm{{obs}}}}$')
        axs[0, col].legend(frameon=False)
        axs[0, col].set_ylabel(f'$u_{name}$ (m)')
        axs[0, col].set_title(f'({"ab"[col]}) Observed vs. synthetic, $u_{name}$, {DIAG_STATION} '
                              f'(iteration {it})', loc='left')
        axs[1, col].plot(t_us, f_adj[col], color='0.3', lw=0.9)
        axs[1, col].axhline(0, color='k', lw=0.4, ls='--')
        axs[1, col].set_xlabel(r'Time ($\mu$s)')
        axs[1, col].set_ylabel(rf'$f^\dagger_{name}$')
        axs[1, col].set_title(rf'({"cd"[col]}) Adjoint source $f^\dagger = d - u$, {name}', loc='left')
        for ax in axs[:, col]:
            ax.grid(True, **GRID_KW)
    if save:
        plt.savefig(FWI_IMAGE_DIR / f'iter_{it:03d}_waveforms_adjoint_source_alpha_{alpha}.png')
    plt.close(fig1)

    # ── Figure 1b: gathers, one panel per receiving side, displacement normal to the side,
    #    traces offset by the position along the side ───────────────────────
    rx = rx_of_event[DIAG_EVENT]
    rx_sides = [s for s in SIDES if s in set(trd_side[rx])]
    fig1b, axs = plt.subplots(1, len(rx_sides), figsize=(7.0, 4.0), layout='constrained', sharey=True)
    axs = np.atleast_1d(axs)
    scale = 0.45 / (np.abs(u_obs.values).max() or 1.0)       # common scale: 0.45 mm per max |u_obs|
    for k, (ax, side) in enumerate(zip(axs, rx_sides)):
        comp, name = ('Y', 'z') if side in ('bottom', 'top') else ('X', 'x')
        pos = 0 if side in ('bottom', 'top') else 1          # coordinate along the side
        for n, j in enumerate(j for j in rx if trd_side[j] == side):
            rid = f"XX.{trd_code[j]}."
            ax.plot(trd_loc[j, pos] * mm + scale * u_obs.sel(receiver_id=rid, component=comp).values.squeeze(),
                    t_us, color=C_OBS, lw=0.7, ls='--', label='Observed' if n == 0 else None)
            ax.plot(trd_loc[j, pos] * mm + scale * u_syn.sel(receiver_id=rid, component=comp).values.squeeze(),
                    t_us, color=C_SYN, lw=0.7, label=f'Iteration {it}' if n == 0 else None)
        ax.set_xlabel(f'Transducer ${"xz"[pos]}$ (mm)')
        ax.set_title(f'({"abcd"[k]}) {side}, $u_{name}$', loc='left')
        ax.grid(True, **GRID_KW)
    axs[0].set_ylabel(r'Time ($\mu$s)')
    axs[0].invert_yaxis()
    axs[0].legend(frameon=False, loc='lower left')
    if save:
        plt.savefig(FWI_IMAGE_DIR / f'iter_{it:03d}_receiver_gather_alpha_{alpha}.png')
    plt.close(fig1b)

    # ── Figure 2: K_theta, full and masked to the Fresnel zone ─────────────
    fig2, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), layout='constrained')
    for k, (ax, K, name) in enumerate(zip(axs, [K_theta, np.where(_ellipse, K_theta, 0.0)],
                                          ['', ', Fresnel zone'])):
        vmax = np.percentile(np.abs(K), 99.0) or 1.0
        _field_axes(fig2, ax, K, rf'({"ab"[k]}) $K_\theta${name} (iteration {it})',
                    'RdBu', -vmax, vmax, r'$K_\theta$')
    axs[0].scatter(tx_locations[:, 0] * mm, tx_locations[:, 1] * mm, marker='s', color='red',
                   edgecolors='k', linewidths=0.4, s=12, zorder=5, clip_on=False)
    if save:
        plt.savefig(FWI_IMAGE_DIR / f'iter_{it:03d}_kernel_theta_alpha_{alpha}.png')
    plt.close(fig2)


def plot_theta_model(theta_nodal, it, ls):
    """Current theta field vs the true field of one objective() call."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), layout='constrained')
    _field_axes(fig, axes[0], np.rad2deg(theta_nodal),
                rf'(a) Current $\theta$ (iteration {it}, ls {ls})', 'viridis', 0.0, 90.0, r'$\theta$ (deg)')
    _field_axes(fig, axes[1], np.rad2deg(_theta_true_nodal),
                r'(b) True $\theta$', 'viridis', 0.0, 90.0, r'$\theta$ (deg)')
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
    Multi-source FWI objective — model parameter is theta [rad] at every mesh node.

    Salvus runs forward + adjoint for all events and returns K_C summed over
    events; K_theta = K_C :: dC/dtheta is evaluated at the current theta(x, z).

    Returns
    -------
    misfit : float                       total L2 misfit, summed over sources
    grad   : np.ndarray, (n_nodes,)      d misfit / d theta_i, zeroed outside the Fresnel zone
    """
    it, ls = _fwi_iter[0], _fwi_ls[0]
    _fwi_ls[0] += 1
    _fwi_call[0] += 1
    sim_name = f'fwi_theta_a{stage_idx}_{it:03d}_ls{ls}_c{_fwi_call[0]:03d}_r{RUN_TAG}'
    print(f"\n=== objective() call: alpha={alpha} iter={it} ls={ls} sim_name={sim_name!r} ===")

    # ── 1. Decode theta, build the mesh ─────────────────────────────────────
    mesh_it, theta_nodal = build_mesh_from_model(m_flat)
    print(f'  [obj] theta: mean={np.rad2deg(theta_nodal.mean()):.2f} deg  '
          f'range=[{np.rad2deg(theta_nodal.min()):.2f}, {np.rad2deg(theta_nodal.max()):.2f}] deg')
    plot_theta_model(theta_nodal, it, ls)

    p.add_to_project(
        sn.UnstructuredMeshSimulationConfiguration(
            unstructured_mesh=mesh_it,
            name=sim_name,
            event_configuration=event_config_fwd,
        ),
        overwrite=True,
    )

    # ── 2. Forward + adjoint (Salvus built-in), all sources, one event group per GPU ─
    # Every poll asks each group: the first call launches its forward runs, later calls its
    # adjoint runs; None until all groups are done
    def _all_groups(action):
        results = [action(site, evs) for site, evs in event_groups(event_ls)]
        return None if any(r is None for r in results) else results

    t_start = time.perf_counter()
    _wait_for_result(lambda: _all_groups(lambda site, evs: p.actions.inversion.compute_gradients(
        simulation_configuration=sim_name,
        misfit_configuration=misfit_name,
        events=evs,
        site_name=site,
        ranks_per_job=RANKS,
        gradient_parameters=STIFFNESS_PARAMETERS,
    )), "compute_gradients")
    misfits = {}
    for group_misfits in _wait_for_result(lambda: _all_groups(
            lambda site, evs: p.actions.inversion.compute_misfits(
                simulation_configuration=sim_name,
                misfit_configuration=misfit_name,
                events=evs,
                site_name=site,
                ranks_per_job=RANKS,
            )), "compute_misfits"):
        misfits.update(group_misfits)
    assert len(misfits) == len(event_ls), (len(misfits), len(event_ls))
    print(f'  [obj] forward + adjoint done in {time.perf_counter() - t_start:.1f} s')
    misfit_total = float(sum(misfits.values()))

    gradient_C = p.actions.inversion.sum_gradients(
        simulation_configuration=sim_name,
        misfit_configuration=misfit_name,
        events=event_ls,
        gradient_parameters=STIFFNESS_PARAMETERS,
    )

    # ── 3. K_theta = K_C :: dC/dtheta at the current theta(x, z) ───────────
    K_theta = elemental_nodal_to_nodal_field(orientation_kernel(gradient_C, mesh_it, matl),
                                             mesh_it.connectivity)

    # ── Diagnostics for DIAG_EVENT: synthetic gather, DIAG_RECEIVER adjoint source
    u_syn = extract_data(sim_name, DIAG_EVENT, receiver_field='displacement')
    event_misfit = sn.EventMisfit(
        observed_event=p.waveforms.get(data_name=obs_simulation, events=DIAG_EVENT)[0],
        synthetic_event=p.waveforms.get(data_name=sim_name, events=DIAG_EVENT)[0],
        misfit_function="L2",
        receiver_field="displacement",
    )
    _diag_name = next(r for r in event_misfit.receiver_name_list if r.split('.')[1] == DIAG_STATION)
    adj = event_misfit.get_misfit_and_adjoint_source_for_receiver(_diag_name)
    # Salvus stores -(u - d) * dt; plot the continuous adjoint force d - u
    f_adj = adj['adjoint_sources'].values.astype(float) / time_step_in_seconds
    _diag_cache.update(u_syn=u_syn, f_adj=f_adj, K_theta=K_theta)

    # ── 4. Discrete gradient, Fresnel-zone mask ─────────────────────────────
    # K_theta is a pointwise kernel density [misfit / rad / m^2]; the discrete gradient is
    # d chi / d theta_i = K_theta(x_i) * M_i with the lumped nodal mass M_i (FemMassMatrix:
    # assembled quadrature weight * |J|, no density). Without it pred = -g.dm is ~1/M too
    # large, rho ~ 0 and every step is rejected. No x-averaging: every node is free.
    M_nodal = np.zeros(base_mesh.npoint)
    M_nodal[base_mesh.connectivity] = gradient_C.element_nodal_fields['FemMassMatrix']
    _M_nodal[0] = M_nodal
    grad = np.where(_ellipse, K_theta * M_nodal, 0.0)

    # ── 5. Illumination map E(x, z) -> preconditioner 1/E (first call of a stage)
    # |K_theta| (not K_theta) is averaged: the kernel oscillates and would cancel.
    # E = smoothed integral of |K_theta| / smoothed area, i.e. a local mean of |K_theta|.
    # Kept fixed for the whole stage so L-BFGS sees one consistent metric.
    if PRECOND_ILLUMINATION and _P_nodes[0] is None:
        num = _grid_smooth((_B.T @ np.where(_ellipse, np.abs(K_theta) * M_nodal, 0.0)).reshape(_GRID_SHAPE),
                           PRECOND_SMOOTH)
        den = _grid_smooth((_B.T @ np.where(_ellipse, M_nodal, 0.0)).reshape(_GRID_SHAPE),
                           PRECOND_SMOOTH)
        E = _B @ (num / np.maximum(den, den.max() * 1e-6)).ravel()
        E = np.maximum(E, PRECOND_WATER_LEVEL * E.max())
        _P_nodes[0] = E.max() / E                         # 1 at the best-lit node
        np.save(FWI_DATA_DIR / f"illumination_map_alpha_{alpha}.npy", E)
        print(f"  [precond] illumination 1/E(x, z): range [{_P_nodes[0].min():.2f}, {_P_nodes[0].max():.2f}]")

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
                           theta_init=None, resume=None):
    """
    L-BFGS + trust region FWI optimizer for theta [rad] (see the acoustic script).

    Steps and radii are in radians (L-inf ball). The update is smoothed in x and z
    through the L-BFGS initial inverse Hessian H0 = P^1/2 B D^-1/2 G D^-1/2 B^T P^1/2
    (P = illumination preconditioner 1/E(x, z), B = bilinear grid -> node interpolation,
    G = Gaussian filter on the grid, sigma = UPDATE_SMOOTH_WAVELENGTHS * lambda_S of the
    stage, D = G 1 normalises the edges symmetrically). `max_iter` counts accepted steps;
    the gradient is cached across rejected steps.

    `theta_init` (nodal theta, (n_nodes,)) warm-starts a frequency stage;
    defaults to the homogeneous starting theta. `resume` (the opt_state of a checkpoint)
    continues an interrupted stage after its last completed attempt. The state is saved with
    save_checkpoint() after the baseline and after every attempt.

    Returns
    -------
    misfit_history    : list[float],              length n_accepted + 1
    model_history     : (n_accepted + 1, n_nodes)          theta [rad]
    synthetic_history : (n_accepted + 1, 2 n_rx, n_time)  DIAG_EVENT synthetic gather
                        (traces ordered as the waveform xarray: receiver, then X / Y)
    adjoint_history   : (n_accepted + 1, 2, n_time)       DIAG_RECEIVER adjoint source (X, Y)
    """
    print(f"Starting L-BFGS trust-region optimization: "
          f"max_iter={max_iter}, max_tries={max_tries}, M={M}")

    def _diag_plots(it):
        plot_fwi_diagnostics(it=it, u_obs=u_obs_dict[DIAG_EVENT], u_syn=_diag_cache['u_syn'],
                             f_adj=_diag_cache['f_adj'], K_theta=_diag_cache['K_theta'])

    delta_tr = delta_tr_init
    m = (np.full(base_mesh.npoint, THETA_HOMO) if theta_init is None
         else np.clip(theta_init, THETA_MIN, THETA_MAX))
    s_list, y_list, rho_list = [], [], []
    prev_grad = None
    prev_s    = None

    _fwi_iter[0] = 0
    _fwi_ls[0]   = 0
    _P_nodes[0]  = None   # recomputed from this stage's first gradient

    # Gaussian smoothing of the update for this stage, sigma in metres. g carries the nodal
    # mass (g = K_theta * M, the quadrature weight), so B^T g is the integral of K_theta per
    # grid cell, independent of the uneven GLL node density; G is symmetric (zero padding) and
    # D^-1/2 G D^-1/2 keeps it symmetric while undoing the edge loss, so H0 stays SPD and d
    # stays a descent direction.
    sigma_upd = UPDATE_SMOOTH_WAVELENGTHS * VS_0deg / f_c
    _D_isqrt  = 1.0 / np.sqrt(_grid_smooth(np.ones(_GRID_SHAPE), sigma_upd))
    print(f"  update smoothing: sigma = {sigma_upd * mm:.3f} mm "
          f"({sigma_upd / SMOOTH_GRID_H:.1f} grid cells)")

    def _apply_H0(q):
        """Initial inverse Hessian: P^1/2 B D^-1/2 G D^-1/2 B^T P^1/2 q."""
        sP = np.sqrt(_P_nodes[0]) if _P_nodes[0] is not None else np.ones_like(q)
        v = (_B.T @ (sP * q)).reshape(_GRID_SHAPE) * _D_isqrt
        return sP * (_B @ (_grid_smooth(v, sigma_upd) * _D_isqrt).ravel())

    def _opt_state():
        """Everything needed to continue this stage after the current attempt."""
        return dict(m=m, k=k, tries=tries, delta_tr=delta_tr, s_list=s_list, y_list=y_list,
                    rho_list=rho_list, prev_grad=prev_grad, prev_s=prev_s,
                    cached_misfit=cached_misfit, cached_grad=cached_grad, P_nodes=_P_nodes[0],
                    misfit_history=misfit_history, model_history=model_history,
                    synthetic_history=synthetic_history, adjoint_history=adjoint_history)

    if resume is None:
        # Starting model = iteration-0 baseline (seeds all histories together)
        print("  evaluating objective() for the starting model (iteration 0 baseline) ...")
        misfit0, grad0 = objective(m)

        misfit_history    = [float(misfit0)]
        model_history     = [np.clip(m, THETA_MIN, THETA_MAX)]
        synthetic_history = [np.array(_diag_cache['u_syn'])]
        adjoint_history   = [np.array(_diag_cache['f_adj'])]
        cached_misfit, cached_grad = misfit0, grad0
        _diag_plots(0)

        k     = 0   # accepted iteration counter
        tries = 0   # total attempt counter
        save_checkpoint(stage_idx, _opt_state())
    else:
        m, k, tries, delta_tr = resume['m'], resume['k'], resume['tries'], resume['delta_tr']
        s_list, y_list, rho_list = resume['s_list'], resume['y_list'], resume['rho_list']
        prev_grad, prev_s = resume['prev_grad'], resume['prev_s']
        cached_misfit, cached_grad = resume['cached_misfit'], resume['cached_grad']
        _P_nodes[0] = resume['P_nodes']
        misfit_history, model_history = resume['misfit_history'], resume['model_history']
        synthetic_history, adjoint_history = resume['synthetic_history'], resume['adjoint_history']
        print(f"  resumed: {k} accepted / {tries} attempt(s), misfit {cached_misfit:.6e}, "
              f"delta_tr {np.rad2deg(delta_tr):.3f} deg")

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

        # ── L-BFGS two-loop recursion, H0 = gamma P^1/2 S P^1/2 (illumination
        #    1/E(x, z) and Gaussian update smoothing S) ─────────────────────────
        q, alphas = grad.copy(), []
        for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a = rho * np.dot(s, q); q -= a * y; alphas.append(a)
        gamma_h = (np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], _apply_H0(y_list[-1]))
                   if s_list else 1.0)
        r = gamma_h * _apply_H0(q)
        for s, y, rho, a in zip(s_list, y_list, rho_list, reversed(alphas)):
            r += s * (a - rho * np.dot(y, r))
        d = -r

        # ── Active bounds: nodes on a bound whose direction points outward
        #    would be clipped anyway; drop them so they do not set d_max and
        #    shrink the step of the rest of the model ──────────────────────
        _tol = 1e-9
        _active = ((m <= THETA_MIN + _tol) & (d < 0)) | ((m >= THETA_MAX - _tol) & (d > 0))
        if _active.any():
            print(f"  [bounds] {_active.sum()} node(s) on a bound with an outward direction: frozen this step")
            d = np.where(_active, 0.0, d)

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
            prev_grad     = grad.copy()
            prev_s        = m_new - m
            m             = m_new
            cached_misfit = misfit_new
            cached_grad   = grad_new
            misfit_history.append(misfit_new)
            model_history.append(m.copy())
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

        rms_true = np.rad2deg(np.sqrt(np.mean(((m - _theta_true_nodal + np.pi / 2) % np.pi - np.pi / 2) ** 2)))
        print(f"Iteration {k}: n_events={len(event_ls)}  theta RMS error={rms_true:.2f} deg  "
              f"chi={misfit:.6e}  ||g||={np.linalg.norm(grad):.3e}")
        print(f"  pred={predicted_red:.3e}  ared={actual_red:.3e}  rho={rho_k:.3f}  "
              f"||dm||_inf={np.rad2deg(step_norm):.3f} deg  "
              f"delta_tr={np.rad2deg(delta_tr):.3f} deg  ({status})")
        save_checkpoint(stage_idx, _opt_state())

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
obs_names             = {}   # stage -> (observed-data simulation, misfit configuration)

theta_warm_start = None      # None -> first stage starts from the homogeneous theta
first_stage, resume_opt = 0, None

if ckpt is not None:         # resume: finished stages and the state of the interrupted one
    all_misfit_history    = ckpt['all_misfit_history']
    all_model_history     = ckpt['all_model_history']
    all_synthetic_history = ckpt['all_synthetic_history']
    all_adjoint_history   = ckpt['all_adjoint_history']
    all_observed_history  = ckpt['all_observed_history']
    stage_boundaries      = ckpt['stage_boundaries']
    stage_summary         = ckpt['stage_summary']
    obs_names             = ckpt['obs_names']
    theta_warm_start      = ckpt['theta_warm_start']
    _fwi_call[0]          = ckpt['fwi_call']
    _M_nodal[0]           = ckpt['M_nodal']
    first_stage, resume_opt = ckpt['next_stage'], ckpt['opt']

for stage_idx, (alpha, end_time_stage) in enumerate(zip(alpha_ls, end_time_ls)):
    if stage_idx < first_stage:
        print(f"Stage {stage_idx} (alpha={alpha}) finished in a previous run, skipped")
        continue
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

    # Observed data = true (Voronoi) model at this stage's frequency. Computed once per stage:
    # a resumed stage reuses it (names in the checkpoint); if a previous run was interrupted
    # while computing it, it is recomputed under new names
    if stage_idx in obs_names:
        obs_simulation, misfit_name = obs_names[stage_idx]
        print(f"Stage {stage_idx} observed data reused from '{obs_simulation}'")
    else:
        obs_simulation = f'true_model_a{stage_idx}'
        misfit_name    = f'L2_theta_a{stage_idx}'
        if p.entities.has('simulation_configuration', obs_simulation):
            obs_simulation += f'_r{RUN_TAG}'
            misfit_name    += f'_r{RUN_TAG}'
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
        obs_names[stage_idx] = (obs_simulation, misfit_name)
        save_checkpoint(stage_idx, None)

    u_obs_dict = {ev: extract_data(obs_simulation, ev, receiver_field='displacement')
                  for ev in event_ls}

    # Fresnel-zone mask: margin = half a P wavelength at this stage's frequency
    fresnel_margin = 0.5 * VP_0deg / f_c
    _ellipse = _ellipse_mask(_x, _y, sr_pairs, fresnel_margin)
    print(f"Stage {stage_idx} ellipse mask: {_ellipse.sum()} / {len(_x)} "
          f"nodes inside the Fresnel zone (margin {fresnel_margin * mm:.2f} mm)")

    (misfit_history_stage, model_history_stage,
     synthetic_history_stage, adjoint_history_stage) = run_lbfgs_trust_region(
        max_iter=MAX_ITER_PER_STAGE, theta_init=theta_warm_start,
        resume=resume_opt if stage_idx == first_stage else None,
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
    save_checkpoint(stage_idx + 1, None)

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
np.save(FWI_DATA_DIR / "node_area.npy", _M_nodal[0])
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


# ── Model error vs the true model (RMS / relative theta error, maps, grains) ──
save_grain_model_error_figures(
    _all_model_history, np.column_stack([_x, _y]), _M_nodal[0], grain_node,
    np.rad2deg(_theta_true_nodal), alpha_ls, np.diff(stage_boundaries), CENTRAL_FREQUENCY,
    (x0, x1, y0, y1), PAPER_FIG_DIR,
)
save_checkpoint(len(alpha_ls), None, finished=True)   # a resubmitted job exits straight away
