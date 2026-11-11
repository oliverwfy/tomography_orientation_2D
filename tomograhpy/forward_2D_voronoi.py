"""
2D elastic forward simulation through a Voronoi polycrystal of austenite grains.

Workflow
--------
1. Voronoi grain map: the PNG of voronoi_generator.ipynb (N_GRAIN random seeds in the unit square,
   one flat colour per grain), drawn without anti-aliasing so every pixel has exactly one grain
   colour. Generated if it does not exist yet.
2. Grain indexing: VoronoiGrainIndexer (my_code/utilities.py) segments the PNG by colour and returns
   the grain id of any normalised point (x, z) in [0, 1]^2 (ids sorted by area, 0 = largest).
3. Grain orientations: theta_g ~ N(THETA_MEAN_DEG, THETA_STD_DEG) = N(45, 15) deg, one per grain
   (ClockwiseAngle convention: symmetry axis clockwise from +z), fixed random seed.
4. Mesh: regular spectral-element box mesh (no conforming interfaces); every GLL node takes the
   rotated stiffness of the grain it lies in, so grain boundaries are resolved at the GLL-node
   spacing (~0.05-0.13 mm), not by element edges.
5. Acquisition: 36 transducers on the 4 boundaries (a finite specimen: all 4 sides are free
   surfaces, no absorbing layer), 9 per side at 1 ... 9 mm. One event: transducer FIRING_SIDE /
   FIRING_INDEX transmits (force normal to its side), the 27 transducers on the other 3 sides
   receive (the 8 others on the transmitting side are inactive).
6. Forward simulation of that one event (displacement at the receiving transducers), traces saved,
   figures in the project journal format.
7. Mesh convergence (RUN_CONVERGENCE): the same event on meshes built for CONVERGENCE_FACTORS x the
   centre frequency (coarse -> fine); relative L2 error of the received traces against the finest
   mesh and a trace overlay (paper_figures/voronoi_mesh_convergence).

Run from the tomograhpy folder:
    python forward_2D_voronoi.py

Units: SI throughout (m, s, Pa, kg/m^3); plots use mm.
"""

import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file-only backend: figures are saved, never shown
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

import salvus.mesh
import salvus.mesh.layered_meshing as lm
import salvus.namespace as sn
from salvus.material._details import material as md

from my_code.utilities import VoronoiGrainIndexer
from my_code.plot_style import (apply_paper_style, save_paper_figure, plot_orientation_field,
                                DOUBLE_COL, MM, C_SYN, GRID_KW)
from material_ela_constants.Elastic_Material import Austenite


# =============================================================================
# 1. Paths and compute site
# =============================================================================

_ON_ISAMBARD = os.environ.get('SCRATCH', '').startswith('/scratch')
if _ON_ISAMBARD:
    _BASE           = os.environ['SCRATCH'] + '/Salvus_project/elastic_model/anisotropic'
    PROJECT_DIR_WIN = _BASE + '/Project'
    DATA_DIR_WIN    = _BASE + '/data'
    IMAGE_DIR_WIN   = _BASE + '/image'
else:
    PROJECT_DIR_WIN = '/mnt/f/Salvus_project/elastic_model/anisotropic/Project'
    DATA_DIR_WIN    = '/mnt/f/Salvus_project/elastic_model/anisotropic/data'
    IMAGE_DIR_WIN   = '/mnt/f/Salvus_project/elastic_model/anisotropic/image'

PROJECT_NAME  = "forward_voronoi"
VORONOI_DATA  = Path(DATA_DIR_WIN) / "voronoi"
VORONOI_IMAGE = Path(IMAGE_DIR_WIN) / "voronoi"
PAPER_FIG_DIR = VORONOI_IMAGE / "paper_figures"
for _d in (VORONOI_DATA, PAPER_FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SITE_NAME = "isambard_oliver" if _ON_ISAMBARD else "oliver_wsl"
RANKS     = 8


# =============================================================================
# 2. Voronoi grains and their orientations
# =============================================================================

N_GRAIN       = 10         # number of grains (Voronoi seeds)
VORONOI_SEED  = 300        # seed of the grain geometry (Voronoi seed points and colours)
OUTPUT_PX     = 512        # square PNG size [px]
VORONOI_PNG   = VORONOI_DATA / f"voronoi_{N_GRAIN}_seed{VORONOI_SEED}.png"

THETA_MEAN_DEG = 45.0      # grain orientations ~ N(mean, std) [deg]
THETA_STD_DEG  = 15.0
THETA_SEED     = 0         # seed of the orientation draw (independent of the geometry)


# =============================================================================
# 3. Wavelet, domain, material, mesh (as FWI_orientation_multilayer.py at 1 MHz)
# =============================================================================

CENTRAL_FREQUENCY = 1e6                                        # Ricker centre frequency [Hz]
start_time        = -1.5 / CENTRAL_FREQUENCY                   # Ricker peaks at t = 0: start early
end_time          = 6e-6                                       # [s], qSV ~4.1 us across 10 mm
amplitude_ratio   = 1e10                                       # transducer force [N]

x0, x1 = 0.0, 0.01                                             # physical domain [m]
y0, y1 = 0.0, 0.01

matl    = Austenite()
VP_0deg = np.sqrt(matl.C33 / matl.RHO)
STIFFNESS_PARAMETERS = ['C11', 'C12', 'C13', 'C22', 'C23', 'C33']   # Salvus 2D Voigt names

elements_per_wavelength = 4
model_order             = 4
reference_frequency     = CENTRAL_FREQUENCY * 2
number_of_wavelengths   = 3
free_surfaces           = ['x0', 'x1', 'y0', 'y1']   # finite specimen: transducers on every side

# Mesh convergence: the same event on meshes built for reference_frequency = factor * CENTRAL_FREQUENCY
# (coarse -> fine, elements_per_wavelength and model_order fixed); traces compared with the finest
# mesh. The main mesh (and the FWI) uses factor 2.
RUN_CONVERGENCE     = True
CONVERGENCE_FACTORS = [1.0, 1.5, 2.0, 3.0, 4.0]

# 36 transducers, 9 per side at 1 ... 9 mm, ordered bottom, top, left, right. One event: the transducer
# FIRING_INDEX (0-8 along the side) on FIRING_SIDE transmits a force normal to its side (unit normal
# SIDE_NORMALS, pointing into the specimen); the 27 transducers on the other 3 sides receive.
FIRING_SIDE  = 'top'
FIRING_INDEX = 4                                                           # top side, x = 5 mm
_line = np.linspace(0.001, 0.009, 9)
SIDES = ['bottom', 'top', 'left', 'right']
_side_xy = {'bottom': np.column_stack([_line, np.full(9, y0)]), 'top':   np.column_stack([_line, np.full(9, y1)]),
            'left':   np.column_stack([np.full(9, x0), _line]), 'right': np.column_stack([np.full(9, x1), _line])}
SIDE_NORMALS = {'bottom': (0.0, 1.0), 'top': (0.0, -1.0), 'left': (1.0, 0.0), 'right': (-1.0, 0.0)}
trd_loc  = np.vstack([_side_xy[s] for s in SIDES])                         # all 36 transducers [m]
trd_side = np.repeat(SIDES, 9)
tx_locations = _side_xy[FIRING_SIDE][[FIRING_INDEX]]                       # transmitting transducer [m]
r_loc_all    = trd_loc[trd_side != FIRING_SIDE]                            # 27 receiving transducers [m]
r_side       = trd_side[trd_side != FIRING_SIDE]
RX_SIDES     = [s for s in SIDES if s != FIRING_SIDE]


# =============================================================================
# Functions
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


def oriented_material(material_unoriented, angle_in_degrees):
    """Rotate a hexagonal material and convert it to the 2D solver form (C11..C33)."""
    orientation = sn.material.orientation.ClockwiseAngle.from_params(angle_in_degrees=angle_in_degrees)
    return md.to_solver_form(material_unoriented.with_orientation(orientation), ndim=2)


def stiffness_from_theta(theta, matl):
    """Rotated 2D stiffness for orientation theta [rad] (ClockwiseAngle, Salvus Voigt names), as the FWI."""
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


def grain_ids_of_points(indexer, xz):
    """
    Grain id of every point (x, z) [m] (any array shape (..., 2)); points outside the physical
    domain (absorbing layers) take the grain at the nearest domain edge.
    """
    xn = np.clip((xz[..., 0] - x0) / (x1 - x0), 0.0, 1.0)
    zn = np.clip((xz[..., 1] - y0) / (y1 - y0), 0.0, 1.0)
    col = np.rint(xn * (indexer.W - 1)).astype(int)
    row = np.rint((1.0 - zn) * (indexer.H - 1)).astype(int)    # image rows run top -> bottom
    gid = indexer.seg_labels[row, col]
    for idx in zip(*np.nonzero(gid < 0)):                       # unlabelled pixels: neighbour vote
        gid[idx] = indexer.point_to_grain_id(float(xn[idx]), float(zn[idx]))
    return gid


def forward_simulation(simulation_name, events, fields=None, sampling_interval_in_time_steps=10):
    """Launch a forward simulation in project `p` and block until it finishes."""
    extra = ({"volume_data": {"sampling_interval_in_time_steps": sampling_interval_in_time_steps,
                              "fields": fields}} if fields else None)
    t_start = time.perf_counter()
    p.simulations.launch(ranks_per_job=RANKS, site_name=SITE_NAME, events=events,
                         simulation_configuration=simulation_name,
                         extra_output_configuration=extra, delete_conflicting_previous_results=True)
    while True:   # query(block=True) occasionally raises while polling a running job
        try:
            p.simulations.query(simulation_configuration=simulation_name, events=events,
                                block=True, verbosity=-1)
            break
        except (AssertionError, ValueError) as e:
            print(f"  [forward_simulation] query retry after {type(e).__name__}: {e}")
            time.sleep(2)
    print(f"  [forward_simulation] '{simulation_name}' finished in {time.perf_counter() - t_start:.1f} s")


# =============================================================================
# Voronoi grain map and orientations
# =============================================================================

if not VORONOI_PNG.exists():
    generate_voronoi_png(VORONOI_PNG, N_GRAIN, VORONOI_SEED, OUTPUT_PX)
indexer = VoronoiGrainIndexer(str(VORONOI_PNG), min_percent=0.1, dilate_iters=1)
indexer.process()
n_grains_found = len(indexer.kept_colors_sorted)
print(f"Grain map {VORONOI_PNG.name}: {n_grains_found} grains, areas (%) "
      f"{np.round(indexer.areas_df['Percentage (%)'].to_numpy(), 1).tolist()}")
assert n_grains_found == N_GRAIN, f"expected {N_GRAIN} grains, segmentation found {n_grains_found}"

rng = np.random.default_rng(THETA_SEED)
theta_grain_deg = rng.normal(THETA_MEAN_DEG, THETA_STD_DEG, size=n_grains_found)   # one per grain id
for g, t in enumerate(theta_grain_deg):
    print(f"  grain {g} ({indexer.areas_df['Percentage (%)'][g]:5.1f} % area): theta = {t:6.2f} deg")
if np.any((theta_grain_deg < 0) | (theta_grain_deg > 90)):
    print("  [note] some grain angles fall outside 0-90 deg (Gaussian tails); they are kept as drawn")


# =============================================================================
# Project, mesh and per-node orientation
# =============================================================================

apply_paper_style()
domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)
p = sn.Project.from_domain(path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True)

material_unoriented = sn.material.elastic.hexagonal.TensorComponents.from_params(**matl.VTI_parameters())
matl_mean = oriented_material(material_unoriented, THETA_MEAN_DEG)   # mesh sizing only (rotation-invariant)

def build_voronoi_mesh(reference_frequency):
    """
    Box mesh resolving `reference_frequency` (elements_per_wavelength, model_order), every GLL node
    set to the rotated stiffness of the grain it lies in; time step 90 % of the CFL limit.

    Returns mesh, grain id per GLL node (nelem, n_gll), time step [s], CFL time step [s].
    """
    ab_params = salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
        free_surface=free_surfaces, number_of_wavelengths=number_of_wavelengths,
        reference_velocity=VP_0deg, reference_frequency=reference_frequency)
    mesh_res = sn.MeshResolution(reference_frequency=reference_frequency,
                                 elements_per_wavelength=elements_per_wavelength, model_order=model_order)
    mesh = lm.mesh_from_domain(domain=domain,
                               # all 4 sides free -> no absorbing layer (an empty one leaves a '' side
                               # set that Salvus rejects at launch)
                               model=sn.layered_meshing.MeshingProtocol(
                                   lm.LayeredModel([matl_mean]),
                                   ab=None if len(free_surfaces) == 4 else ab_params),
                               mesh_resolution=mesh_res)

    # every GLL node: grain id from the PNG -> theta of that grain -> rotated stiffness
    grain_en = grain_ids_of_points(indexer, mesh.get_element_nodes())   # (nelem, n_gll)
    C = stiffness_from_theta(np.deg2rad(theta_grain_deg[grain_en]), matl)
    for k in STIFFNESS_PARAMETERS:
        mesh.element_nodal_fields[k][:] = C[k]

    # time step: 90 % of the CFL limit (fastest qP bounded by the largest Voigt eigenvalue)
    f = mesh.element_nodal_fields
    Cv = np.stack([np.stack([f['C11'], f['C12'], f['C13']], -1),
                   np.stack([f['C12'], f['C22'], f['C23']], -1),
                   np.stack([f['C13'], f['C23'], f['C33']], -1)], -2)
    v_max = np.sqrt(np.linalg.eigvalsh(Cv)[..., -1] / f['RHO']).max(axis=1)
    dt_cfl = mesh.compute_time_step(max_velocity=v_max)[0]
    return mesh, grain_en, np.floor(0.9 * dt_cfl * 1e11) / 1e11, dt_cfl


def snapped_time_window(dt):
    """Simulation start / end time snapped to the dt grid (an integer number of steps)."""
    return np.floor(START_TIME / dt) * dt, np.round(END_TIME / dt) * dt


START_TIME, END_TIME = start_time, end_time                     # before snapping to a mesh's dt
mesh, grain_en, time_step_in_seconds, _dt_cfl = build_voronoi_mesh(reference_frequency)
print(f"Mesh: {mesh.nelem} elements, {mesh.npoint} nodes; grains per GLL node: "
      f"{np.bincount(grain_en.ravel(), minlength=n_grains_found).tolist()}")

# consistency of the closed-form rotation with Salvus' own rotation
_C_check = stiffness_from_theta(np.deg2rad(THETA_MEAN_DEG), matl)
for k in STIFFNESS_PARAMETERS:
    assert np.isclose(_C_check[k], getattr(matl_mean, k).p, rtol=1e-9, atol=1e-3), k

start_time, end_time = snapped_time_window(time_step_in_seconds)
print(f"CFL {_dt_cfl * 1e9:.3f} ns -> dt {time_step_in_seconds * 1e9:.2f} ns, "
      f"t = [{start_time * 1e6:.2f}, {end_time * 1e6:.2f}] us")

np.savez_compressed(VORONOI_DATA / "voronoi_model.npz", theta_grain_deg=theta_grain_deg,
                    grain_areas_percent=indexer.areas_df['Percentage (%)'].to_numpy(),
                    n_grain=N_GRAIN, voronoi_seed=VORONOI_SEED, theta_seed=THETA_SEED,
                    theta_mean_deg=THETA_MEAN_DEG, theta_std_deg=THETA_STD_DEG)


# =============================================================================
# Events and configuration figure
# =============================================================================

receivers = [sn.simple_config.receiver.cartesian.Point2D(x=r[0], y=r[1], station_code=f"R{j + 1}",
                                                         fields=["displacement"])
             for j, r in enumerate(r_loc_all)]
EVENT = f"{FIRING_SIDE}_{FIRING_INDEX + 1}"                       # e.g. 'top_5'
nx, nz = SIDE_NORMALS[FIRING_SIDE]                                  # force normal to the side
ev = sn.Event(event_name=EVENT, receivers=receivers,
              sources=sn.simple_config.source.cartesian.VectorPoint2D(
                  x=tx_locations[0, 0], y=tx_locations[0, 1],
                  fx=nx * amplitude_ratio, fy=nz * amplitude_ratio))
p.events.update(ev) if p.events.exists(EVENT) else p.add_to_project(ev)
event_ls = [EVENT]

# configuration: receiving transducers (gold), transmitting (red), inactive on the firing side (open)
fig, ax = plot_orientation_field(mesh, tx_locations, r_loc_all, extent=(x0, x1, y0, y1),
                                 transducers=True)
_idle = np.array([xy for xy in _side_xy[FIRING_SIDE] if not np.allclose(xy, tx_locations[0])])
ax.scatter(_idle[:, 0] * MM, _idle[:, 1] * MM, marker='s', facecolors='white', edgecolors='k',
           linewidths=0.4, s=18, zorder=5, clip_on=False, label='Inactive')
ax.scatter(tx_locations[:, 0] * MM, tx_locations[:, 1] * MM, marker='s', color='red', edgecolors='k',
           linewidths=0.4, s=24, zorder=6, clip_on=False, label='Transmitting')
_h, _l = ax.get_legend_handles_labels()
ax.legend(_h, ['Receiving' if l == 'Transducer' else l for l in _l], loc='upper right',
          framealpha=0.85, edgecolor='none', handletextpad=0.3)
save_paper_figure(fig, PAPER_FIG_DIR / "voronoi_configuration")
plt.close(fig)


# =============================================================================
# Forward simulation
# =============================================================================

def add_simulation(name, mesh, dt):
    """Simulation configuration `name`: Ricker at CENTRAL_FREQUENCY on `mesh` with time step dt."""
    t0, t1 = snapped_time_window(dt)
    p.add_to_project(sn.UnstructuredMeshSimulationConfiguration(
        unstructured_mesh=mesh, name=name,
        event_configuration=sn.EventConfiguration(
            wavelet=sn.simple_config.stf.Ricker(center_frequency=CENTRAL_FREQUENCY),
            waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
                start_time_in_seconds=t0, end_time_in_seconds=t1, time_step_in_seconds=dt))),
        overwrite=True)


SIM_NAME = "forward_voronoi"
add_simulation(SIM_NAME, mesh, time_step_in_seconds)

print(f"Event {EVENT}: transducer at ({tx_locations[0, 0] * MM:.0f}, {tx_locations[0, 1] * MM:.0f}) mm "
      f"transmits, {len(r_loc_all)} transducers on the other 3 sides receive")
forward_simulation(SIM_NAME, event_ls)                             # traces at the transducers only


def receiver_traces(event, sim_name=SIM_NAME):
    """Displacement traces (n_rx, 2, n_t) of one event, rows ordered as r_loc_all, components (x, z)."""
    d = p.waveforms.get(data_name=sim_name, events=event)[0].get_waveform_data_xarray(
        receiver_field='displacement')
    return np.stack([np.stack([d.sel(receiver_id=f"XX.R{j + 1}.", component=c).values.squeeze()
                               for c in ('X', 'Y')]) for j in range(len(r_loc_all))]), d.time.values


traces, t_rx = receiver_traces(EVENT)                             # (n_rx, 2, n_t)
np.savez_compressed(VORONOI_DATA / "voronoi_transducer_traces.npz", traces=traces, time=t_rx,
                    receiver_locs=r_loc_all, receiver_side=r_side, source_loc=tx_locations[0],
                    source_side=FIRING_SIDE)
print(f"Saved transducer traces {traces.shape} to {VORONOI_DATA / 'voronoi_transducer_traces.npz'}")


# =============================================================================
# Gathers of the receiving transducers: one panel per receiving side, the displacement component
# normal to that side, traces offset by the position along the side
# =============================================================================

_tr = traces
_scale = 0.45 / (np.abs(_tr).max() or 1.0)                        # common scale: 0.45 mm per max |u|
fig, axes = plt.subplots(1, len(RX_SIDES), figsize=(DOUBLE_COL, 3.2), sharey=True, layout='constrained')
for k, (ax, side) in enumerate(zip(axes, RX_SIDES)):
    comp, name = (1, 'z') if side in ('bottom', 'top') else (0, 'x')
    pos = 0 if side in ('bottom', 'top') else 1                    # coordinate along the side
    for j in np.flatnonzero(r_side == side):
        ax.plot(r_loc_all[j, pos] * MM + _scale * _tr[j, comp], t_rx * 1e6, color=C_SYN, lw=0.6)
    ax.set_xlabel(f'Transducer ${"xz"[pos]}$ (mm)')
    ax.set_title(f'({"abcd"[k]}) {side}, $u_{name}$', loc='left')
    ax.grid(True, **GRID_KW)
axes[0].set_ylabel(r'Time ($\mu$s)')
axes[0].invert_yaxis()
save_paper_figure(fig, PAPER_FIG_DIR / "voronoi_transducer_gather")
plt.close(fig)


# =============================================================================
# Mesh convergence: the same event on meshes resolving reference_frequency = factor * CENTRAL_FREQUENCY,
# coarse -> fine. Traces are interpolated onto the time axis of the finest mesh (the reference) and
# compared through the relative L2 error over all receiving transducers and both components.
# =============================================================================

if RUN_CONVERGENCE:
    conv = {}
    for fac in CONVERGENCE_FACTORS:
        if np.isclose(fac * CENTRAL_FREQUENCY, reference_frequency):    # the main mesh: already simulated
            conv[fac] = dict(traces=traces, time=t_rx, nelem=mesh.nelem, dt=time_step_in_seconds)
            continue
        mesh_k, _, dt_k, _ = build_voronoi_mesh(fac * CENTRAL_FREQUENCY)
        name = f"forward_voronoi_conv_{fac:g}"
        add_simulation(name, mesh_k, dt_k)
        print(f"Convergence: mesh for {fac:g} f_c = {fac * CENTRAL_FREQUENCY / 1e6:.1f} MHz, "
              f"{mesh_k.nelem} elements, dt {dt_k * 1e9:.2f} ns")
        forward_simulation(name, event_ls)
        tr_k, t_k = receiver_traces(EVENT, name)
        conv[fac] = dict(traces=tr_k, time=t_k, nelem=mesh_k.nelem, dt=dt_k)

    fac_ref = max(CONVERGENCE_FACTORS)
    t_ref, u_ref = conv[fac_ref]['time'], conv[fac_ref]['traces']

    def on_reference_time(c):
        """Traces of one mesh interpolated onto t_ref (zero outside their own time window)."""
        return np.stack([np.stack([np.interp(t_ref, c['time'], c['traces'][j, i], left=0.0, right=0.0)
                                   for i in range(2)]) for j in range(c['traces'].shape[0])])

    u_on_ref = {fac: on_reference_time(conv[fac]) for fac in CONVERGENCE_FACTORS}
    rel_err = {fac: np.linalg.norm(u_on_ref[fac] - u_ref) / np.linalg.norm(u_ref)
               for fac in CONVERGENCE_FACTORS}
    print(f"\nMesh convergence (reference: {fac_ref:g} f_c), {elements_per_wavelength} elements per "
          f"minimum wavelength at the mesh frequency, order {model_order}")
    print(f"{'f_mesh/f_c':>10s} {'elements':>9s} {'dt (ns)':>8s} {'rel. L2 error':>14s}")
    for fac in CONVERGENCE_FACTORS:
        print(f"{fac:10g} {conv[fac]['nelem']:9d} {conv[fac]['dt'] * 1e9:8.2f} {rel_err[fac]:14.3e}")
    np.savez_compressed(VORONOI_DATA / "voronoi_mesh_convergence.npz",
                        factors=np.array(CONVERGENCE_FACTORS), central_frequency=CENTRAL_FREQUENCY,
                        nelem=[conv[f]['nelem'] for f in CONVERGENCE_FACTORS],
                        dt=[conv[f]['dt'] for f in CONVERGENCE_FACTORS],
                        rel_err=[rel_err[f] for f in CONVERGENCE_FACTORS],
                        time=t_ref, traces=np.stack([u_on_ref[f] for f in CONVERGENCE_FACTORS]),
                        receiver_locs=r_loc_all)

    # (a) relative error vs elements per wavelength at f_c, (b) u_z at the bottom-middle transducer
    j_rx = np.flatnonzero(r_side == 'bottom')[len(_line) // 2]
    epw_fc = elements_per_wavelength * np.array(CONVERGENCE_FACTORS)   # per min. wavelength at f_c
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8), layout='constrained')
    tested = [f for f in CONVERGENCE_FACTORS if f != fac_ref]
    axes[0].loglog(elements_per_wavelength * np.array(tested), [rel_err[f] for f in tested],
                   color=C_SYN, marker='o', ms=3, lw=0.9)
    axes[0].axvline(elements_per_wavelength * reference_frequency / CENTRAL_FREQUENCY, color='0.5',
                    lw=0.6, ls='--', label='Mesh used (forward, FWI)')
    axes[0].set_xticks(epw_fc[:-1], [f'{e:g}' for e in epw_fc[:-1]])
    axes[0].minorticks_off()
    axes[0].set_xlabel(r'Elements per minimum wavelength at $f_c$')
    axes[0].set_ylabel(r'Relative $L_2$ error')
    axes[0].set_title(f'(a) Convergence (reference: {elements_per_wavelength * fac_ref:g} el./$\\lambda$)',
                      loc='left')
    axes[0].legend(frameon=False, loc='upper right')
    axes[0].grid(True, which='both', **GRID_KW)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(tested)))
    for f, c in zip(tested, colors):
        axes[1].plot(t_ref * 1e6, u_on_ref[f][j_rx, 1], color=c, lw=0.8,
                     label=f'{elements_per_wavelength * f:g} el./$\\lambda$')
    axes[1].plot(t_ref * 1e6, u_ref[j_rx, 1], color='k', lw=0.8, ls='--',
                 label=f'{elements_per_wavelength * fac_ref:g} el./$\\lambda$ (ref.)')
    axes[1].set_xlabel(r'Time ($\mu$s)'); axes[1].set_ylabel('$u_z$ (m)')
    axes[1].set_title(f'(b) $u_z$ at $x$ = {r_loc_all[j_rx, 0] * MM:g} mm (bottom)', loc='left')
    axes[1].legend(frameon=False, loc='upper left', ncol=2)
    axes[1].grid(True, **GRID_KW)
    save_paper_figure(fig, PAPER_FIG_DIR / "voronoi_mesh_convergence")
    plt.close(fig)
