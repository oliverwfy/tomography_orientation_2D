"""
Elastic reverse time migration (RTM) of point defects, multi-source acquisition, migrated with the
grain-orientation profile reconstructed by FWI_orientation_multilayer_multisource.py.

Same two-pass RTM as RTM_orientation.py, with the source array of
FWI_orientation_multilayer_multisource.py: 9 sources at x = 1 ... 9 mm on the top surface,
each a separate event recorded by the 9 receivers at x = 1 ... 9 mm on the bottom surface.
Per source event i:

  observed data   : true layered theta(z) model WITH the 4 point defects
  reference data  : true layered model WITHOUT defects (same mesh, same acquisition)
  residual_i      : observed_i - reference_i = defect-scattered displacement only
  forward  u_f,i  : wavefield of source i in the migration model
  backward u_b,i  : time-reversed residual_i injected at all receivers (2-component forces)
  image_i         : zero-lag cross-correlation, I_i(x) = sum_t sum_c u_f,i,c(x, t) u_b,i,c(x, T - t) dt

The stacked image is I = sum_i I_i; the illumination-normalised image is
I / sum_i sum_t |u_f,i|^2 (saved as well). Migration models: 'reconstructed' (final multi-source
FWI theta(z) profile), and for comparison 'true' (true theta(z)) and 'homogeneous' (FWI starting
model, 45 deg).

Salvus attaches the wavelet to a simulation configuration, not to an event, so each source's
backward run is its own configuration (rtm_backward_<model>_src_<i>); the forward runs take all
9 events in one configuration.

Defects: Gaussian impedance perturbations of width DEFECT_SIGMA at the RTM_orientation.py centres.
Unlike that script's single-GLL-node scatterers, whose strength depends on where the nearest node
sits in its element (a corner node weighs ~9x less than an interior one, so the defect at z = 1.5 mm,
on a row of element corners, scattered ~85x less energy than the others), a smooth Gaussian is
integrated accurately by the GLL quadrature: all four defects have the same strength (to 0.1 %)
whatever the mesh. Wavelet, time window and mesh as RTM_orientation.py.

Run from the tomograhpy folder (after FWI_orientation_multilayer_multisource.py has saved its
histories):
    python RTM_orientation_multisource.py
"""

import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file-only backend: figures are saved, never shown
import matplotlib.pyplot as plt

import salvus.mesh
import salvus.mesh.layered_meshing as lm
import salvus.namespace as sn
import salvus.flow.simple_config as sc
from salvus.material._details import material as md
from salvus.toolbox.helpers.wavefield_output import WavefieldOutput, wavefield_output_to_xarray

from my_code.plot_style import (apply_paper_style, save_paper_figure, plot_orientation_field,
                                plot_sources_receivers, PAPER_FIGSIZE, DOUBLE_COL, MM,
                                C_SYN, C_TRUE, GRID_KW)
from material_ela_constants.Elastic_Material import Austenite


# =============================================================================
# Paths
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

PROJECT_NAME  = "RTM_orientation_multilayer_multisource"                        # separate from the FWI projects
FWI_DATA_DIR  = Path(DATA_DIR_WIN) / "FWI_orientation_multilayer_multisource"    # reconstructed theta (FWI output)
RTM_DATA_DIR  = Path(DATA_DIR_WIN) / "RTM_orientation_multilayer_multisource"
RTM_IMAGE_DIR = Path(IMAGE_DIR_WIN) / "RTM_orientation_multilayer_multisource"
PAPER_FIG_DIR = RTM_IMAGE_DIR / "paper_figures"

# Start every run from scratch: delete this script's previous data, images and Salvus project
for _d in (RTM_IMAGE_DIR, RTM_DATA_DIR, Path(PROJECT_DIR_WIN, PROJECT_NAME)):
    if _d.exists():
        print(f"Deleting previous results: {_d}")
        shutil.rmtree(_d)
for _d in (RTM_DATA_DIR, PAPER_FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SITE_NAME = "isambard_oliver" if _ON_ISAMBARD else "oliver_wsl"   # Salvus site to run simulations on
RANKS     = 2


# =============================================================================
# Model: true layered orientation (as FWI_orientation_multilayer_multisource.py) and defects
# =============================================================================

x0, x1 = 0.0, 0.01                                              # physical domain [m]
y0, y1 = 0.0, 0.01

matl    = Austenite()
VP_0deg = np.sqrt(matl.C33 / matl.RHO)
STIFFNESS_PARAMETERS = ['C11', 'C12', 'C13', 'C22', 'C23', 'C33']   # Salvus 2D Voigt names

BACKGROUND_ANGLE_DEG = 45.0
LAYER_INTERFACES_Y   = [0.009, 0.007, 0.006, 0.0045, 0.002, 0.001]   # top to bottom [m]
LAYER_ANGLES_DEG     = [70.0, 30.0, 60.0, 20.0, 40.0]               # layers 1-5

DOMAIN_SCALE           = 1e-4                                   # RTM.py metres -> here metres (100 m -> 10 mm)
DEFECT_CENTERS         = np.column_stack([np.linspace(20, 80, 4), np.linspace(15, 75, 4)]) * DOMAIN_SCALE
DEFECT_SIGMA           = 0.1e-3                                 # Gaussian width [m]: ~ GLL spacing, lambda_S / 30
DEFECT_IMPEDANCE_FACTOR = 2.0                                   # peak: stiffness AND density x this (velocity unchanged)

MIGRATION_MODELS = ['reconstructed', 'true', 'homogeneous']     # 'reconstructed' is the target


# =============================================================================
# Acquisition (as FWI_orientation_multilayer_multisource.py) and wavelet
# =============================================================================

CENTRAL_FREQUENCY = 1e6                                         # Ricker centre frequency [Hz]
f_c               = CENTRAL_FREQUENCY
amplitude_ratio   = 1e10                                        # source force (fy)

r_loc_all    = np.array([[x, 0.0] for x in np.linspace(0.001, 0.009, 9)])          # bottom surface [m]
tx_locations = np.column_stack([r_loc_all[:, 0], np.full(len(r_loc_all), y1)])     # top surface [m]
n_rx, n_src  = len(r_loc_all), len(tx_locations)

# start at 0 with a Ricker delayed by one period (as RTM.py); 10 us covers the qSV defect
# echoes of the longest source-defect-receiver path (~18 mm / 2.4 km/s ~ 7.5 us + 1 us delay)
start_time             = 0.0
end_time               = 10e-6
time_shift_in_seconds  = 1.0 / f_c
VOLUME_SAMPLES_PER_PERIOD = 20                                  # saved wavefield snapshots per period


# =============================================================================
# Mesh resolution and boundaries (as RTM_orientation.py)
# =============================================================================

elements_per_wavelength = 4
model_order             = 4
reference_frequency     = CENTRAL_FREQUENCY * 2
number_of_wavelengths   = 3
free_surfaces           = ['y0', 'y1']                          # sources / receivers on free surfaces


# =============================================================================
# Orientation helpers (as RTM_orientation.py: importing an FWI script runs the FWI)
# =============================================================================

def oriented_material(material_unoriented, angle_in_degrees):
    """Rotate a hexagonal material and convert it to the 2D solver form (C11..C33)."""
    orientation = sn.material.orientation.ClockwiseAngle.from_params(angle_in_degrees=angle_in_degrees)
    return md.to_solver_form(material_unoriented.with_orientation(orientation), ndim=2)


def stiffness_from_theta(theta, matl):
    """Rotated 2D stiffness for orientation theta [rad] (ClockwiseAngle convention, Salvus Voigt names)."""
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


def reconstructed_profile():
    """Final FWI theta(z) [deg] on the depth rows of the inverted region (physical x only)."""
    models = np.load(FWI_DATA_DIR / 'model_history_theta_rad_multiscale.npy')
    xy = np.load(FWI_DATA_DIR / 'roi_node_xy.npy')
    inside = (xy[:, 0] >= x0 - 1e-9) & (xy[:, 0] <= x1 + 1e-9)
    z_round = np.round(xy[inside, 1], 10)
    z_lv, row = np.unique(z_round, return_inverse=True)
    theta = np.bincount(row, weights=np.rad2deg(models[-1][inside])) / np.bincount(row)
    print(f"Reconstructed profile: {len(models)} saved FWI models, final one used; "
          f"z = {z_lv[0] * MM:.2f}-{z_lv[-1] * MM:.2f} mm, theta {theta.min():.1f}-{theta.max():.1f} deg")
    return z_lv, theta


def mesh_with_theta_profile(base_mesh, z_profile, theta_profile_deg):
    """Copy of `base_mesh` with theta(z) interpolated from the profile inside its z range, background outside."""
    z_nodes = base_mesh.get_element_nodes()[:, :, 1]
    theta = np.where((z_nodes >= z_profile[0] - 1e-9) & (z_nodes <= z_profile[-1] + 1e-9),
                     np.interp(z_nodes, z_profile, theta_profile_deg), BACKGROUND_ANGLE_DEG)
    m = base_mesh.copy()
    C = stiffness_from_theta(np.deg2rad(theta), matl)
    for k in STIFFNESS_PARAMETERS:
        m.element_nodal_fields[k][:] = C[k]
    return m


# =============================================================================
# Project, true mesh, defects and migration models
# =============================================================================

apply_paper_style()
domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)
p = sn.Project.from_domain(path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True)

material_unoriented = sn.material.elastic.hexagonal.TensorComponents.from_params(**matl.VTI_parameters())
matl_background = oriented_material(material_unoriented, BACKGROUND_ANGLE_DEG)
_stack = [matl_background]
for _y_if, _matl in zip(LAYER_INTERFACES_Y,
                        [oriented_material(material_unoriented, a) for a in LAYER_ANGLES_DEG] + [matl_background]):
    _stack += [sn.layered_meshing.interface.Hyperplane.at(_y_if), _matl]

ab_params = salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
    free_surface=free_surfaces, number_of_wavelengths=number_of_wavelengths,
    reference_velocity=VP_0deg, reference_frequency=reference_frequency,
)
mesh_res = sn.MeshResolution(reference_frequency=reference_frequency,
                             elements_per_wavelength=elements_per_wavelength, model_order=model_order)
mesh_true = lm.mesh_from_domain(domain=domain,
                                model=sn.layered_meshing.MeshingProtocol(lm.LayeredModel(_stack), ab=ab_params),
                                mesh_resolution=mesh_res)
print(f"True mesh: {mesh_true.nelem} elements")

# Gaussian scatterers: stiffness AND density x 1 + (factor - 1) exp(-r^2 / 2 sigma^2) around each
# centre (impedance perturbed, velocity unchanged everywhere)
mesh_defects = mesh_true.copy()
_nodes = mesh_defects.get_element_nodes()                       # (nelem, n_gll, 2)
defect_scale = np.ones(_nodes.shape[:2])
for c in DEFECT_CENTERS:
    defect_scale += (DEFECT_IMPEDANCE_FACTOR - 1.0) * np.exp(
        -np.sum((_nodes - c) ** 2, axis=-1) / (2.0 * DEFECT_SIGMA ** 2))
defect_nodes = DEFECT_CENTERS                                   # plotted at the Gaussian centres
for k in STIFFNESS_PARAMETERS + ['RHO']:
    mesh_defects.element_nodal_fields[k][:] *= defect_scale
print(f"Defects: {len(DEFECT_CENTERS)} Gaussian scatterers, sigma {DEFECT_SIGMA * MM:.2f} mm, stiffness and "
      f"density x {DEFECT_IMPEDANCE_FACTOR:g} at the centre (velocity unchanged), "
      f"max scale {defect_scale.max():.3f}")

z_rec, theta_rec = reconstructed_profile()
migration_meshes = {}
for name in MIGRATION_MODELS:
    if name == 'true':
        migration_meshes[name] = mesh_true
    elif name == 'reconstructed':
        migration_meshes[name] = mesh_with_theta_profile(mesh_true, z_rec, theta_rec)
    elif name == 'homogeneous':
        migration_meshes[name] = mesh_with_theta_profile(mesh_true, np.array([y0, y1]),
                                                         np.full(2, BACKGROUND_ANGLE_DEG))


# =============================================================================
# Time step from the CFL limit of the fastest mesh (impedance defects keep the background velocity)
# =============================================================================

def cfl_time_step(mesh):
    """CFL time step; fastest quasi-P speed bounded by the largest eigenvalue of the Voigt tensor."""
    f = mesh.element_nodal_fields
    C = np.stack([np.stack([f['C11'], f['C12'], f['C13']], -1),
                  np.stack([f['C12'], f['C22'], f['C23']], -1),
                  np.stack([f['C13'], f['C23'], f['C33']], -1)], -2)
    v_max = np.sqrt(np.linalg.eigvalsh(C)[..., -1] / f['RHO']).max(axis=1)
    return mesh.compute_time_step(max_velocity=v_max)[0]

_dt_cfl = min(cfl_time_step(m) for m in [mesh_defects] + list(migration_meshes.values()))
time_step_in_seconds   = np.floor(0.9 * _dt_cfl * 1e11) / 1e11           # 90 % of CFL, 0.01 ns grid
sampling_rate_in_hertz = 1.0 / time_step_in_seconds
end_time = np.round(end_time / time_step_in_seconds) * time_step_in_seconds
volume_interval = max(1, int(round(1.0 / (f_c * VOLUME_SAMPLES_PER_PERIOD) / time_step_in_seconds)))
print(f"CFL {_dt_cfl * 1e9:.3f} ns -> dt {time_step_in_seconds * 1e9:.2f} ns, "
      f"{int(round(end_time / time_step_in_seconds))} steps, volume output every {volume_interval} steps")


# =============================================================================
# Events (one per source) and simulation configurations
# =============================================================================

def register_event(event):
    """Add an event, or update it in place (Salvus refuses overwrite=True for events)."""
    if p.events.exists(event.event_name):
        p.events.update(event)
    else:
        p.add_to_project(event)


receivers = [sn.simple_config.receiver.cartesian.Point2D(x=r[0], y=r[1], station_code=f"R{j + 1}",
                                                         fields=["displacement"])
             for j, r in enumerate(r_loc_all)]
EVENTS = []
for i, s in enumerate(tx_locations):
    EVENTS.append(f"src_{i + 1}")
    register_event(sn.Event(event_name=EVENTS[-1], receivers=receivers,
                            sources=sn.simple_config.source.cartesian.VectorPoint2D(
                                x=s[0], y=s[1], fx=0.0, fy=amplitude_ratio)))
print(f"{len(EVENTS)} source events: {EVENTS}")

waveform_config = sn.WaveformSimulationConfiguration(start_time_in_seconds=start_time,
                                                     end_time_in_seconds=end_time,
                                                     time_step_in_seconds=time_step_in_seconds)
event_config = sn.EventConfiguration(
    wavelet=sn.simple_config.stf.Ricker(center_frequency=f_c, time_shift_in_seconds=time_shift_in_seconds),
    waveform_simulation_configuration=waveform_config,
)


def add_simulation(name, mesh, config):
    p.add_to_project(sn.UnstructuredMeshSimulationConfiguration(
        unstructured_mesh=mesh, name=name, event_configuration=config), overwrite=True)


def run_simulation(name, events, volume=False):
    """Launch `name` for `events` (optionally with volume displacement output) and wait for it."""
    t_start = time.perf_counter()
    extra = ({"extra_output_configuration": {"volume_data": {
        "sampling_interval_in_time_steps": volume_interval, "fields": ["displacement"]}}}
             if volume else {})
    p.simulations.launch(ranks_per_job=RANKS, site_name=SITE_NAME, events=events,
                         simulation_configuration=name, delete_conflicting_previous_results=True, **extra)
    while True:   # query(block=True) occasionally raises while polling; keep waiting
        try:
            p.simulations.query(simulation_configuration=name, events=events, block=True, verbosity=-1)
            break
        except (AssertionError, ValueError) as e:
            print(f"  [{name}] query retry after {type(e).__name__}: {e}")
            time.sleep(2)
    print(f"  [{name}] {len(events)} event(s) finished in {time.perf_counter() - t_start:.1f} s")


def delete_results(sim, events):
    """Free disk (volume output); a failed cleanup only warns."""
    try:
        p.simulations.delete_results(sim, events)
    except Exception as e:
        print(f"  [warning] cleanup of '{sim}' failed ({type(e).__name__}: {e})")


def receiver_array(sim_name, event):
    """Displacement traces (n_rx, 2, n_t) of `event`, rows ordered as r_loc_all, components (x, z)."""
    data = p.waveforms.get(data_name=sim_name, events=event)[0].get_waveform_data_xarray(
        receiver_field='displacement')
    return np.stack([np.stack([data.sel(receiver_id=f"XX.R{j + 1}.", component=c).values.squeeze()
                               for c in ('X', 'Y')]) for j in range(n_rx)]), data.time.values


nx = ny = 201
x_img = np.linspace(x0, x1, nx)
y_img = np.linspace(y0, y1, ny)


def volume_field(sim_name, event_name):
    """Volume displacement on the (x_img, y_img) grid, array (ny, nx, 2, n_t)."""
    out = WavefieldOutput.from_file(
        Path(p.simulations.get_simulation_output_directory(sim_name, event_name), "volume_data_output.h5"),
        "displacement", "volume")
    u = wavefield_output_to_xarray(out, [x_img, y_img]).T          # dims (y, x, c, t), as RTM.py
    return u.transpose("y", "x", "c", "t").values.astype(np.float32), u["t"].values


# =============================================================================
# Configuration figure: true orientation, defects, sources and receivers
# =============================================================================

def draw_defects(ax, color='k'):
    """Point scatterers as open circles around their GLL nodes."""
    ax.scatter(defect_nodes[:, 0] * MM, defect_nodes[:, 1] * MM, s=30, facecolors='none',
               edgecolors=color, linewidths=0.7, zorder=6)


fig, ax = plot_orientation_field(mesh_true, tx_locations, r_loc_all, extent=(x0, x1, y0, y1))
draw_defects(ax, color='w')
save_paper_figure(fig, PAPER_FIG_DIR / "RTM_configuration")
plt.close(fig)

# migration models: theta(z) of each (true, reconstructed, homogeneous)
fig, ax = plt.subplots(figsize=PAPER_FIGSIZE, layout='constrained')
_z = np.linspace(y0, y1, 2001)
_t_true = np.full_like(_z, BACKGROUND_ANGLE_DEG)
for (top, bot), a in zip(zip(LAYER_INTERFACES_Y[:-1], LAYER_INTERFACES_Y[1:]), LAYER_ANGLES_DEG):
    _t_true[(_z <= top) & (_z > bot)] = a
_t_rec = np.where((_z >= z_rec[0]) & (_z <= z_rec[-1]), np.interp(_z, z_rec, theta_rec), BACKGROUND_ANGLE_DEG)
ax.plot(_t_true, _z * MM, color=C_TRUE, lw=1.0, ls='--', label='True')
ax.plot(_t_rec, _z * MM, color=C_SYN, lw=1.0, label='Reconstructed (FWI)')
ax.plot(np.full_like(_z, BACKGROUND_ANGLE_DEG), _z * MM, color='0.5', lw=0.8, ls=':', label='Homogeneous')
for c in DEFECT_CENTERS:
    ax.axhline(c[1] * MM, color='0.7', lw=0.4, zorder=0)
ax.set_xlabel(r'$\theta$ (deg)'); ax.set_ylabel('$z$ (mm)')
ax.set_ylim(y0 * MM, y1 * MM)
ax.legend(frameon=False, loc='best'); ax.grid(True, **GRID_KW)
save_paper_figure(fig, PAPER_FIG_DIR / "RTM_migration_models")
plt.close(fig)


# =============================================================================
# 1. Observed (with defects) and reference (without) data -> defect-scattered residual per source
# =============================================================================

t0 = datetime.now()
add_simulation("rtm_observed", mesh_defects, event_config)
add_simulation("rtm_reference", mesh_true, event_config)

print("Observed data (true model with defects), all sources ...")
run_simulation("rtm_observed", EVENTS)
# the reference run is also the forward wavefield of the 'true' migration model
print("Reference data (true model without defects) + forward wavefields of the true model ...")
run_simulation("rtm_reference", EVENTS, volume='true' in MIGRATION_MODELS)

observed, reference = [], []
for ev in EVENTS:
    _obs, t_rx = receiver_array("rtm_observed", ev)
    observed.append(_obs)
    reference.append(receiver_array("rtm_reference", ev)[0])
observed, reference = np.array(observed), np.array(reference)   # (n_src, n_rx, 2, n_t)
residual = observed - reference
np.savez_compressed(RTM_DATA_DIR / "rtm_receiver_traces.npz", observed=observed, reference=reference,
                    residual=residual, time=t_rx, receiver_locs=r_loc_all, source_locs=tx_locations)
print(f"Residual / observed energy: {np.sum(residual ** 2) / np.sum(observed ** 2):.3e}")


# =============================================================================
# 2. Backward events: time-reversed residual of each source injected at every receiver
# =============================================================================

BACKWARD_EVENTS, event_configs_adj = {}, {}
for i, ev in enumerate(EVENTS):
    f_adj = np.flip(residual[i], axis=-1) / sampling_rate_in_hertz   # as RTM.py
    BACKWARD_EVENTS[ev] = f"rtm_backward_sources_{ev}"
    register_event(sn.Event(event_name=BACKWARD_EVENTS[ev], receivers=None, sources=[
        sn.simple_config.source.cartesian.VectorPoint2D(x=r[0], y=r[1], fx=1.0, fy=1.0) for r in r_loc_all]))
    event_configs_adj[ev] = sn.EventConfiguration(
        wavelet=[sc.stf.Custom.from_array(f_adj[j].T, sampling_rate_in_hertz=sampling_rate_in_hertz,
                                          start_time_in_seconds=start_time) for j in range(n_rx)],
        waveform_simulation_configuration=waveform_config,
    )


# =============================================================================
# 3. Forward / backward wavefields per migration model and source, imaging condition, stack
# =============================================================================

images, images_norm, images_per_source = {}, {}, {}
for name in MIGRATION_MODELS:
    print(f"\n=== Migration model: {name} ===")
    fwd_sim = "rtm_reference" if name == 'true' else f"rtm_forward_{name}"
    if name != 'true':
        add_simulation(fwd_sim, migration_meshes[name], event_config)
        run_simulation(fwd_sim, EVENTS, volume=True)

    img_src = np.zeros((n_src, ny, nx))
    illum = np.zeros((ny, nx))
    for i, ev in enumerate(EVENTS):
        bwd_sim = f"rtm_backward_{name}_{ev}"
        add_simulation(bwd_sim, migration_meshes[name], event_configs_adj[ev])
        run_simulation(bwd_sim, [BACKWARD_EVENTS[ev]], volume=True)

        u_f, t_f = volume_field(fwd_sim, ev)
        u_b, t_b = volume_field(bwd_sim, BACKWARD_EVENTS[ev])
        assert np.allclose(t_f, t_b), "forward / backward snapshots on different time grids"
        dt_vol = float(t_f[1] - t_f[0])
        # u_b(x, T - t): flip the backward snapshots in time; sum over components and time
        img_src[i] = np.sum(u_f * np.flip(u_b, axis=-1), axis=(-2, -1)) * dt_vol      # (ny, nx)
        illum += np.sum(u_f ** 2, axis=(-2, -1)) * dt_vol
        del u_f, u_b
        delete_results(bwd_sim, [BACKWARD_EVENTS[ev]])

    images_per_source[name] = img_src
    images[name] = img_src.sum(axis=0)
    images_norm[name] = images[name] / (illum + 1e-3 * illum.max())
    np.savez_compressed(RTM_DATA_DIR / f"rtm_image_{name}.npz", image=images[name],
                        image_illumination_normalised=images_norm[name], image_per_source=img_src,
                        illumination=illum, x=x_img, y=y_img)
    if name != 'true':
        delete_results(fwd_sim, EVENTS)

for sim in ("rtm_observed", "rtm_reference"):
    delete_results(sim, EVENTS)

elapsed = (datetime.now() - t0).total_seconds()
print(f"\nRTM finished in {int(elapsed // 60)}m {elapsed % 60:.0f}s")


# =============================================================================
# Figures: stacked RTM images (one panel per migration model, each normalised by its 99th
# percentile) and, for the reconstructed model, the single central source vs the stack
# =============================================================================

TITLES = {'reconstructed': 'Reconstructed (FWI) model', 'true': 'True model',
          'homogeneous': r'Homogeneous model (45$^\circ$)'}


def image_panels(panels, tag):
    """One RTM image per panel, [(image, title, sources), ...], saved as PAPER_FIG_DIR / tag."""
    fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE_COL, 2.7), sharey=True,
                             layout='constrained')
    axes = np.atleast_1d(axes)
    for k, (ax, (image, title, src)) in enumerate(zip(axes, panels)):
        img = image / (np.percentile(np.abs(image), 99.0) or 1.0)
        tp = ax.pcolormesh(x_img * MM, y_img * MM, img, cmap='RdBu', vmin=-1, vmax=1,
                           shading='auto', rasterized=True)
        draw_defects(ax)
        plot_sources_receivers(ax, src, r_loc_all, legend=(k == 0))
        ax.set_xlim(x0 * MM, x1 * MM); ax.set_ylim(y0 * MM, y1 * MM); ax.set_aspect('equal')
        ax.set_xlabel('$x$ (mm)')
        ax.set_title(f'({"abc"[k]}) {title}', loc='left')
    axes[0].set_ylabel('$z$ (mm)')
    fig.colorbar(tp, ax=axes.tolist(), shrink=0.8, label='Normalised RTM amplitude')
    save_paper_figure(fig, PAPER_FIG_DIR / tag)
    plt.close(fig)


for tag, imgs in [("RTM_images", images), ("RTM_images_illumination_normalised", images_norm)]:
    image_panels([(imgs[name], TITLES[name], tx_locations) for name in MIGRATION_MODELS], tag)

if 'reconstructed' in MIGRATION_MODELS:
    _mid = n_src // 2
    image_panels([(images_per_source['reconstructed'][_mid],
                   f'Single source ($x$ = {tx_locations[_mid, 0] * MM:.0f} mm)', tx_locations[_mid:_mid + 1]),
                  (images['reconstructed'], f'Stack of {n_src} sources', tx_locations)],
                 "RTM_images_single_vs_stack")
