"""
2D elastic forward simulation through a single anisotropic layer.

Workflow
--------
1. Set up paths, solver/site settings and all physical constants.
2. Build the true model: a hexagonal (transversely isotropic) austenite
   background with one horizontal layer of different grain orientation.
3. Mesh the model (spectral elements + absorbing boundaries on x0/x1).
4. Register source/receiver events in the Salvus project.
5. Plot the true orientation field recovered from the meshed stiffness tensor.
6. Register the simulation configuration and run the forward simulation.
7. Sensitivity kernel K_theta = K_C :: dC/dtheta in the homogeneous initial
   model (Salvus built-in adjoint gradient K_C, L2 misfit vs. the true data).

FWI_orientation_single_layer.py and verify_kernel_C.py import this script.

Units: SI throughout (m, s, Pa, kg/m^3); plots use mm.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file-only backend: figures are saved, never shown (no X server needed)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import salvus.mesh
import salvus.mesh.layered_meshing as lm
import salvus.namespace as sn
from salvus.material._details import material as md
from salvus.toolbox.helpers.wavefield_output import wavefield_output_to_xarray

from my_code.utilities import *                          # elemental_nodal_to_nodal_field, ...
from material_ela_constants.Elastic_Material import *    # Austenite, ...


# =============================================================================
# 1. Paths
# =============================================================================

# Directories on the WSL file system
PROJECT_DIR = '/home/oliver/workspace/Salvus/elastic_model/anisotropic/Project'
IMAGE_DIR   = '/home/oliver/workspace/Salvus/elastic_model/anisotropic/image'
DATA_DIR    = '/home/oliver/workspace/Salvus/elastic_model/anisotropic/data'

# Directories on the Windows drive (mounted in WSL) — used for the project and outputs
PROJECT_DIR_WIN = '/mnt/f/Salvus_project/elastic_model/anisotropic/Project'
DATA_DIR_WIN    = '/mnt/f/Salvus_project/elastic_model/anisotropic/data'
IMAGE_DIR_WIN   = '/mnt/f/Salvus_project/elastic_model/anisotropic/image'

for _d in (IMAGE_DIR, DATA_DIR, IMAGE_DIR_WIN, DATA_DIR_WIN):
    Path(_d).mkdir(parents=True, exist_ok=True)

# Project name (= folder name inside PROJECT_DIR_WIN)
PROJECT_NAME = "tomography_solid_orientation_singlelayer"


# =============================================================================
# 2. Solver / compute site
# =============================================================================

SITE_NAME = "oliver_wsl"   # Salvus site to run simulations on
RANKS     = 8              # MPI ranks per simulation job


# =============================================================================
# 3. Source wavelet and time axis
# =============================================================================

CENTRAL_FREQUENCY      = 1e6                         # Ricker centre frequency [Hz]
f_c                    = 0.5e6           # short alias
# output sampling rate [Hz]; dt = 4 ns must stay below the mesh CFL limit
# (mesh.compute_time_step(max_velocity=5864) ~ 4.25 ns for the qS-resolved mesh)
sampling_rate_in_hertz = CENTRAL_FREQUENCY * 250
time_step_in_seconds   = 1 / sampling_rate_in_hertz  # solver/output time step [s]

time_ratio = 1                     # scales the recording length
# Salvus' Ricker peaks at t = 0: start 1.5 periods early so the wavelet rises
# smoothly from ~0 instead of switching on at its peak (broadband ringing)
start_time = -1.5 / CENTRAL_FREQUENCY   # simulation start time [s]
end_time   = 4e-6 * time_ratio     # simulation end time [s]

amplitude_ratio = 1e10              # scalar source amplitude (f of ScalarPoint2D)


# =============================================================================
# 4. Domain geometry
# =============================================================================

x0, x1 = 0.0, 0.01                 # horizontal extent [m] (10 mm)
y0, y1 = 0.0, 0.01                 # vertical extent [m]   (10 mm)

LAYER_TOP_Y    = 0.0065            # upper interface of the oriented layer [m]
LAYER_BOTTOM_Y = 0.0035            # lower interface of the oriented layer [m]


# =============================================================================
# 5. Material and orientations
# =============================================================================

matl     = Austenite()                         # hexagonal austenite (RHO, Cij)
VP_0deg  = np.sqrt(matl.C33 / matl.RHO)        # P velocity along symmetry axis [m/s]
VP_90deg = np.sqrt(matl.C11 / matl.RHO)        # P velocity normal to symmetry axis [m/s]


def min_quasi_shear_velocity(matl, n_angles=721):
    """Slowest qS phase velocity over all propagation directions in the 2D (x-z) plane [m/s]."""
    a = np.linspace(0.0, np.pi, n_angles)
    n1, n2 = np.cos(a), np.sin(a)
    # Christoffel matrix of the unrotated tensor (rotation does not change the minimum)
    G11 = matl.C11 * n1**2 + matl.C44 * n2**2
    G22 = matl.C44 * n1**2 + matl.C33 * n2**2
    G12 = (matl.C13 + matl.C44) * n1 * n2
    lam_min = 0.5 * (G11 + G22) - np.sqrt(0.25 * (G11 - G22)**2 + G12**2)
    return np.sqrt(lam_min.min() / matl.RHO)


VS_min = min_quasi_shear_velocity(matl)        # slowest wave in the model (~2414 m/s)

# Grain orientation angles passed to ClockwiseAngle (symmetry axis measured
# clockwise from +z).
BACKGROUND_ANGLE_DEG = 45.0
LAYER_ANGLE_DEG      = 60.0

smoothing = 0                                  # model smoothing (currently unused)


# =============================================================================
# 6. Mesh resolution and absorbing boundaries
# =============================================================================

elements_per_wavelength = 5                       # elements per minimum wavelength
model_order             = 4                       # spectral-element polynomial order
reference_velocity      = VS_min                  # size elements for the slowest (qS) wave [m/s]
ab_reference_velocity   = VP_0deg                 # absorbing layer thickness measured in P wavelengths [m/s]
reference_frequency     = CENTRAL_FREQUENCY * 2   # max frequency resolved by the mesh [Hz]
number_of_wavelengths   = 3                       # absorbing layer thickness (in wavelengths)
free_surfaces           = False            # sides without absorbing layer


# =============================================================================
# Functions
# =============================================================================

def oriented_material(material_unoriented, angle_in_degrees):
    """Rotate a hexagonal material and convert it to the 2D solver form (C11..C33)."""
    orientation = sn.material.orientation.ClockwiseAngle.from_params(
        angle_in_degrees=angle_in_degrees
    )
    return md.to_solver_form(material_unoriented.with_orientation(orientation), ndim=2)


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


def plot_orientation_field(mesh, src_locations, rec_locations, save_path=None,
                           show_glyphs=False, n_glyphs=15):
    """
    Plot the orientation field of `mesh` in mm with sources and receivers.

    show_glyphs : overlay line segments along the symmetry axis on an
                  n_glyphs x n_glyphs grid.
    """
    mm = 1e3  # m -> mm

    theta_deg, two_theta = orientation_from_stiffness(mesh)
    print(f"Orientation range in model: {theta_deg.min():.2f} - {theta_deg.max():.2f} deg")

    tri = mtri.Triangulation(mesh.points[:, 0] * mm, mesh.points[:, 1] * mm)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    tp = ax.tripcolor(tri, theta_deg, cmap='viridis', vmin=0.0, vmax=90.0, shading='gouraud')
    plt.colorbar(tp, ax=ax, label='Orientation (clockwise from +z) [deg]')

    if show_glyphs:
        # Interpolate cos/sin of 2*theta on a regular grid, then halve the angle
        xg, yg = np.meshgrid(np.linspace(x0 * mm, x1 * mm, n_glyphs)[1:-1],
                             np.linspace(y0 * mm, y1 * mm, n_glyphs)[1:-1])
        c2 = mtri.LinearTriInterpolator(tri, np.cos(two_theta))(xg, yg)
        s2 = mtri.LinearTriInterpolator(tri, np.sin(two_theta))(xg, yg)
        theta_g = 0.5 * np.arctan2(s2, c2)
        ax.quiver(xg, yg, np.sin(theta_g), np.cos(theta_g),
                  pivot='middle', headwidth=0, headlength=0, headaxislength=0,
                  color='k', scale=25, width=0.004)

    ax.scatter(src_locations[:, 0] * mm, src_locations[:, 1] * mm,
               marker='v', color='red', s=60, zorder=5, label='Source')
    ax.scatter(rec_locations[:, 0] * mm, rec_locations[:, 1] * mm,
               marker='^', color='green', s=40, zorder=5, label='Receiver')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('True (layered) model — grain orientation')
    ax.set_xlim(x0 * mm, x1 * mm)
    ax.set_ylim(y0 * mm, y1 * mm)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)


def forward_simulation(simulation_name, events,
                       fields=None, sampling_interval_in_time_steps=10):
    """
    Launch a forward simulation in project `p` and block until it finishes.

    Parameters
    ----------
    simulation_name : name of a registered simulation configuration
    events          : event name or list of event names
    fields          : volume wavefield fields to store (e.g. ["displacement"]);
                      None -> receiver data only
    sampling_interval_in_time_steps : volume output every N time steps
    """
    _n_events = 1 if isinstance(events, str) else len(events)
    print(f"  [forward_simulation] launching '{simulation_name}' "
          f"for {_n_events} event(s), volume_output={'yes' if fields else 'no'} ...")

    extra_output = None
    if fields:
        extra_output = {
            "volume_data": {
                "sampling_interval_in_time_steps": sampling_interval_in_time_steps,
                "fields": fields,
            },
        }

    t_start = time.perf_counter()
    p.simulations.launch(
        ranks_per_job=RANKS,
        site_name=SITE_NAME,
        events=events,
        simulation_configuration=simulation_name,
        extra_output_configuration=extra_output,
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


def theta_from_mesh(mesh):
    """
    Element-nodal grain orientation theta [rad] of a mesh (ClockwiseAngle convention).

    Same atan2 relation as `orientation_from_stiffness`, but evaluated on the
    element-nodal fields so it lines up with element-nodal gradients.
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


def plot_kernel(mesh, kernel, src_locations, rec_locations, title, label, save_path=None,
                clip_percentile=99.0):
    """Plot an element-nodal kernel in mm with a symmetric colour scale."""
    mm = 1e3
    k_nodal = elemental_nodal_to_nodal_field(kernel, mesh.connectivity)
    tri = mtri.Triangulation(mesh.points[:, 0] * mm, mesh.points[:, 1] * mm)

    # colour limit from the physical domain only (exclude the absorbing layer);
    # 99th percentile so the source/receiver singularities do not saturate it
    inside = ((mesh.points[:, 0] >= x0) & (mesh.points[:, 0] <= x1) &
              (mesh.points[:, 1] >= y0) & (mesh.points[:, 1] <= y1))
    vmax = np.percentile(np.abs(k_nodal[inside]), clip_percentile)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    tp = ax.tripcolor(tri, k_nodal, cmap='seismic', vmin=-vmax, vmax=vmax, shading='gouraud')
    plt.colorbar(tp, ax=ax, label=label)
    for y_if in (LAYER_TOP_Y, LAYER_BOTTOM_Y):
        ax.axhline(y_if * mm, color='k', lw=0.8, ls='--')
    ax.scatter(src_locations[:, 0] * mm, src_locations[:, 1] * mm,
               marker='v', color='red', s=60, zorder=5, label='Source')
    ax.scatter(rec_locations[:, 0] * mm, rec_locations[:, 1] * mm,
               marker='^', color='green', s=40, zorder=5, label='Receiver')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    ax.set_xlim(x0 * mm, x1 * mm)
    ax.set_ylim(y0 * mm, y1 * mm)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)


def extract_data(simulation, events, receiver_field='displacement', field=None):
    """
    Load the results of a finished simulation for the first event in `events`.

    Returns
    -------
    data      : receiver waveforms as an xarray
    wavefield : volume wavefield of `field` as an xarray (transposed),
                or None if `field` is not given
    """
    ed = p.waveforms.get(data_name=simulation, events=events)[0]
    data = ed.get_waveform_data_xarray(receiver_field=receiver_field)
    wavefield = None
    if field:
        wavefield = wavefield_output_to_xarray(
            ed.get_wavefield_output(output_type='volume', field=field),
            p.simulations.get_mesh(simulation).points,
        ).T
    return data, wavefield


# =============================================================================
# Project
# =============================================================================

domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)

p = sn.Project.from_domain(
    path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True
)


# =============================================================================
# True model and mesh
# =============================================================================

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

mesh = lm.mesh_from_domain(
    domain=domain,
    model=sn.layered_meshing.MeshingProtocol(m_true, ab=ab_params),
    mesh_resolution=mesh_res,
)


# =============================================================================
# Sources, receivers and events
# =============================================================================

tx_locations = np.array([[0.005, 0.009]])   # source (x, y) [m], one row per source
r_loc_all    = np.array([[0.005, 0.001]])   # receiver (x, y) [m], shared by all events

n_src = tx_locations.shape[0]
n_rxs = r_loc_all.shape[0]

receivers = [
    sn.simple_config.receiver.cartesian.Point2D(
        x=r[0], y=r[1], station_code=f"R{j+1}", fields=["displacement"]
    )
    for j, r in enumerate(r_loc_all)
]

# One event per source, each recorded by all receivers
event_ls = []
print(f"Registering {n_src} source events ...")
for i, s in enumerate(tx_locations):
    event_name = f"src_{i+1}"
    print(f"  [{i+1}/{n_src}] registering event '{event_name}' "
          f"at x={s[0]*1e3:.2f} mm, y={s[1]*1e3:.2f} mm ...")
    source = sn.simple_config.source.cartesian.VectorPoint2D(
        x=s[0], y=s[1], fx=0, fy=amplitude_ratio
    )
    p.add_to_project(
        sn.Event(event_name=event_name, sources=source, receivers=receivers)
    )
    event_ls.append(event_name)

print(f"All {len(event_ls)} events registered: {event_ls}")


# =============================================================================
# Plot true orientation field
# =============================================================================

plot_orientation_field(
    mesh, tx_locations, r_loc_all,
    save_path=Path(IMAGE_DIR_WIN) / 'true_model_orientation_field.png',
)


# =============================================================================
# Simulation configuration
# =============================================================================

waveform_config = sn.WaveformSimulationConfiguration(
    start_time_in_seconds=start_time,
    end_time_in_seconds=end_time,
    time_step_in_seconds=time_step_in_seconds,
)

stf = sn.simple_config.stf.Ricker(center_frequency=CENTRAL_FREQUENCY)

sim_config = sn.UnstructuredMeshSimulationConfiguration(
    unstructured_mesh=mesh,
    name="forward_simulation",
    event_configuration=sn.EventConfiguration(
        wavelet=stf,
        waveform_simulation_configuration=waveform_config,
    ),
)

p.add_to_project(sim_config, overwrite=True)


# =============================================================================
# Run
# =============================================================================

forward_simulation(
    simulation_name="forward_simulation",
    events=event_ls,
    fields=["displacement"],
    sampling_interval_in_time_steps=10,
)


# =============================================================================
# Sensitivity kernel w.r.t. orientation theta
# =============================================================================

STIFFNESS_PARAMETERS = {'C11', 'C12', 'C13', 'C22', 'C23', 'C33'}   # Salvus 2D Voigt names
INITIAL_SIM_NAME     = "initial_model"       # simulation in the initial model
MISFIT_NAME          = "L2_vs_true"          # L2 misfit against the true-model data

# Initial model: same mesh, homogeneous background orientation everywhere
mesh_init = mesh.copy()
for k in STIFFNESS_PARAMETERS:
    mesh_init.element_nodal_fields[k][:] = getattr(matl_background, k).p

p.add_to_project(
    sn.UnstructuredMeshSimulationConfiguration(
        unstructured_mesh=mesh_init,
        name=INITIAL_SIM_NAME,
        event_configuration=sn.EventConfiguration(
            wavelet=stf,
            waveform_simulation_configuration=waveform_config,
        ),
    ),
    overwrite=True,
)

# Observed data = synthetics of the true layered model
p.add_to_project(
    sn.MisfitConfiguration(
        name=MISFIT_NAME,
        observed_data="SYNTHETIC_DATA:forward_simulation",
        misfit_function="L2",
        receiver_field="displacement",
    ),
    overwrite=True,
)

# Forward + adjoint runs -> K_C for every stiffness parameter.
# compute_gradients is non-blocking: it launches the next missing step
# (forward, then adjoint) and returns None until the gradients exist.
t_start = time.perf_counter()
while p.actions.inversion.compute_gradients(
    simulation_configuration=INITIAL_SIM_NAME,
    misfit_configuration=MISFIT_NAME,
    events=event_ls,
    site_name=SITE_NAME,
    ranks_per_job=RANKS,
    gradient_parameters=STIFFNESS_PARAMETERS,
) is None:
    time.sleep(5)
print(f"Gradients computed in {time.perf_counter() - t_start:.1f} s wall-clock.")

gradient_C = p.actions.inversion.sum_gradients(
    simulation_configuration=INITIAL_SIM_NAME,
    misfit_configuration=MISFIT_NAME,
    events=event_ls,
    gradient_parameters=STIFFNESS_PARAMETERS,
)

# K_theta = K_C :: dC/dtheta, evaluated at the initial model's theta(x)
K_theta = orientation_kernel(gradient_C, mesh_init, matl)

plot_kernel(
    mesh_init, K_theta, tx_locations, r_loc_all,
    title=r'Sensitivity kernel $K_\theta$ (initial model)',
    label=r'$K_\theta$ [per rad]',
    save_path=Path(IMAGE_DIR_WIN) / 'kernel_theta.png',
)
