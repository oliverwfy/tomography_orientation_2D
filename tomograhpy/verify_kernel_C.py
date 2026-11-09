"""
How K_theta is built at the first FWI iteration of FWI_orientation_multilayer.py.

Chain rule used by the FWI (orientation_kernel):

    K_theta(x) = sum_ij  K_Cij(x) * dC_ij/dtheta(x)

  * K_Cij       : Salvus built-in adjoint gradient (compute_gradients + sum_gradients) of the L2
                  misfit w.r.t. the 2D Voigt stiffness C11, C12, C13, C22, C23, C33
  * dC_ij/dtheta: analytic derivative of the rotated stiffness (dC_dtheta of the FWI), at the
                  starting model's theta (homogeneous 45 deg -> uniform in space)

First iteration = first frequency stage (alpha = ALPHA) of FWI_orientation_multilayer.py:
observed data from the true 5-layer model, synthetics from the homogeneous 45 deg starting model,
same mesh, source, receivers, wavelet, time axis and L2 misfit. The kernels are shown unmasked
(the FWI additionally restricts K_theta to the Fresnel zone and the inverted region).

Figures (journal format, image/kernel_theta_multilayer/):
  kernel_theta_chain_rule : 6 x 3 panels, rows C11, C13, C15, C33, C35, C55 (x-z plane Voigt),
                            columns K_Cij | dC_ij/dtheta | product
  kernel_theta_total      : K_theta = sum of the third column

Self-contained (importing FWI_orientation_multilayer.py would run the FWI and clear its output);
uses its own Salvus project. Run from the tomograhpy folder:
    python verify_kernel_C.py
"""

import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file-only backend
import matplotlib.pyplot as plt

import salvus.mesh
import salvus.mesh.layered_meshing as lm
import salvus.namespace as sn
from salvus.material._details import material as md

from my_code.utilities import elemental_nodal_to_nodal_field
from my_code.plot_style import (apply_paper_style, save_paper_figure, add_colorbar,
                                plot_sources_receivers, PAPER_FIGSIZE, DOUBLE_COL, MM)
from material_ela_constants.Elastic_Material import Austenite


# =============================================================================
# Paths and compute site
# =============================================================================

_ON_ISAMBARD = os.environ.get('SCRATCH', '').startswith('/scratch')
if _ON_ISAMBARD:
    _BASE           = os.environ['SCRATCH'] + '/Salvus_project/elastic_model/anisotropic'
    PROJECT_DIR_WIN = _BASE + '/Project'
    IMAGE_DIR_WIN   = _BASE + '/image'
else:
    PROJECT_DIR_WIN = '/mnt/f/Salvus_project/elastic_model/anisotropic/Project'
    IMAGE_DIR_WIN   = '/mnt/f/Salvus_project/elastic_model/anisotropic/image'

PROJECT_NAME = "verify_kernel_theta_multilayer"                  # not the FWI project
FIG_DIR      = Path(IMAGE_DIR_WIN) / "kernel_theta_multilayer"   # not inside the FWI image folder
FIG_DIR.mkdir(parents=True, exist_ok=True)

SITE_NAME = "isambard_oliver" if _ON_ISAMBARD else "oliver_wsl"
RANKS     = 2


# =============================================================================
# Settings copied from FWI_orientation_multilayer.py
# =============================================================================

CENTRAL_FREQUENCY      = 2e6
ALPHA                  = 0.2                                     # first frequency stage
sampling_rate_in_hertz = CENTRAL_FREQUENCY * 250
time_step_in_seconds   = 1 / sampling_rate_in_hertz
end_time               = 6e-6                                    # alpha = 1 value, scaled by 1 / alpha
amplitude_ratio        = 1e10

x0, x1 = 0.0, 0.01
y0, y1 = 0.0, 0.01
LAYER_INTERFACES_Y   = [0.009, 0.007, 0.006, 0.0045, 0.002, 0.001]
BACKGROUND_ANGLE_DEG = 45.0
LAYER_ANGLES_DEG     = [70.0, 30.0, 60.0, 20.0, 40.0]

matl    = Austenite()
VP_0deg = np.sqrt(matl.C33 / matl.RHO)
STIFFNESS_PARAMETERS = ['C11', 'C12', 'C13', 'C22', 'C23', 'C33']   # Salvus 2D Voigt names (row order)
# labels: 3D Voigt indices of the x-z plane (1 = xx, 3 = zz, 5 = xz) for the Salvus 2D names
# (1 = xx, 2 = zz, 3 = xz), i.e. C11, C13, C15, C33, C35, C55
XZ_VOIGT = {'C11': '11', 'C12': '13', 'C13': '15', 'C22': '33', 'C23': '35', 'C33': '55'}

elements_per_wavelength = 4
model_order             = 4
reference_frequency     = CENTRAL_FREQUENCY * 2
number_of_wavelengths   = 2
free_surfaces           = ['y0', 'y1']

tx_locations = np.array([[0.005, 0.01]])
r_loc_all    = np.array([[x, 0.0] for x in np.linspace(0.001, 0.009, 9)])


def oriented_material(material_unoriented, angle_in_degrees):
    """Rotate a hexagonal material and convert it to the 2D solver form (C11..C33)."""
    orientation = sn.material.orientation.ClockwiseAngle.from_params(angle_in_degrees=angle_in_degrees)
    return md.to_solver_form(material_unoriented.with_orientation(orientation), ndim=2)


def theta_from_mesh(mesh):
    """Element-nodal theta [rad] from the stiffness (ClockwiseAngle convention), as the FWI."""
    f = mesh.element_nodal_fields
    return 0.5 * np.arctan2(2.0 * (f['C13'] + f['C23']), f['C22'] - f['C11'])


def dC_dtheta(theta, matl):
    """Derivative of the rotated 2D stiffness w.r.t. theta [Pa/rad] (dC_dtheta of the FWI)."""
    alpha = matl.C11 - 2.0 * matl.C13 + matl.C33 - 4.0 * matl.C44
    gamma = matl.C11 - matl.C33
    s2, c2 = np.sin(2.0 * theta), np.cos(2.0 * theta)
    s4, c4 = np.sin(4.0 * theta), np.cos(4.0 * theta)
    return {
        'C11': -0.5 * (alpha * s4 + 2.0 * gamma * s2),
        'C12':  0.5 * alpha * s4,
        'C13': -0.5 * (gamma * c2 + alpha * c4),
        'C22': -0.5 * (alpha * s4 - 2.0 * gamma * s2),
        'C23': -0.5 * (gamma * c2 - alpha * c4),
        'C33':  0.5 * alpha * s4,
    }


# =============================================================================
# Project, true model and starting model (as the FWI)
# =============================================================================

domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)
p = sn.Project.from_domain(path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True)

material_unoriented = sn.material.elastic.hexagonal.TensorComponents.from_params(**matl.VTI_parameters())
matl_background = oriented_material(material_unoriented, BACKGROUND_ANGLE_DEG)
_stack = [matl_background]
for _y_if, _m in zip(LAYER_INTERFACES_Y,
                     [oriented_material(material_unoriented, a) for a in LAYER_ANGLES_DEG] + [matl_background]):
    _stack += [sn.layered_meshing.interface.Hyperplane.at(_y_if), _m]

ab_params = salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
    free_surface=free_surfaces, number_of_wavelengths=number_of_wavelengths,
    reference_velocity=VP_0deg, reference_frequency=reference_frequency)
mesh_res = sn.MeshResolution(reference_frequency=reference_frequency,
                             elements_per_wavelength=elements_per_wavelength, model_order=model_order)
mesh = lm.mesh_from_domain(domain=domain,
                           model=sn.layered_meshing.MeshingProtocol(lm.LayeredModel(_stack), ab=ab_params),
                           mesh_resolution=mesh_res)
mesh_init = mesh.copy()                                         # homogeneous 45 deg starting model
for k in STIFFNESS_PARAMETERS:
    mesh_init.element_nodal_fields[k][:] = getattr(matl_background, k).p
print(f"Mesh: {mesh.nelem} elements")

receivers = [sn.simple_config.receiver.cartesian.Point2D(x=r[0], y=r[1], station_code=f"R{j + 1}",
                                                         fields=["displacement"])
             for j, r in enumerate(r_loc_all)]
event_ls = []
for i, s in enumerate(tx_locations):
    name = f"src_{i + 1}"
    ev = sn.Event(event_name=name, receivers=receivers,
                  sources=sn.simple_config.source.cartesian.VectorPoint2D(x=s[0], y=s[1], fx=0,
                                                                          fy=amplitude_ratio))
    p.events.update(ev) if p.events.exists(name) else p.add_to_project(ev)
    event_ls.append(name)


# =============================================================================
# First iteration: observed (true) vs synthetic (starting model), Salvus gradient K_C
# =============================================================================

f_c = ALPHA * CENTRAL_FREQUENCY
start_time = -np.ceil(1.5 / f_c / time_step_in_seconds) * time_step_in_seconds   # as the FWI stage
end_time_stage = np.round(end_time / ALPHA / time_step_in_seconds) * time_step_in_seconds
print(f"Stage alpha={ALPHA}: f_c = {f_c / 1e6:.2f} MHz, t = [{start_time * 1e6:.2f}, "
      f"{end_time_stage * 1e6:.2f}] us, dt = {time_step_in_seconds * 1e9:.1f} ns")
event_config = sn.EventConfiguration(
    wavelet=sn.simple_config.stf.Ricker(center_frequency=f_c),
    waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
        start_time_in_seconds=start_time, end_time_in_seconds=end_time_stage,
        time_step_in_seconds=time_step_in_seconds))

OBS_SIM, INIT_SIM, MISFIT = "true_model_first_stage", "initial_model_first_stage", "L2_first_stage"
for name, m in [(OBS_SIM, mesh), (INIT_SIM, mesh_init)]:
    p.add_to_project(sn.UnstructuredMeshSimulationConfiguration(
        unstructured_mesh=m, name=name, event_configuration=event_config), overwrite=True)
p.add_to_project(sn.MisfitConfiguration(name=MISFIT, observed_data=f"SYNTHETIC_DATA:{OBS_SIM}",
                                        misfit_function="L2", receiver_field="displacement"),
                 overwrite=True)


def wait_for(call, label, poll_seconds=5, max_errors=60):
    """Repeat a non-blocking Salvus action until it returns a result (as _wait_for_result of the FWI)."""
    n_errors = 0
    while True:
        try:
            result = call(); n_errors = 0
        except (AssertionError, ValueError) as e:
            n_errors += 1
            if n_errors > max_errors:
                raise
            print(f"  [{label}] retry {n_errors}/{max_errors} after {type(e).__name__}: {e}")
            result = None
        if result is not None:
            return result
        time.sleep(poll_seconds)


t0 = time.perf_counter()
p.simulations.launch(ranks_per_job=RANKS, site_name=SITE_NAME, events=event_ls,
                     simulation_configuration=OBS_SIM, delete_conflicting_previous_results=True)
p.simulations.query(simulation_configuration=OBS_SIM, events=event_ls, block=True, verbosity=-1)
wait_for(lambda: p.actions.inversion.compute_gradients(
    simulation_configuration=INIT_SIM, misfit_configuration=MISFIT, events=event_ls,
    site_name=SITE_NAME, ranks_per_job=RANKS, gradient_parameters=set(STIFFNESS_PARAMETERS)),
    "compute_gradients")
misfit = wait_for(lambda: p.actions.inversion.compute_misfits(
    simulation_configuration=INIT_SIM, misfit_configuration=MISFIT, events=event_ls,
    site_name=SITE_NAME, ranks_per_job=RANKS), "compute_misfits")
gradient_C = p.actions.inversion.sum_gradients(
    simulation_configuration=INIT_SIM, misfit_configuration=MISFIT, events=event_ls,
    gradient_parameters=set(STIFFNESS_PARAMETERS))
print(f"Forward + adjoint done in {time.perf_counter() - t0:.0f} s, misfit = {sum(misfit.values()):.4e}")


# =============================================================================
# Chain rule: K_Cij, dC_ij/dtheta and their product (nodal fields)
# =============================================================================

theta_init = theta_from_mesh(mesh_init)                           # element-nodal, 45 deg everywhere
dC = dC_dtheta(theta_init, matl)
# exact zeros (dC12, dC33 ~ sin 4 theta = 0 at 45 deg) are ~1e-6 Pa/rad of round-off: snap them to 0
_dC_max = max(np.abs(v).max() for v in dC.values())
dC = {k: np.where(np.abs(v) < 1e-9 * _dC_max, 0.0, v) for k, v in dC.items()}
conn = mesh_init.connectivity
to_nodal = lambda f: elemental_nodal_to_nodal_field(f, conn)
K_C   = {k: to_nodal(gradient_C.element_nodal_fields[k]) for k in STIFFNESS_PARAMETERS}
dC_n  = {k: to_nodal(dC[k]) for k in STIFFNESS_PARAMETERS}
prod  = {k: to_nodal(gradient_C.element_nodal_fields[k] * dC[k]) for k in STIFFNESS_PARAMETERS}
K_theta = sum(prod.values())
print("dC_ij/dtheta at theta = 45 deg [GPa/rad] (x-z plane Voigt): " +
      ", ".join(f"C{XZ_VOIGT[k]} {dC[k].mean() / 1e9:+.1f}" for k in STIFFNESS_PARAMETERS))

xs, ys = mesh_init.points[:, 0], mesh_init.points[:, 1]
inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)


def sym_limit(f):
    """Symmetric colour limit: 99th percentile of |f| in the physical domain (source/receiver spikes clipped)."""
    v = np.percentile(np.abs(f[inside]), 99.0)
    return v if v > 0 else np.abs(f[inside]).max()             # 0 for an all-zero field


def field_axes(ax, f, label, scale=1.0, vmax=None, legend=False):
    """Signed field (RdBu, symmetric limit `vmax` in plotted units, default 99th percentile)."""
    vmax = ((sym_limit(f) if vmax is None else vmax * scale) / scale) or 1.0   # all-zero field: any scale
    tp = ax.tripcolor(xs * MM, ys * MM, f / scale, cmap='RdBu', vmin=-vmax, vmax=vmax,
                      shading='gouraud', rasterized=True)
    cb = add_colorbar(ax.figure, ax, tp, label)
    cb.formatter.set_powerlimits((-2, 2)); cb.update_ticks()
    plot_sources_receivers(ax, tx_locations, r_loc_all, legend=legend)
    for y_if in LAYER_INTERFACES_Y:   # true interfaces (unknown to the starting model)
        ax.axhline(y_if * MM, color='0.4', lw=0.3, ls=':')
    ax.set_xlim(x0 * MM, x1 * MM); ax.set_ylim(y0 * MM, y1 * MM); ax.set_aspect('equal')


# =============================================================================
# Figures
# =============================================================================

apply_paper_style()
# shared colour scales: column 2 (all dC/dtheta, GPa/rad) and column 3 (all products, so the
# terms that drive K_theta stand out); column 1 per row (the K_C magnitudes differ)
VMAX_DC   = _dC_max / 1e9
VMAX_PROD = max(sym_limit(prod[k]) for k in STIFFNESS_PARAMETERS)
fig, axes = plt.subplots(len(STIFFNESS_PARAMETERS), 3, figsize=(DOUBLE_COL, 1.55 * len(STIFFNESS_PARAMETERS)),
                         sharex=True, sharey=True, layout='constrained')
for i, k in enumerate(STIFFNESS_PARAMETERS):
    kk = XZ_VOIGT[k]                                             # x-z plane label (11, 13, 15, 33, 35, 55)
    panels = [
        (K_C[k], rf'$K_{{C_{{{kk}}}}}$', rf'$K_{{C_{{{kk}}}}}$', 1.0, None),
        (dC_n[k], rf'$\partial C_{{{kk}}}/\partial\theta$ = {dC[k].mean() / 1e9:.1f} GPa/rad',
         r'GPa/rad', 1e9, VMAX_DC),
        (prod[k], rf'$K_{{C_{{{kk}}}}}\,\partial C_{{{kk}}}/\partial\theta$',
         rf'$K_{{C_{{{kk}}}}}\,\partial C_{{{kk}}}/\partial\theta$', 1.0, VMAX_PROD),
    ]
    for j, (f, title, label, scale, vmax) in enumerate(panels):
        ax = axes[i, j]
        field_axes(ax, f, label, scale, vmax)
        # letters run down the columns: (a)-(f) K_C, (g)-(l) dC/dtheta, (m)-(r) products
        ax.set_title(f'({chr(97 + len(STIFFNESS_PARAMETERS) * j + i)}) {title}', loc='left')
        if i == len(STIFFNESS_PARAMETERS) - 1:
            ax.set_xlabel('$x$ (mm)')
        if j == 0:
            ax.set_ylabel('$z$ (mm)')
save_paper_figure(fig, FIG_DIR / 'kernel_theta_chain_rule')
plt.close(fig)

fig, ax = plt.subplots(figsize=PAPER_FIGSIZE, layout='constrained')
field_axes(ax, K_theta, r'$K_\theta = \sum_{ij} K_{C_{ij}}\,\partial C_{ij}/\partial\theta$',
           legend=True)
ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$z$ (mm)')
save_paper_figure(fig, FIG_DIR / 'kernel_theta_total')
plt.close(fig)

# consistency with the FWI: K_theta_FWI = K_C :: dC/dtheta element-nodally (orientation_kernel)
_K_fwi = to_nodal(sum(gradient_C.element_nodal_fields[k] * dC[k] for k in STIFFNESS_PARAMETERS))
print(f"max |K_theta - K_theta_FWI| / max |K_theta| = "
      f"{np.abs(K_theta - _K_fwi).max() / np.abs(K_theta).max():.1e}")
