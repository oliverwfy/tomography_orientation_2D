import copy
import os
from datetime import datetime
_SCRIPT_START_TIME = datetime.now()
import salvus.namespace as sn
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend: we only save figures, never
                        # display them, and this avoids GUI-backend init hangs
                        # (e.g. under WSL with no working display connection)
import matplotlib.pyplot as plt

import sys
import importlib
from pathlib import Path
for path_to_add in (Path.cwd(), Path.cwd() / "tomography"):
    if (path_to_add / "my_code").is_dir() and str(path_to_add) not in sys.path:
        sys.path.append(str(path_to_add))

import my_code.utilities
importlib.reload(my_code.utilities)
from my_code.utilities import *
import salvus

import salvus.mesh.layered_meshing as lm
from scipy.ndimage import gaussian_filter1d
from datetime import datetime   # re-imported: guards against shadowing by the wildcard import below
import salvus.flow.simple_config as sc
from salvus.toolbox.helpers.wavefield_output import (
    WavefieldOutput,
    wavefield_output_to_xarray,
)
import os
import xarray as xr

# ── Configuration, project and mesh ───────────────────────────────────────

# ---- Salvus site --------------------------------------------------------
SITE_NAME = "oliver_wsl"
# SITE_NAME = "isambard_oliver"

density_rxs = 15

# # Directories in Windows
PROJECT_DIR_WIN = fr'/mnt/f/Salvus_project/multilayer_{density_rxs}/Project'
DATA_DIR_WIN = fr'/mnt/f/Salvus_project/multilayer_{density_rxs}/data/data'
IMAGE_DIR_WIN = fr'/mnt/f/Salvus_project/multilayer_{density_rxs}/image'
# PROJECT_DIR_WIN = '/scratch/b6as/oliverwfy.b6as/FWI_multilayers_full_aperture/Project'
# DATA_DIR_WIN    = '/scratch/b6as/oliverwfy.b6as/FWI_multilayers_full_aperture/data'
# IMAGE_DIR_WIN   = '/scratch/b6as/oliverwfy.b6as/FWI_multilayers_full_aperture/image'


RANKS = 8

# ---- Physical parameters ------------------------------------------------
VP                = 1400.0
RHO               = 1025.0
THICKNESS           = 20   # m

# ---- Multi-frequency (multiscale) continuation strategy ------------------
# Invert from the lowest to the highest thickness/wavelength ratio, each
# stage warm-started from the previous stage's recovered v(y) model and
# capped at MAX_ITER_PER_STAGE accepted L-BFGS iterations.
alpha_ls           = [0.4, 0.6, 0.8, 1.0]   # thickness-to-wavelength ratio, low -> high frequency
CENTRAL_FREQUENCY  = 75   # Hz — fixed reference used for mesh resolution / sampling rate
MAX_ITER_PER_STAGE = 20   # max accepted L-BFGS iterations per frequency stage
GAUSSIAN_SIGMA_Y   = 1   # depth-levels — smoothing applied to v(y) after each accepted iteration


# ---- Domain -------------------------------------------------------------
x0, x1 = 0, 20
y0, y1 = 0, 100

# ---- Project ------------------------------------------------------------
Path(PROJECT_DIR_WIN).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR_WIN).mkdir(parents=True, exist_ok=True)
Path(IMAGE_DIR_WIN).mkdir(parents=True, exist_ok=True)

PROJECT_NAME = f'FWI_multilayers'
domain = sn.domain.dim2.BoxDomain(x0=x0, x1=x1, y0=y0, y1=y1)

print(f"Loading/creating Salvus project '{PROJECT_NAME}' at {PROJECT_DIR_WIN} ...")
p = sn.Project.from_domain(
    path=Path(PROJECT_DIR_WIN, PROJECT_NAME), domain=domain, load_if_exists=True
)
print("Project ready.")

# ---- Mesh resolution ----------------------------------------------------
elements_per_wavelength = 3
model_order             = 4
reference_velocity      = VP
number_of_wavelengths   = 2
reference_frequency     = CENTRAL_FREQUENCY * 2
free_surfaces           =  ['y0']

# ---- True (layered) model -----------------------------------------------
m_true = sn.layered_meshing.LayeredModel([
    sn.material.from_params(rho=RHO, vp=1400),
    sn.layered_meshing.interface.Hyperplane.at(90.0),
    sn.material.from_params(rho=RHO, vp=1440),
    sn.layered_meshing.interface.Hyperplane.at(70.0),
    sn.material.from_params(rho=RHO, vp=1420),
    sn.layered_meshing.interface.Hyperplane.at(60.0),
    sn.material.from_params(rho=RHO, vp=1480),
    sn.layered_meshing.interface.Hyperplane.at(30.0),
    sn.material.from_params(rho=RHO, vp=1460),
    sn.layered_meshing.interface.Hyperplane.at(15.0),
    sn.material.from_params(rho=RHO, vp=1500),
])


# ---- Initial (starting) model — two-layer ---------------------------
m_homo = sn.layered_meshing.LayeredModel([
    sn.material.from_params(rho=RHO, vp=1400),
    sn.layered_meshing.interface.Hyperplane.at(10.0),
    sn.material.from_params(rho=RHO, vp=1500),
])

ab_params = salvus.mesh.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
    free_surface=free_surfaces,
    number_of_wavelengths=number_of_wavelengths,
    reference_velocity=reference_velocity,
    reference_frequency=reference_frequency,
)

mesh_res = sn.MeshResolution(
    reference_frequency=reference_frequency,
    elements_per_wavelength=elements_per_wavelength,
    model_order=model_order,
)

print("Building homogeneous-model mesh ...")
mesh_homo = lm.mesh_from_domain(
    domain=domain,
    model=sn.layered_meshing.MeshingProtocol(m_homo, ab=ab_params),
    mesh_resolution=mesh_res,
)
print("Homogeneous-model mesh built.")

print("Building true (layered) model mesh ...")
mesh = lm.mesh_from_domain(
    domain=domain,
    model=sn.layered_meshing.MeshingProtocol(m_true, ab=ab_params),
    mesh_resolution=mesh_res,
)
print("True (layered) model mesh built.")



# source and receivers configuration
amplitude_ratio = 1e3


n_src = 1
s_loc = [(5, 95)]

if density_rxs <=16:
    r_loc  = [(x, y0) for x in np.linspace(2, 18, int((18-2)/density_rxs)+1)]
else:
    r_loc  = [(2, y0), (18, y0)]
r_loc_left = [(x0, y) for y in np.linspace(10, 90, int((90-10)/density_rxs)+1)]
r_loc_right = [(x1, y) for y in np.linspace(10, 90, int((90-10)/density_rxs)+1)]




# Every event gets all three receiver groups (top + left + right walls of
# the ROI) — one combined receiver list shared by every source.
r_loc_all = r_loc + r_loc_left + r_loc_right

receivers = [
    sn.simple_config.receiver.cartesian.Point2D(
        x=r[0], y=r[1], station_code=f"R{j+1}", fields=["phi"]
    )
    for j, r in enumerate(r_loc_all)
]



# # One event per source: each event has a single source and all receivers
# # on the opposite side of the ROI.
event_ls = []
print(f"Registering {n_src} source events ...")
for i, s in enumerate(s_loc):
    event_name = f"src_{i+1}"
    print(f"  [{i+1}/{n_src}] registering event '{event_name}' at x={s[0]:.1f}, y={s[1]:.1f} ...")
    source = sn.simple_config.source.cartesian.ScalarPoint2D(
        x=s[0], y=s[1], f=amplitude_ratio
    )
    p.add_to_project(
        sn.Event(event_name=event_name, sources=source, receivers=receivers)
    )
    event_ls.append(event_name)


print(f"All {len(event_ls)} events registered: {event_ls}")

# Full list of acquired (source, receiver) coordinate pairs — every event
# uses the same receiver list (r_loc_all), so this is the full Cartesian
# product of sources x receivers. Used for the Fresnel-zone ellipse mask
# and diagnostic-plot overlay.
sr_pairs = [(s, r) for s in s_loc for r in r_loc_all]

vp_nodal = elemental_nodal_to_nodal_field(mesh.element_nodal_fields['VP'], mesh.connectivity)

fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
tp = ax.tripcolor(mesh.points[:, 0], mesh.points[:, 1], vp_nodal, cmap='viridis', shading='gouraud')
plt.colorbar(tp, ax=ax, label='VP [m/s]')

ax.scatter([s[0] for s in s_loc ], [s[1] for s in s_loc ],
           marker='v', color='red', s=60, zorder=5, label='Source')
ax.scatter([r[0] for r in r_loc_all], [r[1] for r in r_loc_all],
           marker='^', color='green', s=40, zorder=5, label='Receiver')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('True (layered) model — VP')
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(Path(IMAGE_DIR_WIN) / 'true_model_vp_field.png')
plt.close(fig)

# ── Simulation configuration constants (frequency-independent) ────────────
# Only the wavelet (centre frequency + time shift) changes between
# frequency stages; the sampling/time-stepping configuration is shared.

end_time               = 2e-1
start_time             = 0
sampling_rate_in_hertz = CENTRAL_FREQUENCY * 200

waveform_config = sn.WaveformSimulationConfiguration(
    start_time_in_seconds=start_time,
    end_time_in_seconds=end_time,
    time_step_in_seconds=1 / sampling_rate_in_hertz,
)

# ── Helper functions ──────────────────────────────────────────────────────

def forward_simulation(simulation_name, events,
                       fields=None, sampling_interval_in_time_steps=10):
    _n_events = 1 if isinstance(events, str) else len(events)
    print(f"  [forward_simulation] launching '{simulation_name}' "
          f"for {_n_events} event(s), volume_output={'yes' if fields else 'no'} ...")
    if fields:
        p.simulations.launch(
            ranks_per_job=RANKS,
            site_name=SITE_NAME,
            events=events,
            simulation_configuration=simulation_name,
            extra_output_configuration={
                "volume_data": {
                    "sampling_interval_in_time_steps": sampling_interval_in_time_steps,
                    "fields": fields,
                },
            },
            delete_conflicting_previous_results=True,
        )
    else:
        p.simulations.launch(
            ranks_per_job=RANKS,
            site_name=SITE_NAME,
            events=events,
            simulation_configuration=simulation_name,
            delete_conflicting_previous_results=True,
        )

    p.simulations.query(block=True, verbosity=-1)
    print(f"  [forward_simulation] '{simulation_name}' finished.")



def extract_data(simulation, events, receiver_field='phi', field=None):
    ed = p.waveforms.get(data_name=simulation, events=events)[0]
    data = ed.get_waveform_data_xarray(receiver_field=receiver_field)
    wavefield = None
    if field:
        wavefield = wavefield_output_to_xarray(
            ed.get_wavefield_output(output_type='volume', field=field),
            p.simulations.get_mesh(simulation).points,
        ).T
    return data, wavefield


def misfit_adjoint_source(u_obs, u, event=False, iteration=0):
    sampling_rate = u_obs.sampling_rate_in_hertz
    start_time    = u_obs.time[0]
    end_time      = u_obs.time[-1]

    diff   = u_obs - u
    misfit = 0.5 / sampling_rate * (diff ** 2).sum()
    f_adj  = np.flip(diff, axis=1) / sampling_rate

    event_adj        = None
    event_config_adj = None

    if event:
        rxs, srcs    = p.events.get(event).receivers, p.events.get(event).sources
        rxs_adj_loc  = [r.location for r in srcs]
        src_adj_loc  = [t.location for t in rxs]

        stf_adj_ls = [
            sc.stf.Custom.from_array(
                np.array(f_adj[i]),
                sampling_rate_in_hertz=sampling_rate_in_hertz,
                start_time_in_seconds=0.0,
            )
            for i in range(len(src_adj_loc))
        ]

        srcs_adj = [
            sn.simple_config.source.cartesian.ScalarPoint2D(x=tx[0], y=tx[1], f=1)
            for tx in src_adj_loc
        ]
        rxs_adj = [
            sn.simple_config.receiver.cartesian.Point2D(
                x=rx[0], y=rx[1],
                fields=['phi'],
                station_code=f"ite:{iteration}_adj_{_i:06d}",
            )
            for _i, rx in enumerate(rxs_adj_loc)
        ]

        event_adj = f'{event}_adjoint_{iteration}'
        print(f"    [misfit_adjoint_source] registering adjoint event '{event_adj}' "
              f"(misfit = {float(misfit):.6e}) ...")
        p.add_to_project(sn.Event(event_name=event_adj, sources=srcs_adj, receivers=rxs_adj))

        event_config_adj = sn.EventConfiguration(
            wavelet=stf_adj_ls,
            waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
                start_time_in_seconds=start_time,
                end_time_in_seconds=end_time,
                time_step_in_seconds=1 / sampling_rate_in_hertz,
            ),
        )

    return misfit, f_adj, event_adj, event_config_adj


def adjoint_wavefield(p, event_adj, event_config_adj,
                      model_simulation_name='forward_simulation_homogeneous_model',
                      field='phi_t', iteration=0,
                      sampling_interval_in_time_steps=10):
    simulation_adj  = f"adjoint_{iteration}"
    mesh_current    = p.simulations.get_mesh(model_simulation_name)

    p.add_to_project(
        sn.UnstructuredMeshSimulationConfiguration(
            unstructured_mesh=mesh_current,
            name=simulation_adj,
            event_configuration=event_config_adj,
        ),
        overwrite=True,
    )

    print(f"    [adjoint_wavefield] launching '{simulation_adj}' for '{event_adj}' ...")
    p.simulations.launch(
        ranks_per_job=RANKS,
        site_name=SITE_NAME,
        events=event_adj,
        simulation_configuration=simulation_adj,
        extra_output_configuration={
            "volume_data": {
                "sampling_interval_in_time_steps": sampling_interval_in_time_steps,
                "fields": [field],
            },
        },
        delete_conflicting_previous_results=True,
    )
    p.simulations.query(block=True, verbosity=1)
    print(f"    [adjoint_wavefield] '{simulation_adj}' finished.")


    _, wavefield_adj = extract_data(
        simulation=simulation_adj,
        events=event_adj,
        receiver_field='phi',
        field=field,
    )
    return wavefield_adj


def compute_gradient(forward_wavefield, adjoint_wavefield):
    """
    Sensitivity kernel K_A = ∂J/∂A  (zero-lag cross-correlation):

        K_A(x) = Σ_t φ†_t(x, T-t) · φ_t(x, t) · Δt

    φ† is time-reversed before the inner product because the adjoint
    simulation runs forward in time (source = time-reversed residual).

    Parameters
    ----------
    forward_wavefield : xarray   φ_t  shape (nodes, T)
    adjoint_wavefield : xarray   φ†_t    shape (nodes, T_adj)

    Returns
    -------
    K_A : np.ndarray, shape (n_nodes,)   ∂J/∂A
    """
    fwd = forward_wavefield.values.squeeze()    # (nodes, T)
    adj = adjoint_wavefield.values.squeeze()    # (nodes, T_adj)

    t_coord = forward_wavefield.coords.get('t', forward_wavefield.coords.get('time'))
    t_fwd   = t_coord.values
    dt      = float(t_fwd[1] - t_fwd[0])

    # Interpolate adjoint onto forward time axis if sampling strides differ
    if adj.shape[-1] != fwd.shape[-1]:
        from scipy.interpolate import interp1d
        t_adj = adjoint_wavefield.coords.get('t', adjoint_wavefield.coords.get('time')).values
        adj   = interp1d(t_adj, adj, axis=-1, kind='linear',
                         bounds_error=False, fill_value=0.0)(t_fwd)


    # Time-reverse φ† → φ†(T-t), then dot with φ_tt(t)
    adj_flipped = np.flip(adj, axis=-1)                          # φ†(x, T-t)
    return np.einsum('nt,nt->n', adj_flipped, fwd) * dt          # (n_nodes,)

# ── Diagnostics helpers (ellipse overlay, plot_fwi_diagnostics) ───────────

# ── Shared setup for diagnostics / sensitivity-region overlays ───────────
ROI = (2, 18, 10, 90)   # (x0, x1, y0, y1) [m] — inversion region of interest

# Grid the ellipse-union contour is drawn on (spans the full domain).
_xg, _yg = np.linspace(x0, x1, 80), np.linspace(y0, y1, 400)
_Xg, _Yg = np.meshgrid(_xg, _yg)


def normalize(arr):
    """Normalise to [-1, 1]."""
    m = np.nanmax(np.abs(arr))
    return arr / m if m > 0 else arr


def _add_ellipse_overlay(ax, pairs, margin, label=True):
    """Draw the UNION of all acquired source-receiver ellipses (one
    boundary, not one contour per pair) and scatter source/receiver
    markers. margin = VP * width_3dB; the per-pair semi-major axis
    a(s, r) = (dist(s, r) + margin) / 2 is computed for each acquired
    (s, r) pair individually so distant pairs aren't silently excluded."""
    from matplotlib.lines import Line2D
    _union_mask = np.zeros_like(_Xg, dtype=bool)
    for s, r in pairs:
        _d_sr = np.sqrt((s[0] - r[0])**2 + (s[1] - r[1])**2)
        _a    = (_d_sr + margin) / 2
        _d1 = np.sqrt((_Xg - s[0])**2 + (_Yg - s[1])**2)
        _d2 = np.sqrt((_Xg - r[0])**2 + (_Yg - r[1])**2)
        _union_mask |= (_d1 + _d2) <= 2 * _a
    # Fold onto its own mirror image about x=50 (_xg is linspace(x0, x1, ...),
    # symmetric about its own midpoint, so a column flip is an exact mirror),
    # consistently with the gradient's mirror symmetrization.
    _union_mask = _union_mask | _union_mask[:, ::-1]
    ax.contour(_Xg, _Yg, _union_mask.astype(float), levels=[0.5],
               colors='white', linewidths=1.5, linestyles='dashed', alpha=0.85)
    if label:
        s_locs = list(dict.fromkeys(s for s, _ in pairs))
        r_locs = list(dict.fromkeys(r for _, r in pairs))
        ax.scatter([s[0] for s in s_locs], [s[1] for s in s_locs],
                   color='red',  s=60, zorder=5, label='Source')
        ax.scatter([r[0] for r in r_locs], [r[1] for r in r_locs],
                   color='lime', s=60, zorder=5, label='Receiver')
        ellipse_proxy = Line2D([0], [0], color='white', linewidth=1.5,
                               linestyle='dashed', label='Sensitivity region')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [ellipse_proxy], labels=labels + ['Sensitivity region'],
                  fontsize=8, loc='upper right',
                  facecolor='dimgray', labelcolor='white', edgecolor='white')


def plot_fwi_diagnostics(it, phi_obs, phi_syn, f_adj, phi_t, phi_t_adj,
                          K_VP, K_roi, x, y, x_roi, y_roi,
                          x0_roi, x1_roi, y0_roi, y1_roi,
                          sr_pairs, margin, save=True):
    """Plot 4 FWI diagnostic figures for iteration `it` and optionally save them.

    `sr_pairs` is the FULL list of acquired (source, receiver) pairs (used
    for the sensitivity-region ellipse union, which covers every acquired
    pair) — the waveform panels (phi_obs, phi_syn, f_adj, phi_t, phi_t_adj)
    are for whichever single representative source the caller chooses to
    visualise.
    """
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec

    time_wf = phi_obs.time.values

    # ── Figure 1: waveforms + adjoint source ─────────────────────────────
    fig1, axs = plt.subplots(1, 2, figsize=(14, 4), dpi=100)
    axs[0].plot(time_wf * 1e3, np.array(phi_obs[0]), color='red',   lw=1.5, label=r'$\phi_\mathrm{obs}$')
    axs[0].plot(time_wf * 1e3, np.array(phi_syn[0]), color='green', lw=1.5, label=rf'$\phi^{{({it})}}$')
    axs[0].legend(fontsize=11)
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel(r'$\phi$')
    axs[0].set_title(f'Observed vs synthetic  (iter {it})')
    axs[0].grid(alpha=0.35)
    axs[1].plot(time_wf * 1e3, np.array(-f_adj[0])*sampling_rate_in_hertz, color='gray', lw=1.5)
    axs[1].axhline(0, color='k', lw=0.5, ls='--')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel(r'$f^\dagger$')
    axs[1].set_title(r'Adjoint source $f^\dagger$ (time-reversed residual)')
    axs[1].grid(alpha=0.35)
    plt.tight_layout()
    if save:
        plt.savefig(Path(IMAGE_DIR_WIN) / f'iter_{it:03d}_adjoint_source_alpha_{alpha}.png')
    plt.close(fig1)

    # ── Figure 2: wavefield snapshots ────────────────────────────────────
    fwd = phi_t.values.squeeze()
    adj = phi_t_adj.values.squeeze()
    t_fwd = phi_t.coords.get('t', phi_t.coords.get('time')).values
    t_adj = phi_t_adj.coords.get('t', phi_t_adj.coords.get('time')).values
    if adj.shape[-1] != fwd.shape[-1]:
        from scipy.interpolate import interp1d
        adj = interp1d(t_adj, adj, axis=-1, kind='linear',
                       bounds_error=False, fill_value=0.0)(t_fwd)
    adj_flipped = np.flip(adj, axis=-1)
    T = fwd.shape[-1]
    t_indices = [T // 4, T // 2, 3 * T // 4]

    fig2 = plt.figure(figsize=(17, 15), dpi=100)
    gs = GridSpec(3, 4, figure=fig2, width_ratios=[1, 1, 1, 0.04],
                  hspace=0.45, wspace=0.35)
    axes2 = [[fig2.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]
    cax   = fig2.add_subplot(gs[1, 3])
    col_titles = [r'$\phi_{tt}(x,\,t)$  [normalised]',
                  r'$\phi^\dagger(x,\,T{-}t)$  [normalised]',
                  r'$\phi_{tt}\;\cdot\;\phi^\dagger$  [normalised]']
    for col, title in enumerate(col_titles):
        axes2[0][col].set_title(title, fontsize=12, pad=8)
    tp_last = None
    for row, ti in enumerate(t_indices):
        t_ms = t_fwd[ti] * 1e3
        s_fwd  = normalize(fwd[:, ti])
        s_adj  = normalize(adj_flipped[:, ti])
        s_prod = normalize(s_fwd * s_adj)
        for col, (snap, lbl) in enumerate([(s_fwd,  r'$\phi_{t} (t)$'),
                                            (s_adj,  r'$\phi^\dagger_{t} (T-t)$'),
                                            (s_prod, r'$\phi_{t} (t) * \phi^\dagger_{t} (T-t)$')]):
            ax = axes2[row][col]
            tp = ax.tripcolor(x, y, snap, cmap='RdBu', shading='gouraud', vmin=-1, vmax=1)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_title(f'{lbl}  at  t = {t_ms:.2f} ms', fontsize=10)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.grid(alpha=0.15)
            if row == 1 and col == 2:
                tp_last = tp
    fig2.colorbar(tp_last, cax=cax, label='normalised amplitude')
    if save:
        plt.savefig(Path(IMAGE_DIR_WIN) / f'iter_{it:03d}_sensitivity_kernel_intermediate_alpha_{alpha}.png')
    plt.close(fig2)

    # ── Figure 3: full sensitivity kernel ────────────────────────────────
    fig3, ax_K = plt.subplots(figsize=(8, 6), dpi=100)
    _K_VP_max = np.max(np.abs(K_VP))
    tp = ax_K.tripcolor(x, y, K_VP, cmap='RdBu', shading='gouraud',
                        vmin=-_K_VP_max, vmax=_K_VP_max)
    plt.colorbar(tp, ax=ax_K, label=r'$\partial J/\partial \ln V_P$')
    _add_ellipse_overlay(ax_K, sr_pairs, margin)
    ax_K.set_xlabel('X (m)'); ax_K.set_ylabel('Y (m)')
    ax_K.set_title(rf'$K_{{VP}}$  —  full domain  (iter {it})', fontsize=12)
    ax_K.set_xlim(x0, x1); ax_K.set_ylim(y0, y1); ax_K.set_aspect('equal'); ax_K.grid(alpha=0.15)
    plt.tight_layout()
    if save:
        plt.savefig(Path(IMAGE_DIR_WIN) / f'iter_{it:03d}_sensitivity_kernel_full_alpha_{alpha}.png')
    plt.close(fig3)

    # ── Figure 4: ROI sensitivity kernel with ellipse mask ───────────────
    # Union over every acquired (source, receiver) pair, per-pair
    # semi-major axis (matches objective()'s multi-source _ellipse_roi mask).
    _ellipse_mask_roi = np.zeros_like(x_roi, dtype=bool)
    for _s, _r in sr_pairs:
        _d1   = np.sqrt((x_roi - _s[0])**2 + (y_roi - _s[1])**2)
        _d_sr = np.sqrt((_s[0] - _r[0])**2 + (_s[1] - _r[1])**2)
        _a    = (_d_sr + margin) / 2
        _d2   = np.sqrt((x_roi - _r[0])**2 + (y_roi - _r[1])**2)
        _ellipse_mask_roi |= (_d1 + _d2) <= 2 * _a
    # Fold onto its own mirror image about x=50, consistently with the
    # gradient's mirror symmetrization (see objective()'s _ellipse_roi).
    _ellipse_mask_roi = _ellipse_mask_roi | _mirror_x_roi(_ellipse_mask_roi)
    K_roi_masked = np.where(_ellipse_mask_roi, K_roi, 0.0)
    _K_roi_max = np.max(np.abs(K_roi_masked))
    fig4, ax_2d = plt.subplots(figsize=(8, 6), dpi=100)
    tp = ax_2d.tripcolor(x_roi, y_roi, K_roi_masked,
                         cmap='RdBu', shading='gouraud', vmin=-_K_roi_max, vmax=_K_roi_max)
    plt.colorbar(tp, ax=ax_2d, label=r'$\partial J/\partial \ln V_P$')
    _add_ellipse_overlay(ax_2d, sr_pairs, margin, label=False)
    _ellipse_proxy = Line2D([0], [0], color='white', linewidth=1.5,
                            linestyle='dashed', label='Sensitivity region')
    ax_2d.legend(handles=[_ellipse_proxy], fontsize=8, loc='upper right',
                 facecolor='dimgray', labelcolor='white', edgecolor='white')
    ax_2d.set_xlabel('X (m)'); ax_2d.set_ylabel('Y (m)')
    ax_2d.set_title(
        f'$K_{{VP}}$  in ROI  [{x0_roi}, {x1_roi}] × [{y0_roi}, {y1_roi}]  (iter {it})',
        fontsize=12,
    )
    ax_2d.set_xlim(x0_roi, x1_roi); ax_2d.set_ylim(y0_roi, y1_roi)
    ax_2d.set_aspect('equal'); ax_2d.grid(alpha=0.15)
    plt.tight_layout()
    if save:
        plt.savefig(Path(IMAGE_DIR_WIN) / f'iter_{it:03d}_sensitivity_kernel_roi_alpha_{alpha}.png')
    plt.close(fig4)

# ── Multi-source FWI: geometry-only setup (frequency independent) ─────────

VP_MIN, VP_MAX    = VP * 0.5, VP * 2.0   # VP bounds [m/s]
SAMPLING_INTERVAL = 10

DIAG_EVENT = event_ls[len(event_ls) // 2]   # representative source for diagnostics / width_3dB measurement
print(f"DIAG_EVENT = {DIAG_EVENT!r}")

# ---- Single (tx, rx) pair whose observed/synthetic trace is saved every
# iteration — tx indexes event_ls / s_loc (source), rx indexes r_loc_all
# (receiver, order: top wall, then left wall, then right wall).
DIAG_TX_IDX     = len(event_ls) // 2
DIAG_RX_IDX     = 0
DIAG_PAIR_EVENT = event_ls[DIAG_TX_IDX]
DIAG_RX_STATION = f"R{DIAG_RX_IDX + 1}"
print(f"DIAG pair for per-iteration trace saving: tx={DIAG_PAIR_EVENT!r} "
      f"(src #{DIAG_TX_IDX}, loc={s_loc[DIAG_TX_IDX]}), "
      f"rx={DIAG_RX_STATION!r} (idx {DIAG_RX_IDX}, loc={r_loc_all[DIAG_RX_IDX]})")

# base_mesh is just the homogeneous starting-model mesh geometry + VP field
# — it doesn't depend on frequency, so it's computed once for all stages.
base_mesh = copy.deepcopy(mesh_homo)

# nodal ROI mask — one entry per global node
_x, _y  = base_mesh.points[:, 0], base_mesh.points[:, 1]
x0_roi, x1_roi, y0_roi, y1_roi = ROI
roi_mask = (_x >= x0_roi) & (_x <= x1_roi) & (_y >= y0_roi) & (_y <= y1_roi)
_x_roi, _y_roi = _x[roi_mask], _y[roi_mask]

# Pre-compute background nodal VP once (avoids repeated averaging)
_vp_bg_nodal = elemental_nodal_to_nodal_field(
    base_mesh.element_nodal_fields['VP'], base_mesh.connectivity
).copy()

# Pre-compute rounded y-coordinates for fast per-level grouping
_y_round_full = np.round(_y, decimals=8)
_y_round_roi  = np.round(_y[roi_mask], decimals=8)

# Physical-domain mask — excludes the absorbing-boundary padding added
# around the box by AbsorbingBoundaryParameters, so full-domain diagnostics
# (e.g. the v(y) profile) only average over real (non-PML) nodes.
_domain_mask = (_x >= x0) & (_x <= x1) & (_y >= y0) & (_y <= y1)

# ── Mirror-symmetry map about x = 50 ───────────────────────────────────────
# The layered medium (horizontal interfaces only) and the acquisition
# geometry (sources/receivers mirrored top<->bottom) are both symmetric
# about x = (x0 + x1) / 2 = 50, so the true sensitivity kernel is too.
# Nearest-node lookup maps each node to its mirror-image node; used to fold
# the summed gradient onto its own reflection at the end of objective().
from scipy.spatial import cKDTree
_mirror_axis  = (x0 + x1) / 2.0
_mirror_tree  = cKDTree(np.column_stack([_x, _y]))
_mirror_dist, _mirror_idx = _mirror_tree.query(
    np.column_stack([2.0 * _mirror_axis - _x, _y])
)
if np.max(_mirror_dist) > 1e-6:
    print(f"  [warning] mirror-symmetry node matching has max residual "
          f"{np.max(_mirror_dist):.3e} m (mesh may not be perfectly x-symmetric)")


def _mirror_x(field_nodal):
    """Reflect a nodal field about x = _mirror_axis via nearest-node lookup."""
    return field_nodal[_mirror_idx]


# Same mirror map, restricted to ROI nodes — used to fold the Fresnel-zone
# sensitivity mask onto its own reflection, consistently with the gradient
# symmetrization above (a point stays "sensed" if it OR its mirror point
# was inside the raw union-of-ellipses region).
_mirror_tree_roi = cKDTree(np.column_stack([_x_roi, _y_roi]))
_mirror_dist_roi, _mirror_idx_roi = _mirror_tree_roi.query(
    np.column_stack([2.0 * _mirror_axis - _x_roi, _y_roi])
)
if np.max(_mirror_dist_roi) > 1e-6:
    print(f"  [warning] ROI mirror-symmetry node matching has max residual "
          f"{np.max(_mirror_dist_roi):.3e} m (mesh may not be perfectly x-symmetric)")


def _mirror_x_roi(field_roi):
    """Reflect a ROI-restricted nodal field about x = _mirror_axis."""
    return field_roi[_mirror_idx_roi]


_fwi_iter   = [0]
_fwi_ls     = [0]
_diag_cache      = {}   # populated by objective() from DIAG_EVENT; read by the L-BFGS loop for plotting
_diag_pair_cache = {}   # populated by objective() from DIAG_PAIR_EVENT/DIAG_RX_IDX; read by the L-BFGS loop


def _average_vp_in_x(vp_nodal):
    """Return vp_nodal with VP replaced by its x-average at every y-level.

    Enforces the layered-medium prior v(y): all nodes sharing the same
    depth receive the mean VP of that depth slice.
    """
    out = vp_nodal.copy()
    for yv in np.unique(_y_round_full):
        mask = _y_round_full == yv
        out[mask] = vp_nodal[mask].mean()
    return out


def build_mesh_from_model(m_flat):
    """Decode ln(VP) -> VP, enforce the v(y) prior, attach it to a fresh
    copy of the mesh.

    Returns
    -------
    mesh_it  : mesh with the new VP field attached
    vp_nodal : np.ndarray, shape (n_nodes,)       full (x-averaged) VP field
    vp_roi   : np.ndarray, shape (n_roi_nodes,)   VP restricted to the ROI
    """
    vp_roi   = np.clip(np.exp(m_flat), VP_MIN, VP_MAX)
    vp_nodal = _vp_bg_nodal.copy()
    vp_nodal[roi_mask] = vp_roi

    vp_nodal = _average_vp_in_x(vp_nodal)
    vp_roi   = vp_nodal[roi_mask]

    mesh_it = copy.deepcopy(base_mesh)
    new_enf = vp_nodal[mesh_it.connectivity]
    mesh_it.attach_field("VP", new_enf)
    return mesh_it, vp_nodal, vp_roi


def run_source(event, sim_name, vp_nodal, iteration_tag):
    """Adjoint source, adjoint wavefield and sensitivity kernel for ONE
    source, given the forward synthetic already computed under `sim_name`.

    Returns
    -------
    misfit_i : float
    K_lnVP_i : np.ndarray, shape (n_nodes,)   dJ_i/dln(VP)
    diag_i   : dict of arrays for plot_fwi_diagnostics (this source only)
    """
    print(f'    [{event}] extracting forward synthetic ...')
    phi_syn_i, phi_t_i = extract_data(
        sim_name, event, receiver_field='phi', field='phi_t',
    )

    misfit_i, f_adj_i, event_adj_i, event_config_adj_i = misfit_adjoint_source(
        phi_obs_dict[event], phi_syn_i, event=event, iteration=iteration_tag,
    )

    print(f'    [{event}] running adjoint simulation ...')
    phi_adj_i = adjoint_wavefield(
        p, event_adj_i, event_config_adj_i,
        model_simulation_name=sim_name,
        field='phi_t', iteration=iteration_tag,
        sampling_interval_in_time_steps=SAMPLING_INTERVAL,
    )

    print(f'    [{event}] computing sensitivity kernel ...')
    # A = 1/VP^2  ->  dA/dVP = -2/VP^3
    # dJ/dln(VP) = VP . dJ/dVP = VP . K_A . (-2/VP^3) = K_A . (-2/VP^2)
    K_A_i    = compute_gradient(phi_t_i, phi_adj_i)
    K_A_i    = np.where(np.isfinite(K_A_i), K_A_i, 0.0)
    K_lnVP_i = K_A_i * (-2.0 / vp_nodal**2)

    diag_i = dict(phi_syn=phi_syn_i, f_adj=f_adj_i, phi_t=phi_t_i, phi_t_adj=phi_adj_i)
    return misfit_i, K_lnVP_i, diag_i


# ── Objective function (multi-source) ─────────────────────────────────────
def objective(m_flat):
    """
    Multi-source FWI objective — model parameter is ln(VP).

    Misfit and sensitivity kernel are summed over all `event_ls` sources
    (standard adjoint-state multi-source FWI: J = sum_i J_i, dJ/dm = sum_i dJ_i/dm).

    Parameters
    ----------
    m_flat : np.ndarray, shape (n_roi_nodes,)
        Model vector: ln(VP) at each ROI node.

    Returns
    -------
    misfit : float                             total misfit, summed over sources
    grad   : np.ndarray, shape (n_roi_nodes,)   dJ/dln(VP), zeroed outside Fresnel zone
    """
    it      = _fwi_iter[0]
    ls      = _fwi_ls[0]
    _fwi_ls[0] += 1
    sim_name = f'fwi_a{stage_idx}_{it:03d}_grad' if ls == 0 else f'fwi_a{stage_idx}_{it:03d}_ls_{ls}'
    print(f"\n=== objective() call: alpha={alpha} iter={it} ls={ls} sim_name={sim_name!r} ===")

    # ── 1. Decode ln(VP) -> VP, enforce v(y) by x-averaging ───────────────
    mesh_it, vp_nodal, vp_roi = build_mesh_from_model(m_flat)
    vp_check = elemental_nodal_to_nodal_field(
        mesh_it.element_nodal_fields['VP'], mesh_it.connectivity
    )
    print(f'  [obj] VP(y) in ROI: mean={vp_roi.mean():.1f}  '
          f'range=[{vp_roi.min():.0f}, {vp_roi.max():.0f}]'
          f'  | mesh check: {vp_check[roi_mask].mean():.1f}')

    # ── Debug: v(y) profile + 2-D VP field ────────────────────────────────
    # Restricted to _domain_mask so the profile excludes absorbing-boundary
    # nodes outside the physical [x0,x1] x [y0,y1] box.
    _y_unique   = np.unique(_y_round_full[_domain_mask])
    _vp_profile = np.array([
        vp_nodal[_domain_mask & (_y_round_full == yv)].mean() for yv in _y_unique
    ])
    fig_vp, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    tp_vp = axes[0].tripcolor(
        base_mesh.points[:, 0], base_mesh.points[:, 1],
        vp_nodal, cmap="RdBu_r", shading="gouraud",
    )
    plt.colorbar(tp_vp, ax=axes[0], label="VP  [m/s]")
    axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Y (m)")
    axes[0].set_title(rf"VP field — iter {it}  ls {ls}", fontsize=12)
    axes[0].set_xlim(x0, x1); axes[0].set_ylim(y0, y1); axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.15)
    axes[1].plot(_vp_profile, _y_unique, "b-o", markersize=3, linewidth=1.5)
    axes[1].set_xlabel("VP  [m/s]"); axes[1].set_ylabel("Y (m)")
    axes[1].set_title(rf"v(y) profile — iter {it}  ls {ls}", fontsize=12)
    axes[1].set_ylim(y0, y1)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(IMAGE_DIR_WIN) / f'iter_{it:03d}_ls_{ls}_vp_field_alpha_{alpha}.png')
    plt.close(fig_vp)

    # ── 2. Register and run forward simulation — ALL sources, one launch ──
    p.add_to_project(
        sn.UnstructuredMeshSimulationConfiguration(
            unstructured_mesh=mesh_it,
            name=sim_name,
            event_configuration=event_config_fwd,
        ),
        overwrite=True,
    )
    _stored = p.simulations.get_mesh(sim_name)
    _stored_vp = elemental_nodal_to_nodal_field(
        _stored.element_nodal_fields['VP'], _stored.connectivity
    )
    print(f'  [stored mesh] VP in ROI: mean={_stored_vp[roi_mask].mean():.1f}  '
          f'range=[{_stored_vp[roi_mask].min():.0f}, {_stored_vp[roi_mask].max():.0f}]')
    forward_simulation(sim_name, event_ls,
                       fields=['phi_t'],
                       sampling_interval_in_time_steps=SAMPLING_INTERVAL)

    # ── 3. Per-source misfit + adjoint + kernel, summed over event_ls ─────
    misfit_total = 0.0
    K_lnVP_total = np.zeros_like(vp_nodal)
    for src_idx, event in enumerate(event_ls):
        print(f'  Source {src_idx + 1}/{len(event_ls)}: {event} ...')
        misfit_i, K_lnVP_i, diag_i = run_source(
            event, sim_name, vp_nodal, iteration_tag=f'{sim_name}_{event}',
        )
        misfit_total += misfit_i
        K_lnVP_total += K_lnVP_i
        print(f'    [{event}] misfit = {float(misfit_i):.6e}')
        if event == DIAG_EVENT:
            _diag_cache.update(diag_i)

        # Cache observed vs. synthetic trace at the configured (tx, rx) pair
        # for this evaluation. Not saved here — the L-BFGS loop saves one
        # snapshot per accepted iteration (plus the initial model) so that
        # the saved-pair count always matches model_history/misfit_history.
        if event == DIAG_PAIR_EVENT:
            _diag_pair_cache.update(
                time=np.array(phi_obs_dict[event].time.values),
                obs=np.array(phi_obs_dict[event][DIAG_RX_IDX]),
                syn=np.array(diag_i['phi_syn'][DIAG_RX_IDX]),
            )

    # ── Enforce mirror symmetry about x=50 (medium + acquisition geometry
    # are both symmetric, so fold the kernel onto its own reflection) ─────
    K_lnVP = K_lnVP_total + _mirror_x(K_lnVP_total)
    _diag_cache['K_VP']  = K_lnVP
    _diag_cache['K_roi'] = K_lnVP[roi_mask]

    # ── Debug: plot dJ/dln(VP) over full domain ───────────────────────────
    _K_lnVP_max = np.max(np.abs(K_lnVP)) if np.max(np.abs(K_lnVP)) > 0 else 1.0
    fig_k, ax_k = plt.subplots(figsize=(8, 6), dpi=100)
    tp_k = ax_k.tripcolor(base_mesh.points[:, 0], base_mesh.points[:, 1],
                           K_lnVP, cmap="RdBu", shading="gouraud", vmin=-_K_lnVP_max, vmax=_K_lnVP_max)
    plt.colorbar(tp_k, ax=ax_k, label=r"$\partial J/\partial \ln V_P$")
    ax_k.set_xlabel("X (m)"); ax_k.set_ylabel("Y (m)")
    ax_k.set_title(rf"$\partial J/\partial \ln V_P$ — iter {it}  ls {ls}", fontsize=12)
    ax_k.set_xlim(x0, x1); ax_k.set_ylim(y0, y1); ax_k.set_aspect("equal")
    ax_k.grid(alpha=0.15); plt.tight_layout()
    plt.savefig(Path(IMAGE_DIR_WIN) / f'iter_{it:03d}_ls_{ls}_gradient_full_alpha_{alpha}.png')
    plt.close(fig_k)

    # ── 4. Apply ellipse mask: zero gradient outside Fresnel zone ─────────
    grad_lnvp = K_lnVP[roi_mask]
    grad_lnvp[~_ellipse_roi] = 0.0

    g_max = np.max(np.abs(grad_lnvp))
    label = 'grad' if ls == 0 else f'ls_{ls}'
    print(f"  iter {it:3d} {label:6s} | misfit = {float(misfit_total):.6e}"
          f" | |dJ/dlnVP| max = {g_max:.3e}")
    return float(misfit_total), grad_lnvp.ravel()

def _save_diag_pair_snapshot(tag):
    """Plot the currently-cached (tx, rx) observed/synthetic trace.

    Reads `_diag_pair_cache`, populated by the most recent objective() call
    for DIAG_PAIR_EVENT/DIAG_RX_IDX. Returns the synthetic trace so callers
    can build a per-iteration history array aligned with model_history.
    `time`/`obs` are constant within a stage (only the wavelet's centre
    frequency changes, and that's fixed per stage), so they aren't saved
    per iteration here — the stage-level save bundles one `time`/`obs`
    pair together with the full syn_history array instead.
    """
    _t_pair   = _diag_pair_cache['time']
    _obs_pair = _diag_pair_cache['obs']
    _syn_pair = _diag_pair_cache['syn']
    fig_pair, ax_pair = plt.subplots(figsize=(8, 4), dpi=100)
    ax_pair.plot(_t_pair * 1e3, _obs_pair, color='red', lw=1.5, label=r'$\phi_\mathrm{obs}$')
    ax_pair.plot(_t_pair * 1e3, _syn_pair, color='green', lw=1.5, label=rf'$\phi^{{({tag})}}$')
    ax_pair.set_xlabel('Time (ms)'); ax_pair.set_ylabel(r'$\phi$')
    ax_pair.set_title(rf"tx={DIAG_PAIR_EVENT}  rx={DIAG_RX_STATION} — {tag}", fontsize=12)
    ax_pair.legend(); ax_pair.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(IMAGE_DIR_WIN) / f'{tag}_txrx_pair_alpha_{alpha}.png')
    plt.close(fig_pair)
    return _syn_pair


def run_lbfgs_trust_region(max_iter=10, max_tries=30, M=10,
                            delta_tr_init=0.3, delta_max=1.0,
                            eta1=0.0001, eta2=0.75,
                            gamma1=0.5, gamma2=0.9, sigma=2.0,
                            vp_init=None):
    """L-BFGS + trust region multi-source FWI optimizer.

    Model parameter m = ln(VP). All steps and radii are dimensionless
    (fractional changes in VP). Meshing uses VP = exp(m) directly.
    Trust region replaces a fixed step size with an adaptive radius delta_tr.
    After each step, rho = actual_reduction / predicted_reduction decides
    whether to accept the step and whether to shrink/expand delta_tr.
    Gradient is cached across rejected steps — no redundant simulations.
    `max_iter` counts accepted steps only; rejected attempts do not advance k.

    v(y) layered prior: objective() x-averages VP before meshing, and after
    each accepted step m is projected to the same layered space so the
    L-BFGS memory (s, y pairs) stays consistent with the effective model.

    `vp_init`, if given, is the ROI VP field (shape (n_roi_nodes,)) to warm-
    start from — used to chain frequency stages together in a multiscale
    continuation. Defaults to the homogeneous starting VP.

    Returns
    -------
    misfit_history : list[float],       length n_accepted + 1
    model_history  : np.ndarray, shape (n_accepted + 1, n_roi_nodes)
    syn_history    : np.ndarray, shape (n_accepted + 1, n_time) — synthetic
                      trace at (DIAG_PAIR_EVENT, DIAG_RX_IDX) per iteration

    All three include a leading entry for the initial (pre-iteration) model,
    so they are always the same length: n_accepted + 1.
    """
    print(f"Starting L-BFGS trust-region optimization: "
          f"max_iter={max_iter}, max_tries={max_tries}, M={M}")

    def _mesh_vp_roi(vp_roi_vals):
        """Return the ROI VP after the full-mesh x-averaging applied in objective()."""
        _vp_n = _vp_bg_nodal.copy()
        _vp_n[roi_mask] = vp_roi_vals
        _vp_n = _average_vp_in_x(_vp_n)
        return np.clip(_vp_n[roi_mask], VP_MIN, VP_MAX)

    delta_tr = delta_tr_init

    if vp_init is None:
        m = np.full(roi_mask.sum(), np.log(VP))
    else:
        m = np.log(np.clip(vp_init, VP_MIN, VP_MAX))
    model_history  = [_mesh_vp_roi(np.clip(np.exp(m), VP_MIN, VP_MAX))]

    s_list, y_list, rho_list = [], [], []
    prev_grad = None
    prev_s    = None

    _fwi_iter[0] = 0
    _fwi_ls[0]   = 0

    # ── Initial evaluation (iteration 0): seeds the gradient cache and
    # gives misfit_history / syn_history a leading entry, so both stay the
    # same length as model_history (n_accepted + 1) ───────────────────────
    print("  evaluating objective() for the initial model ...")
    cached_misfit, cached_grad = objective(m)
    misfit_history = [cached_misfit]
    syn_history    = [_save_diag_pair_snapshot('iter_000')]

    k     = 0   # accepted iteration counter
    tries = 0   # total attempt counter

    while k < max_iter and tries < max_tries:
        tries += 1
        _fwi_iter[0] = k
        _fwi_ls[0]   = 0

        # ── Reuse cached gradient if step was rejected (m unchanged) ──────
        print(f"\n--- L-BFGS attempt {tries} (accepted so far: {k}/{max_iter}) ---")
        if cached_misfit is not None:
            misfit, grad = cached_misfit, cached_grad
            print(f"  [cache] reusing gradient from previous rejected step")
        else:
            print("  evaluating objective() for the gradient step ...")
            misfit, grad = objective(m)

        # ── Update L-BFGS memory ────────────────────────────────────────
        if prev_grad is not None and prev_s is not None:
            y_k = grad - prev_grad
            sy  = np.dot(prev_s, y_k)
            if sy > 0:
                s_list.append(prev_s.copy())
                y_list.append(y_k.copy())
                rho_list.append(1.0 / sy)
                if len(s_list) > M:
                    s_list.pop(0); y_list.pop(0); rho_list.pop(0)

        # ── L-BFGS two-loop recursion ───────────────────────────────────
        q, alphas = grad.copy(), []
        for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a = rho * np.dot(s, q); q -= a * y; alphas.append(a)
        gamma_h = (np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
                   if s_list else 1.0)
        r = gamma_h * q
        for s, y, rho, a in zip(s_list, y_list, rho_list, reversed(alphas)):
            r += s * (a - rho * np.dot(y, r))
        d = -r

        # ── Constrain step to trust region (ln-space, L∞ ball) ─────────
        d_max = np.abs(d).max()
        if d_max < 1e-30:
            print(f"iter {k} | direction vanished — stopped"); break
        step        = delta_tr * (d / d_max)
        m_new       = np.clip(m + step, np.log(VP_MIN), np.log(VP_MAX))
        actual_step = m_new - m

        # ── Evaluate proposed model ─────────────────────────────────────
        _fwi_ls[0] = 1
        print("  evaluating objective() for the line-search trial ...")
        misfit_new, grad_new = objective(m_new)

        # ── rho = actual / predicted reduction ──────────────────────────
        actual_red    = misfit - misfit_new
        predicted_red = -np.dot(grad, actual_step)
        rho_k = actual_red / predicted_red if abs(predicted_red) > 1e-30 else 0.0
        step_norm = np.abs(actual_step).max()   # L∞ norm of accepted step

        # ── Accept / reject ─────────────────────────────────────────────
        if rho_k >= eta1:
            prev_grad = grad.copy()

            # Project m_new to v(y): x-average ln(VP) at each y-level in ROI
            # This keeps m consistent with the x-averaged mesh used in objective()
            _vp_acc  = np.clip(np.exp(m_new), VP_MIN, VP_MAX)
            _y_levels_roi = np.unique(_y_round_roi)
            _vp_profile_roi = np.array([
                _vp_acc[_y_round_roi == _yv].mean() for _yv in _y_levels_roi
            ])
            # Smooth the depth profile to regularize against per-iteration noise
            _vp_profile_roi = gaussian_filter1d(_vp_profile_roi, sigma=GAUSSIAN_SIGMA_Y)
            for _yv, _vp_val in zip(_y_levels_roi, _vp_profile_roi):
                _vp_acc[_y_round_roi == _yv] = _vp_val
            m_proj = np.clip(np.log(_vp_acc), np.log(VP_MIN), np.log(VP_MAX))

            prev_s        = m_proj - m      # step measured to projected model
            m             = m_proj
            cached_misfit = misfit_new
            cached_grad   = grad_new
            misfit_history.append(misfit_new)
            model_history.append(_mesh_vp_roi(_vp_acc))
            status = "accepted"
            k += 1
            # _diag_pair_cache still holds the trace from the objective(m_new)
            # call above, i.e. the just-accepted state — snapshot it here so
            # syn_history stays index-aligned with model_history/misfit_history.
            syn_history.append(_save_diag_pair_snapshot(f'iter_{k:03d}'))

            # ── Diagnostic plots for this accepted step (DIAG_EVENT) ──────
            plot_fwi_diagnostics(
                it=k,
                phi_obs=phi_obs_dict[DIAG_EVENT],
                phi_syn=_diag_cache['phi_syn'],
                f_adj=_diag_cache['f_adj'],
                phi_t=_diag_cache['phi_t'],
                phi_t_adj=_diag_cache['phi_t_adj'],
                K_VP=_diag_cache['K_VP'],
                K_roi=_diag_cache['K_roi'],
                x=_x, y=_y, x_roi=_x_roi, y_roi=_y_roi,
                x0_roi=x0_roi, x1_roi=x1_roi, y0_roi=y0_roi, y1_roi=y1_roi,
                sr_pairs=sr_pairs, margin=fresnel_margin,
            )
        else:
            cached_misfit = misfit
            cached_grad   = grad
            status = "REJECTED"

        # ── Update trust region radius (Nocedal & Wright §4.1) ──────────────────
        if rho_k < eta1:
            delta_tr = min(delta_tr / sigma, step_norm)
        elif rho_k > eta2 and step_norm >= gamma2 * delta_tr:
            delta_tr = min(sigma * delta_tr, delta_max)
        elif step_norm < gamma1 * delta_tr:
            delta_tr = max(delta_tr / sigma, sigma * step_norm)

        n_events  = len(event_ls)
        grad_norm = np.linalg.norm(grad)
        vp_mean   = _mesh_vp_roi(np.clip(np.exp(m), VP_MIN, VP_MAX))[_ellipse_roi].mean()
        print(f"Iteration {k}: n_events={n_events}  VP_mean={vp_mean:.1f} m/s  "
              f"chi={misfit:.6e}  ||g||={grad_norm:.3e}")
        print(f"  pred={predicted_red:.3e}  ared={actual_red:.3e}  "
              f"||Δm||_∞={step_norm:.4f}  Δtr={delta_tr:.4f}  ({status})")

    print(f"L-BFGS finished: {k} accepted iteration(s), {tries} total attempt(s).")
    assert len(misfit_history) == len(model_history) == len(syn_history), (
        f"history length mismatch: misfit={len(misfit_history)} "
        f"model={len(model_history)} syn={len(syn_history)}"
    )
    return misfit_history, np.array(model_history), np.array(syn_history)


# ── Multi-frequency (multiscale) FWI driver ────────────────────────────────
# Stages run from low to high alpha (= low to high centre frequency), each
# warm-started from the previous stage's recovered v(y) model and capped at
# MAX_ITER_PER_STAGE accepted L-BFGS iterations.

all_misfit_history = []
all_model_history  = []
all_syn_history    = []
stage_boundaries    = [0]   # cumulative accepted-iteration index at the START of each stage
stage_summary       = []    # (alpha, f_c, n_accepted)

vp_warm_start = None   # None -> first stage starts from the homogeneous VP

for stage_idx, alpha in enumerate(alpha_ls):
    f_c = alpha * CENTRAL_FREQUENCY   # centre frequency [Hz] for this stage
    print(f"\n{'='*70}\nFrequency stage {stage_idx + 1}/{len(alpha_ls)}: "
          f"alpha={alpha}  f_c={f_c:.1f} Hz\n{'='*70}")

    time_shift_in_seconds = 1 / f_c
    wavelet = sn.simple_config.stf.Ricker(
        center_frequency=f_c,
        time_shift_in_seconds=time_shift_in_seconds,
    )
    event_config = sn.EventConfiguration(
        wavelet=wavelet,
        waveform_simulation_configuration=waveform_config,
    )
    event_config_fwd = event_config

    init_simulation = f'forward_simulation_homogeneous_model_a{stage_idx}'
    obs_simulation  = f'forward_simulation_layred_model_a{stage_idx}'

    # Homogeneous (starting) model — used only to measure the wavelet's
    # 3dB pulse width for this stage's Fresnel margin.
    print(f"Registering simulation configuration {init_simulation!r} ...")
    p.add_to_project(
        sn.UnstructuredMeshSimulationConfiguration(
            unstructured_mesh=mesh_homo,
            name=init_simulation,
            event_configuration=event_config,
        ),
        overwrite=True,
    )

    # True (layered) model — used to generate observed data at this frequency
    print(f"Registering simulation configuration {obs_simulation!r} ...")
    p.add_to_project(
        sn.UnstructuredMeshSimulationConfiguration(
            unstructured_mesh=mesh,
            name=obs_simulation,
            event_configuration=event_config,
        ),
        overwrite=True,
    )

    t0 = datetime.now()

    print(f"Stage {stage_idx}: homogeneous-model simulation (DIAG_EVENT only) ...")
    forward_simulation(
        simulation_name=init_simulation,
        events=DIAG_EVENT,
        fields=['phi_t'],
        sampling_interval_in_time_steps=10,
    )
    try:
        p.simulations.get_simulation_output_directory(init_simulation, DIAG_EVENT)
    except Exception as e:
        raise RuntimeError(
            f"{init_simulation} produced no results for DIAG_EVENT={DIAG_EVENT!r} "
            f"even though the launch call returned without raising."
        ) from e
    print("  DIAG_EVENT homogeneous-model results confirmed present.")

    print(f"Stage {stage_idx}: true-model simulation for {len(event_ls)} source(s) (batched) ...")
    forward_simulation(
        simulation_name=obs_simulation,
        events=event_ls,
        fields=None,
    )

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"Stage {stage_idx} initial simulations done in {int(elapsed//60)}m {elapsed%60:.1f}s")

    phi_obs_diag, _ = extract_data(
        simulation=obs_simulation,
        events=DIAG_EVENT,
        receiver_field='phi',
        field=None,
    )
    fig_obs, ax_obs = plt.subplots()
    ax_obs.plot(phi_obs_diag.time, phi_obs_diag.values[0, :])
    ax_obs.set_xlabel('Time (s)')
    ax_obs.set_ylabel('phi')
    ax_obs.set_title(f'Observed trace — {DIAG_EVENT}  (alpha={alpha}, f_c={f_c:.1f} Hz)')
    plt.savefig(Path(IMAGE_DIR_WIN) / f'observed_trace_{DIAG_EVENT}_alpha_{alpha}.png')
    plt.close(fig_obs)

    # Observed data for every source — fixed for the duration of this stage
    # (the true model doesn't change, but the wavelet does between stages).
    phi_obs_dict = {
        ev: extract_data(obs_simulation, ev, receiver_field='phi', field=None)[0]
        for ev in event_ls
    }

    # DIAG_EVENT's homogeneous-model synthetic at this stage's frequency —
    # only used to measure the wavelet's 3dB pulse width for the Fresnel margin.
    phi_homo, _ = extract_data(
        simulation=init_simulation, events=DIAG_EVENT, receiver_field='phi', field=None,
    )
    time_wf = phi_homo.time.values

    # ── Ellipse sensitivity mask for ROI nodes (first Fresnel zone) ───────
    # Recomputed each stage: the Fresnel margin depends on the wavelet's
    # pulse width, which shrinks as the centre frequency increases.
    width_3dB      = envelope_3dB_width(phi_homo[0], time_wf, plot=False)['width'] / 4
    fresnel_margin = VP * width_3dB   # wavelet-duration term, shared by every pair

    # Union over every acquired (source, receiver) pair; per-pair
    # semi-major axis a(s, r) = (dist(s, r) + fresnel_margin) / 2 —
    # computed per pair so distant pairs aren't silently excluded.
    _ellipse_roi = np.zeros(roi_mask.sum(), dtype=bool)
    for _s, _r in sr_pairs:
        _d1   = np.sqrt((_x_roi - _s[0])**2 + (_y_roi - _s[1])**2)
        _d_sr = np.sqrt((_s[0] - _r[0])**2 + (_s[1] - _r[1])**2)
        _a    = (_d_sr + fresnel_margin) / 2
        _d2   = np.sqrt((_x_roi - _r[0])**2 + (_y_roi - _r[1])**2)
        _ellipse_roi |= (_d1 + _d2) <= 2 * _a

    # Fold the mask onto its own mirror image about x=50, consistently with
    # the gradient's mirror symmetrization: a point stays "sensed" if it OR
    # its mirror point was inside the raw union-of-ellipses region.
    _ellipse_roi = _ellipse_roi | _mirror_x_roi(_ellipse_roi)
    print(f"Stage {stage_idx} ellipse mask: {_ellipse_roi.sum()} / {roi_mask.sum()} "
          f"ROI nodes inside Fresnel zone")

    misfit_history_stage, model_history_stage, syn_history_stage = run_lbfgs_trust_region(
        max_iter=MAX_ITER_PER_STAGE, vp_init=vp_warm_start,
    )
    assert len(misfit_history_stage) == len(model_history_stage) == len(syn_history_stage)

    vp_warm_start = model_history_stage[-1]   # warm-start the next (higher-frequency) stage

    # misfit/model/syn_history_stage all include the stage's starting state
    # as row 0. For stage_idx > 0 that row's model is vp_warm_start, i.e. a
    # duplicate of the previous stage's last accepted model already in
    # all_model_history — so only stage 0's row 0 (the true pre-FWI starting
    # state) is kept. This keeps all_misfit_history / all_model_history /
    # all_syn_history the same combined length.
    _stage_slice = slice(None) if stage_idx == 0 else slice(1, None)
    all_misfit_history.extend(misfit_history_stage[_stage_slice])
    all_model_history.append(model_history_stage[_stage_slice])
    all_syn_history.append(syn_history_stage[_stage_slice])

    n_rows_added = len(misfit_history_stage[_stage_slice])
    stage_boundaries.append(stage_boundaries[-1] + n_rows_added)
    stage_summary.append((alpha, f_c, len(misfit_history_stage) - 1))

    _data_dir = Path(DATA_DIR_WIN)
    np.save(_data_dir / f"misfit_history_multilayer_alpha_{alpha}.npy", np.array(misfit_history_stage))
    np.save(_data_dir / f"model_history_multilayer_alpha_{alpha}.npy",  model_history_stage)
    # time/obs are constant across this stage's iterations (only f_c varies
    # between stages) — save them once alongside the full syn_history array
    # instead of duplicating them in a per-iteration file.
    np.savez(_data_dir / f"txrx_pair_alpha_{alpha}.npz",
              time=np.array(phi_obs_dict[DIAG_PAIR_EVENT].time.values),
              obs=np.array(phi_obs_dict[DIAG_PAIR_EVENT][DIAG_RX_IDX]),
              syn=syn_history_stage)
    print(f"Stage {stage_idx} (alpha={alpha}, f_c={f_c:.1f} Hz): "
          f"{len(misfit_history_stage) - 1} accepted iteration(s) saved.")

print(f"\nMulti-frequency FWI finished: {stage_summary}")

# ── Combined misfit curve across all stages ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
ax.semilogy(all_misfit_history, marker="o", linewidth=1.5)
for b in stage_boundaries[1:-1]:
    ax.axvline(b - 0.5, color='gray', linestyle='--', linewidth=1)
for (_alpha, _f_c, _n), b0 in zip(stage_summary, stage_boundaries[:-1]):
    ax.text(b0, ax.get_ylim()[1], f'α={_alpha}\n{_f_c:.0f} Hz',
            ha='left', va='top', fontsize=8)
ax.set_xlabel("Accepted Iteration (cumulative across stages)")
ax.set_ylabel("L2 Misfit")
ax.set_title("Multi-frequency L-BFGS + Trust Region  [v(y) layered parameterisation]")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(Path(IMAGE_DIR_WIN) / "misfit_history_multilayer_multiscale.png")
plt.close(fig)

# ── Save combined results ───────────────────────────────────────────────────
_data_dir = Path(DATA_DIR_WIN)
_all_model_history = np.concatenate(all_model_history, axis=0)
_all_syn_history    = np.concatenate(all_syn_history, axis=0)
assert len(all_misfit_history) == len(_all_model_history) == len(_all_syn_history), (
    f"combined history length mismatch: misfit={len(all_misfit_history)} "
    f"model={len(_all_model_history)} syn={len(_all_syn_history)}"
)
np.save(_data_dir / "misfit_history_multilayer_multiscale.npy", np.array(all_misfit_history))
np.save(_data_dir / "model_history_multilayer_multiscale.npy",  _all_model_history)
np.save(_data_dir / "syn_history_multilayer_multiscale.npy",    _all_syn_history)
print(f"Saved combined misfit_history ({len(all_misfit_history)} entries), "
      f"model_history and syn_history to {_data_dir}")

# ── Total wall-clock time for the whole script ─────────────────────────────
_total_elapsed = (datetime.now() - _SCRIPT_START_TIME).total_seconds()
_h, _rem = divmod(_total_elapsed, 3600)
_m, _s   = divmod(_rem, 60)
np.save(_data_dir / "total_computation_time_seconds.npy", np.array(_total_elapsed))
with open(_data_dir / "total_computation_time.txt", "w") as _f:
    _f.write(f"{int(_h)}h {int(_m)}m {_s:.1f}s ({_total_elapsed:.1f} s)\n")
