"""
Verify Salvus' sensitivity kernels against the adjoint-method definition

    K_rho(x) = - int_0^T  du_dag/dt (x, T-t) . du/dt (x, t)  dt
    K_C(x)   =   int_0^T  grad u_dag (x, T-t) (x) grad u (x, t)  dt

for the initial (homogeneous 45 deg) model and the L2 misfit set up in
FWI_2D_singlelayer.py.

Procedure
---------
1. Forward run in the initial model, storing grad u and du/dt in the volume.
2. Take Salvus' own L2 adjoint source at the receiver, reverse it in time and
   inject it as a vector point force -> adjoint run, storing grad u_dag and
   du_dag/dt in the volume. Frame k of the reversed run is u_dag(x, T - t_k).
3. Build K_C per Salvus 2D Voigt parameter from the engineering strains
   e = (e_xx, e_yy, 2 e_xy):  dchi/dC_ab = int e_dag_a e_b (+ e_dag_b e_a if a != b)
4. Compare with Salvus' gradient (correlation, best-fit scale, relative error).

Run from the tomograhpy folder:  python verify_kernel_C.py
"""

from scipy.spatial import cKDTree

from FWI_2D_singlelayer import *   # model, mesh, project, initial model, gradients (cached)


# =============================================================================
# Settings
# =============================================================================

VOLUME_INTERVAL = 11                         # volume output every N steps (must divide n_steps)
FWD_VOLUME_SIM  = "initial_model_volume"     # forward run in the initial model with volume output
ADJ_SIM         = "adjoint_check"            # time-reversed adjoint-source run
ADJ_EVENT       = "src_1_adjoint"            # point force at the receiver
EVENT           = event_ls[0]                # event to verify (single source here)
KERNEL_PARAMETERS = sorted(STIFFNESS_PARAMETERS) + ['RHO']
VOLUME_FIELDS   = ["gradient-of-displacement", "velocity"]

n_steps = int(round((end_time - start_time) / time_step_in_seconds))
assert n_steps % VOLUME_INTERVAL == 0, \
    f"VOLUME_INTERVAL={VOLUME_INTERVAL} must divide n_steps={n_steps} so reversed frames align"
dt_frame = VOLUME_INTERVAL * time_step_in_seconds


# =============================================================================
# Salvus gradient incl. RHO
# =============================================================================

while p.actions.inversion.compute_gradients(
    simulation_configuration=INITIAL_SIM_NAME,
    misfit_configuration=MISFIT_NAME,
    events=EVENT,
    site_name=SITE_NAME,
    ranks_per_job=RANKS,
    gradient_parameters=set(KERNEL_PARAMETERS),
) is None:
    time.sleep(5)

gradient_salvus = p.actions.inversion.sum_gradients(
    simulation_configuration=INITIAL_SIM_NAME,
    misfit_configuration=MISFIT_NAME,
    events=EVENT,
    gradient_parameters=set(KERNEL_PARAMETERS),
)


# =============================================================================
# 1. Forward run with volume output
# =============================================================================

p.add_to_project(
    sn.UnstructuredMeshSimulationConfiguration(
        unstructured_mesh=mesh_init,
        name=FWD_VOLUME_SIM,
        event_configuration=sn.EventConfiguration(
            wavelet=stf,
            waveform_simulation_configuration=waveform_config,
        ),
    ),
    overwrite=True,
)
forward_simulation(FWD_VOLUME_SIM, EVENT, fields=VOLUME_FIELDS,
                   sampling_interval_in_time_steps=VOLUME_INTERVAL)


# =============================================================================
# 2. Adjoint run: Salvus' L2 adjoint source, time reversed, at the receiver
# =============================================================================

event_misfit = sn.EventMisfit(
    observed_event=p.waveforms.get(data_name="forward_simulation", events=EVENT)[0],
    synthetic_event=p.waveforms.get(data_name=INITIAL_SIM_NAME, events=EVENT)[0],
    misfit_function="L2",
    receiver_field="displacement",
)
rec_name = event_misfit.receiver_name_list[0]
adj = event_misfit.get_misfit_and_adjoint_source_for_receiver(rec_name)
assert list(adj['component'].values) == ['X', 'Y']
# Salvus stores the L2 adjoint source as -(u - d) * dt (time-integration weight
# included); the continuous adjoint force is f_dag = -(u - d), so divide by dt.
adj_src = adj['adjoint_sources'].values.astype(float) / time_step_in_seconds   # (2, nt), forward time
print(f"Misfit chi = {event_misfit.misfit_value:.6e}")

adjoint_stf = sn.simple_config.stf.Custom.from_array(
    adj_src[:, ::-1].T,                                       # reversed: s(t') = f_dag(T - t')
    sampling_rate_in_hertz=sampling_rate_in_hertz,
    start_time_in_seconds=start_time,
)
p.add_to_project(
    sn.Event(
        event_name=ADJ_EVENT,
        sources=sn.simple_config.source.cartesian.VectorPoint2D(
            x=r_loc_all[0, 0], y=r_loc_all[0, 1], fx=1.0, fy=1.0
        ),
        receivers=receivers,
    )
)
p.add_to_project(
    sn.UnstructuredMeshSimulationConfiguration(
        unstructured_mesh=mesh_init,
        name=ADJ_SIM,
        event_configuration=sn.EventConfiguration(
            wavelet=adjoint_stf,
            waveform_simulation_configuration=waveform_config,
        ),
    ),
    overwrite=True,
)
forward_simulation(ADJ_SIM, ADJ_EVENT, fields=VOLUME_FIELDS,
                   sampling_interval_in_time_steps=VOLUME_INTERVAL)


# =============================================================================
# 3. Kernels from the definition
# =============================================================================

def load_volume(sim, event, field):
    """
    Volume output as (n_frames, n_elem, n_comp, n_nodes) in mesh_init's element order.

    The solver writes elements in its own (permuted) order; they are matched
    back to mesh_init by centroid and every GLL node coordinate is checked.
    """
    wo = p.waveforms.get(data_name=sim, events=event)[0].get_wavefield_output(
        output_type='volume', field=field)
    assert wo.data.shape[0] == n_steps // VOLUME_INTERVAL + 1, wo.data.shape
    dist, perm = cKDTree(wo.get_element_centroid()).query(mesh_init.get_element_centroid())
    assert dist.max() < 1e-12 and len(np.unique(perm)) == len(perm), "element matching failed"
    assert np.allclose(wo.get_element_nodes()[perm], mesh_init.points[mesh_init.connectivity],
                       rtol=0.0, atol=1e-12), "GLL node order differs within elements"
    return wo.data[:, perm]


def engineering_strain(G):
    """grad u (.., 4, nodes) -> Voigt engineering strain (e_xx, e_yy, 2 e_xy)."""
    # diagonal entries are components 0 and 3; the off-diagonal sum is order-independent
    return G[:, :, 0], G[:, :, 3], G[:, :, 1] + G[:, :, 2]


grad_u   = load_volume(FWD_VOLUME_SIM, EVENT, "gradient-of-displacement")
grad_dag = load_volume(ADJ_SIM, ADJ_EVENT, "gradient-of-displacement")[::-1]   # u_dag(x, T - t_k)

e_fwd = engineering_strain(grad_u)
e_dag = engineering_strain(grad_dag)
del grad_u, grad_dag

voigt_index = {'C11': (0, 0), 'C12': (0, 1), 'C13': (0, 2),
               'C22': (1, 1), 'C23': (1, 2), 'C33': (2, 2)}
kernel_def = {}
for k, (a, b) in voigt_index.items():
    integrand = e_dag[a] * e_fwd[b]
    if a != b:
        integrand = integrand + e_dag[b] * e_fwd[a]
    kernel_def[k] = np.trapezoid(integrand.astype(np.float64), dx=dt_frame, axis=0)
del e_fwd, e_dag

v_fwd = load_volume(FWD_VOLUME_SIM, EVENT, "velocity")
v_dag = load_volume(ADJ_SIM, ADJ_EVENT, "velocity")[::-1]
# reversing time flips the sign of velocity: the reversed run's frames hold
# -du_dag/dt(x, T - t_k), so K_rho = -int du_dag/dt . du/dt dt = +int v_dag . v_fwd dt
kernel_def['RHO'] = np.trapezoid((v_dag * v_fwd).sum(axis=2).astype(np.float64), dx=dt_frame, axis=0)
del v_fwd, v_dag


# =============================================================================
# 4. Compare with Salvus
# =============================================================================

# nodes inside the physical domain, away from the source/receiver singularities
pts = mesh_init.points[mesh_init.connectivity]                    # (n_elem, n_nodes, 2)
inside = ((pts[..., 0] >= x0) & (pts[..., 0] <= x1) & (pts[..., 1] >= y0) & (pts[..., 1] <= y1))
d_src = np.linalg.norm(pts - tx_locations[0], axis=-1)
d_rec = np.linalg.norm(pts - r_loc_all[0], axis=-1)
far = inside & (d_src > 0.5e-3) & (d_rec > 0.5e-3)

print(f"\n{'param':6s} {'corr(all)':>10s} {'corr(far)':>10s} {'scale':>12s} {'rel.err(far)':>13s}")
results = {}
for k in KERNEL_PARAMETERS:
    ks = gradient_salvus.element_nodal_fields[k].astype(np.float64)   # float64: squares underflow in float32
    kd = kernel_def[k]
    corr_all = np.corrcoef(ks[inside], kd[inside])[0, 1]
    corr_far = np.corrcoef(ks[far], kd[far])[0, 1]
    scale = np.dot(ks[far], kd[far]) / np.dot(kd[far], kd[far])     # Salvus ~ scale * definition
    rel_err = np.linalg.norm(ks[far] - scale * kd[far]) / np.linalg.norm(ks[far])
    results[k] = (corr_all, corr_far, scale, rel_err)
    print(f"{k:6s} {corr_all:10.5f} {corr_far:10.5f} {scale:12.5g} {rel_err:13.3e}")


# side-by-side plots: Salvus | definition (scaled) | difference
mm = 1e3
tri = mtri.Triangulation(mesh_init.points[:, 0] * mm, mesh_init.points[:, 1] * mm)
fig, axes = plt.subplots(len(KERNEL_PARAMETERS), 3, figsize=(12, 3.6 * len(KERNEL_PARAMETERS)), dpi=100)
for row, k in zip(axes, KERNEL_PARAMETERS):
    scale = results[k][2]
    fields = [gradient_salvus.element_nodal_fields[k], scale * kernel_def[k],
              gradient_salvus.element_nodal_fields[k] - scale * kernel_def[k]]
    vmax = np.percentile(np.abs(fields[0][inside]), 99.0)
    for ax, f, title in zip(row, fields, ['Salvus', f'definition x {scale:.3g}', 'difference']):
        tp = ax.tripcolor(tri, elemental_nodal_to_nodal_field(f, mesh_init.connectivity),
                          cmap='seismic', vmin=-vmax, vmax=vmax, shading='gouraud')
        ax.set_title(f'K_{k} — {title}', fontsize=10)
        ax.set_xlim(x0 * mm, x1 * mm)
        ax.set_ylim(y0 * mm, y1 * mm)
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
    plt.colorbar(tp, ax=row.tolist(), shrink=0.8)
plt.savefig(Path(IMAGE_DIR_WIN) / 'kernel_C_verification.png', bbox_inches='tight')
plt.close(fig)
