"""
Model-error metrics and figures of an orientation FWI run against the true layered model.

Called at the end of every orientation FWI script (and by plot_model_error.py for saved runs):

    from my_code.model_error import save_model_error_figures
    save_model_error_figures(model_history_rad, roi_xy, alpha_ls, stage_counts, CENTRAL_FREQUENCY,
                             interfaces_y, layer_angles_deg, background_deg, (x0, x1), out_dir)

Figures (project journal format, my_code/plot_style.py):
  model_error_convergence : RMS theta error vs iteration, relative error ||d|| / ||d_0|| on the right axis
  model_error_profile     : recovered vs true vs starting theta(z)
  model_error_layers      : mean +- std of theta in each layer vs the true value
Interior RMS and interface depths are printed only.

Metrics (inside the physical x range, rows exactly on an interface excluded):
  * error  d = wrap(theta - theta_true) into [-90, 90) deg (orientation is 180-deg periodic)
  * RMS    sqrt(sum w d^2 / sum w), w = thickness of each depth row (area weighting; GLL rows
           are unevenly spaced, a plain node mean would over-weight the dense rows)
  * relative error ||d|| / ||d_0|| = RMS / RMS of the starting model (1 = no gain, 0 = exact)
"""

import numpy as np
import matplotlib.pyplot as plt

from my_code.plot_style import (apply_paper_style, save_paper_figure, SINGLE_COL, PAPER_FIGSIZE, MM,
                                C_SYN, C_TRUE, GRID_KW)


def wrap_deg(d):
    """Orientation difference wrapped into [-90, 90) deg."""
    return (d + 90.0) % 180.0 - 90.0


def true_theta(z, interfaces_y, layer_angles_deg, background_deg):
    """True theta(z) [deg]: layers between consecutive interfaces (top to bottom); NaN on an interface."""
    t = np.full_like(z, background_deg, dtype=float)
    for (top, bot), a in zip(zip(interfaces_y[:-1], interfaces_y[1:]), layer_angles_deg):
        t[(z < top) & (z > bot)] = a
    t[np.min(np.abs(z[:, None] - np.asarray(interfaces_y)[None, :]), axis=1) < 1e-9] = np.nan
    return t


def save_model_error_figures(model_history_rad, roi_xy, stage_alphas, stage_counts,
                             central_frequency, interfaces_y, layer_angles_deg, background_deg,
                             x_range, out_dir, interior_margin=0.25e-3):
    """
    Compute the model-error metrics, print a summary and save the three figures in `out_dir`.

    model_history_rad : (n_models, n_roi) theta [rad] of every saved model, multiscale order
    roi_xy            : (n_roi, 2) node coordinates [m]
    stage_alphas      : alpha of each frequency stage
    stage_counts      : number of models of each stage in model_history_rad
    interfaces_y      : true interfaces, top to bottom [m]; layer k lies between [k] and [k + 1]
    layer_angles_deg  : true theta of each layer [deg]
    x_range           : (x0, x1) physical domain [m]; nodes outside (absorbing layers) are ignored

    Returns a dict of the metric histories and the final per-layer statistics.
    """
    apply_paper_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    x0, x1 = x_range
    models = np.rad2deg(np.asarray(model_history_rad))
    assert sum(stage_counts) == len(models), (stage_counts, models.shape)

    # theta(z) per model: mean over each depth row (the inversion is x-averaged anyway)
    inside = (roi_xy[:, 0] >= x0 - 1e-9) & (roi_xy[:, 0] <= x1 + 1e-9)
    z_round = np.round(roi_xy[:, 1], 10)
    z_lv = np.unique(z_round[inside])
    row_of = np.searchsorted(z_lv, z_round[inside])
    n_per_row = np.bincount(row_of)
    profiles = np.stack([np.bincount(row_of, weights=m[inside]) / n_per_row for m in models])

    w = np.gradient(z_lv)                                   # row thickness [m]
    t_true = true_theta(z_lv, interfaces_y, layer_angles_deg, background_deg)
    valid = ~np.isnan(t_true)
    d_if = np.min(np.abs(z_lv[:, None] - np.asarray(interfaces_y)[None, :]), axis=1)
    interior = valid & (d_if >= interior_margin)

    def rms(d, mask):
        return np.sqrt(np.sum(w[mask] * d[mask] ** 2) / np.sum(w[mask]))

    errors = wrap_deg(profiles - t_true[None, :])
    rms_all = np.array([rms(e, valid) for e in errors])
    rms_int = np.array([rms(e, interior) for e in errors])
    rel_err = rms_all / rms_all[0]
    mae = np.array([np.sum(w[valid] * np.abs(e[valid])) / np.sum(w[valid]) for e in errors])
    max_err = np.array([np.abs(e[valid]).max() for e in errors])
    # undefined for a constant profile: the starting model, or a single-layer true model
    corr = np.array([np.corrcoef(p[valid], t_true[valid])[0, 1]
                     if np.ptp(p[valid]) > 0 and np.ptp(t_true[valid]) > 0 else np.nan
                     for p in profiles])
    bounds = np.concatenate([[0], np.cumsum(stage_counts)])
    it = np.arange(len(models))

    # ── per-layer statistics and inner-interface depths of the final model ──
    final = profiles[-1]
    layer_rows = []
    for (top, bot), a in zip(zip(interfaces_y[:-1], interfaces_y[1:]), layer_angles_deg):
        s = (z_lv < top) & (z_lv > bot)
        mean = np.sum(w[s] * final[s]) / np.sum(w[s])
        std = np.sqrt(np.sum(w[s] * (final[s] - mean) ** 2) / np.sum(w[s]))
        layer_rows.append((bot, top, a, mean, std))
    # outer interfaces are the edges of the inverted region (jump imposed, not recovered)
    iface = []
    for y_if, a_up, a_lo in zip(interfaces_y[1:-1], layer_angles_deg[:-1], layer_angles_deg[1:]):
        mid = 0.5 * (a_up + a_lo)
        near = np.where(np.abs(z_lv - y_if) < 0.7e-3)[0]
        zz, pp = z_lv[near], final[near]
        c = np.where(np.diff(np.sign(pp - mid)))[0]
        if len(c):   # crossing closest to the true interface
            est = [zz[j] + (mid - pp[j]) * (zz[j + 1] - zz[j]) / (pp[j + 1] - pp[j]) for j in c]
            iface.append((y_if, min(est, key=lambda e: abs(e - y_if))))

    # ── console summary ──────────────────────────────────────────────────────
    print(f"\nModel error: {len(models)} models, stages alpha={list(stage_alphas)} "
          f"({[int(c) for c in stage_counts]} models)")
    print(f"{'':10s} {'RMS':>7s} {'interior':>9s} {'MAE':>7s} {'max':>7s} {'rel.err':>8s} {'corr':>6s}")
    for name, i in [('start', 0)] + [(f'end a={a}', b - 1) for a, b in zip(stage_alphas, bounds[1:])]:
        print(f"{name:10s} {rms_all[i]:7.2f} {rms_int[i]:9.2f} {mae[i]:7.2f} {max_err[i]:7.1f} "
              f"{rel_err[i]:8.3f} {corr[i]:6.3f}")
    print("layer        true   mean +- std  [deg]")
    for bot, top, a, mean, std in layer_rows:
        print(f"{bot * MM:4.1f}-{top * MM:4.1f} mm  {a:5.1f}  {mean:5.1f} +- {std:4.1f}")
    for t, e in iface:
        print(f"interface {t * MM:5.2f} mm -> recovered {e * MM:5.2f} mm ({(e - t) * MM:+.2f})")

    # ── 1. convergence: RMS, relative error on the right axis (= RMS / RMS_0) ──
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6), layout='constrained')
    for k, a in enumerate(stage_alphas):
        lo, hi = bounds[k] - 0.5, bounds[k + 1] - 0.5
        if k % 2 == 1:
            ax.axvspan(lo, hi, color='0.93', zorder=0, lw=0)
        ax.text(0.5 * (lo + hi), 1.0, f"{a * central_frequency / 1e6:.1f} MHz",
                transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
    ax.plot(it, rms_all, color=C_SYN, marker='o', ms=2.5, lw=0.9)
    ax.set_xlabel('FWI iteration'); ax.set_ylabel(r'RMS $\theta$ error (deg)')
    ax.set_ylim(0, 1.05 * rms_all[0])
    ax.yaxis.grid(True, **GRID_KW)
    ax_rel = ax.secondary_yaxis('right', functions=(lambda r: r / rms_all[0], lambda q: q * rms_all[0]))
    ax_rel.set_ylabel(r'$\|\theta - \theta_\mathrm{true}\| \,/\, \|\theta_0 - \theta_\mathrm{true}\|$')
    ax.spines['right'].set_visible(True)
    save_paper_figure(fig, out_dir / 'model_error_convergence')
    plt.close(fig)

    # ── 2. final theta(z): recovered vs true vs starting ─────────────────────
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE, layout='constrained')
    z_true = np.repeat([interfaces_y[0] + 1e-3] + list(interfaces_y) + [interfaces_y[-1] - 1e-3], 2)[1:-1]
    t_step = np.repeat([background_deg] + list(layer_angles_deg) + [background_deg], 2)
    ax.plot(t_step, z_true * MM, color=C_TRUE, lw=1.0, ls='--', label='True')
    ax.plot(profiles[0], z_lv * MM, color='0.5', lw=0.8, ls=':', label='Starting')
    ax.plot(final, z_lv * MM, color=C_SYN, lw=1.0, label='Recovered')
    for y_if in interfaces_y:
        ax.axhline(y_if * MM, color='0.6', lw=0.4, ls=':', zorder=1)
    ax.set_xlabel(r'$\theta$ (deg)'); ax.set_ylabel('$z$ (mm)')
    ax.set_ylim(interfaces_y[-1] * MM - 0.3, interfaces_y[0] * MM + 0.3)
    ax.legend(frameon=False, loc='best'); ax.grid(True, **GRID_KW)
    save_paper_figure(fig, out_dir / 'model_error_profile')
    plt.close(fig)

    # ── 3. per-layer recovery ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE, layout='constrained')
    k = np.arange(1, len(layer_rows) + 1)
    ax.bar(k - 0.18, [r[2] for r in layer_rows], width=0.36, color=C_TRUE, alpha=0.8, label='True')
    ax.bar(k + 0.18, [r[3] for r in layer_rows], width=0.36, color=C_SYN, alpha=0.8,
           yerr=[r[4] for r in layer_rows], error_kw=dict(lw=0.6, capsize=2),
           label=r'Recovered (mean $\pm$ std)')
    ax.set_xticks(k, [f'L{i}' for i in k])
    ax.set_xlabel('Layer (top to bottom)'); ax.set_ylabel(r'$\theta$ (deg)')
    ax.set_ylim(0, 90)
    ax.legend(frameon=False, loc='upper right'); ax.yaxis.grid(True, **GRID_KW)
    save_paper_figure(fig, out_dir / 'model_error_layers')
    plt.close(fig)

    return dict(rms=rms_all, rms_interior=rms_int, relative=rel_err, mae=mae, max=max_err,
                corr=corr, layers=layer_rows, interfaces=iface)
