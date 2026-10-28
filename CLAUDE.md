# tomography_orientation_2D

Elastic FWI of grain orientation in anisotropic (austenite) layers with Salvus.
Code lives in `tomograhpy/` (sic); run scripts from that folder so `my_code` and
`material_ela_constants` import.

## Figures (rule: every figure in this project)

All figures use the journal (scientific paper) format in `tomograhpy/my_code/plot_style.py`.
Do not define local rcParams, colour constants or save helpers; import them:

```python
from my_code.plot_style import (apply_paper_style, save_paper_figure, plot_orientation_field,
                                add_colorbar, plot_sources_receivers,
                                SINGLE_COL, DOUBLE_COL, MM, C_OBS, C_SYN, C_TRUE, GRID_KW)
apply_paper_style()   # once, before the first figure
```

- **Size**: width `SINGLE_COL` (3.5 in) or `DOUBLE_COL` (7.0 in); never pass `dpi` to `plt.subplots`.
- **Style**: 8 pt text, 7 pt legends, inward ticks, no top/right spines (set by `apply_paper_style`).
- **Labels**: "quantity (unit)" with math symbols: `'$x$ (mm)'`, `'$z$ (mm)'`, `r'Time ($\mu$s)'`,
  `r'$\theta$ (deg)'`, `'$u_z$ (m)'`. Lengths in mm (`MM = 1e3`); the vertical axis is **z**
  (Salvus calls it y / component 'Y').
- **Titles**: none on paper figures (the caption carries it); multi-panel figures get
  left-aligned panel labels `'(a) ...'`, `'(b) ...'` (`set_title(..., loc='left')`).
- **Colours**: observed `C_OBS` (red, dashed), synthetic `C_SYN` (blue), true model `C_TRUE`
  (green, dashed); orientation maps `viridis` 0-90 deg; signed kernels `RdBu` with a symmetric
  limit; grids `GRID_KW`.
- **Fields**: `tripcolor(..., rasterized=True)`; colour bars with `add_colorbar` (same height as
  the axes).
- **Simulation configuration**: plot with `plot_orientation_field(mesh, src, rec, extent=(x0, x1, y0, y1),
  save_path=...)` - orientation map, dashed symmetry-axis segments, source (red down-triangle)
  and receivers (green up-triangle).
- **Saving**: `save_paper_figure(fig, path_without_extension)` writes a 300 dpi PNG only. PDF
  (embedded fonts) is off by default: turn it on with `plot_style.SAVE_PDF = True` (all figures,
  e.g. a final paper run) or `save_paper_figure(..., pdf=True)` (one figure). Do not turn it on
  unless asked. Per-iteration FWI diagnostics may use `plt.savefig` PNG directly, but still
  follow the size/label rules.
- Scripts use the `Agg` backend and `plt.close(fig)`; notebooks call `plt.show()`.

## FWI results (rule: every orientation FWI script)

When an FWI run finishes, it saves the model-error figures with
`my_code.model_error.save_model_error_figures(...)` into `<FWI_IMAGE_DIR>/paper_figures/` (`PAPER_FIG_DIR`, with the other paper figures):

- `model_error_convergence`: RMS theta error vs iteration, one curve, with the relative error
  `||theta - theta_true|| / ||theta_0 - theta_true||` (= RMS / RMS_0) on the right axis. Do not plot
  the data misfit or an interior RMS here.
- `model_error_profile`: recovered vs true vs starting theta(z). Do not plot theta - theta_true.
- `model_error_layers`: mean +- std of theta per layer vs the true value. Do not plot interface-depth
  errors (interior RMS and interface depths are printed to the console only).

Metrics: wrap-aware difference (180-deg periodic), area-weighted by depth-row thickness, inside the
physical x range. `plot_model_error.py` redraws the figures from a saved run.
