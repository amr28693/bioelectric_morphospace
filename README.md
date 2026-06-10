# Bioelectric Morphospace: Synthetic Demonstration

Code accompanying:

> **An Operational Bioelectric Morphospace with Fisher--Rao Geometric Structure**
> Anderson M. Rodriguez (2026)
> *BioSystems* (submitted)

## SCRIPT OPERATION

1. **Simulates bioelectric tissue** — 32×32 cell grid with FitzHugh-Nagumo dynamics and gap-junctional (diffusive) coupling
2. **Runs six perturbation conditions** — excitable, oscillatory, and bistable regimes at weak and strong coupling (15 replicates each, 90 simulations total)
3. **Extracts morphospace coordinates** — four scalars per recording: baseline voltage (V₀), pattern amplitude (A_δV), characteristic temporal scale (T), effective connectivity (C_eff)
4. **Computes the Fisher-Rao metric** — empirical covariance of coordinates across replicates, estimated per-condition and pooled
5. **Defines attractor basins** — KL-divergence free energy from two target attractors, with basin assignment via the covariant matrix
6. **Tests V₀ exclusion** — verifies all conditions remain separable using only (A_δV, T, C_eff)
7. **Generates five figures** — PDF and PNG at 300 DPI (four in manuscript, one supplementary bar chart)
8. **(Optional) Allen Brain Atlas demo** — fetches 100 neurons from the Cell Types Database and maps electrophysiology features to morphospace coordinates

## Quick start

```bash
pip install -r requirements.txt
python morphospace_synthetic.py
```

This runs the core simulation and produces all figures in `./figures/`. Takes ~2 minutes on a modern laptop.

To include the Allen Brain Atlas API demo:

```bash
python morphospace_synthetic.py --with-api
```

This requires internet access and the `requests` package. If the API is unreachable, the script falls back to synthetic neuronal data automatically.

## Output

All files are written to `./figures/`:

| File | Description |
|------|-------------|
| `fig1_voltage_fields.pdf` | Voltage snapshots across six conditions |
| `fig2_morphospace.pdf` | Pairwise morphospace coordinate projections |
| `fig3_fisher_rao.pdf` | Per-condition Fisher-Rao and pooled covariance correlation matrices |
| `figsupp-bar_summary.pdf` | Bar charts of coordinate means ± SD (supplementary) |
| `numerical_results.json` | All coordinate statistics, pooled metric, V₀ exclusion test |

With `--with-api` (Allen Brain Atlas demo):

| File | Description |
|------|-------------|
| `fig4_allen_api.pdf` | Allen Brain Atlas morphospace embedding |
| `allen_api_results.json` | Allen data coordinates and metric |

Without `--with-api`, or if the API is unreachable during `--with-api`:

| File | Description |
|------|-------------|
| `fig_supp_neuronal_fallback.pdf` | Synthetic neuronal ephys morphospace embedding |

PNG versions (300 DPI) are generated alongside each PDF.

## Reproducibility

All random number generation uses `numpy.random.default_rng` with deterministic seeds:

- **Tissue simulations:** `seed = 42 + condition_index * 1000 + replicate_number`
- **Neuronal fallback data:** `seed = 99`

Results are bitwise identical across runs on the same NumPy version and platform. The underlying PCG64 generator is stable across NumPy ≥1.17, but minor floating-point differences may occur across architectures or major NumPy versions.

Allen Brain Atlas API results depend on the database state at query time and are not guaranteed to be identical across runs, though the pipeline output is deterministic given the same input records. The fallback synthetic neuronal data is fully reproducible.

## Perturbation conditions

| Label | Coupling (C) | Drive (I_ext) | Recovery (ε) | a | Regime |
|-------|-------------|---------------|-------------|-----|--------|
| Exc-WC | 0.08 | 0.20 | 0.04 | 0.7 | Excitable, weak coupling |
| Exc-SC | 1.20 | 0.20 | 0.04 | 0.7 | Excitable, strong coupling |
| Osc-WC | 0.08 | 0.50 | 0.08 | 0.7 | Oscillatory, weak coupling |
| Osc-SC | 1.20 | 0.50 | 0.08 | 0.7 | Oscillatory, strong coupling |
| Bi-WC | 0.08 | 0.80 | 0.12 | 0.5 | Bistable, weak coupling |
| Bi-SC | 1.20 | 0.80 | 0.12 | 0.5 | Bistable, strong coupling |

## Model equations

FitzHugh-Nagumo with diffusive coupling and noise:

```
dV/dt = V - V³/3 - w + I_ext + C ∇²V + σ ξ(x,t)
dw/dt = ε (V + a - bw)
```

32×32 grid, periodic boundary conditions, Euler-Maruyama integration (dt = 0.02, T_warmup = 25.0, T_record = 40.0, 200 snapshots recorded). Noise amplitude varies across replicates: σ = 0.06 + 0.04 sin(rep × 1.1).

## Coordinate extraction

- **V₀**: spatiotemporal mean voltage
- **A_δV**: RMS spatial deviation from instantaneous spatial mean
- **T**: autocorrelation 1/e decay time (with linear interpolation for sub-sample precision)
- **C_eff**: effective coupling from least-squares fit of ∂V/∂t ≈ C ∇²V (absolute value)

## Allen Brain Atlas mapping

Electrophysiology features from the [Cell Types Database](https://celltypes.brain-map.org/) are mapped to morphospace coordinates:

| Morphospace coordinate | Allen feature | Units |
|----------------------|---------------|-------|
| V₀ | Resting potential (v_rest) | mV |
| A_δV | \|Peak − Trough\| voltage | mV |
| T | Membrane time constant (τ) | ms |
| C_eff | Inverse input resistance (1/Ri) | 1/MΩ |

## Dependencies

- Python ≥ 3.8
- NumPy ≥ 1.17
- SciPy ≥ 1.4
- Matplotlib ≥ 3.2
- Requests ≥ 2.20 (optional, for `--with-api`)

## License

MIT

## Citation

```bibtex
@article{rodriguez2026morphospace,
  title={An Operational Bioelectric Morphospace with Fisher--Rao Geometric Structure},
  author={Rodriguez, Anderson M.},
  journal={BioSystems},
  year={2026},
  note={Submitted}
}
```
