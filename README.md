Bioelectric State Space: Synthetic Demonstration
Code accompanying:

An Operational Bioelectric State Space with Fisher--Rao Geometric Structure
Anderson M. Rodriguez (2026) BioSystems (under revision)

SCRIPT OPERATION
Simulates bioelectric tissue — 32×32 cell grid with FitzHugh-Nagumo dynamics and gap-junctional (diffusive) coupling
Runs six perturbation conditions — excitable, oscillatory, and high-drive regimes at weak and strong coupling (100 replicates each, 600 simulations total)
Extracts state-space coordinates — four scalars per recording: baseline voltage (V₀), pattern amplitude (A_δV), characteristic temporal scale (T), effective connectivity (C_eff)
Computes the Fisher-Rao metric — empirical covariance of coordinates across replicates, estimated per-condition and pooled
Tests V₀ exclusion — verifies all conditions remain separable using only (A_δV, T, C_eff)
Generates six figures — PDF and PNG at 300 DPI
(Optional) Allen Brain Atlas demo — fetches 100 neurons from the Cell Types Database and maps electrophysiology features to state-space coordinates
Quick start
pip install -r requirements.txt
python bioelectric_state_space_synthetic.py
This runs the core simulation and produces all figures in ./figures/. Takes several minutes on a modern laptop (time will vary but likely it will run in less than 1 hour).

To include the Allen Brain Atlas API demo:

python bioelectric_state_space_synthetic.py --with-api
This requires internet access and the requests package. If the API is unreachable, the script falls back to synthetic neuronal data automatically.

Output
All files are written to ./figures/:

File	Description
fig1_phase_portraits.pdf	Single-cell phase portraits for regime verification
fig2_voltage_fields.pdf	Voltage snapshots across six conditions
fig3_time_series.pdf	Spatial-mean voltage time series (representative replicate)
fig4_morphospace.pdf	Pairwise state-space coordinate projections
fig5_fisher_rao.pdf	Per-condition Fisher-Rao and pooled covariance correlation matrices
numerical_results.json	All coordinate statistics, per-condition covariance, eigenvalues, bootstrap 95% CIs, pooled metric, V₀ exclusion test
With --with-api (Allen Brain Atlas demo):

File	Description
fig6_allen_api.pdf	Allen Brain Atlas state-space embedding
allen_api_results.json	Allen data coordinates and metric
Without --with-api, or if the API is unreachable during --with-api:

File	Description
fig_supp_neuronal_fallback.pdf	Synthetic neuronal ephys state-space embedding
PNG versions (300 DPI) are generated alongside each PDF.

Reproducibility
All random number generation uses numpy.random.default_rng with deterministic seeds:

Tissue simulations: seed = 42 + condition_index * 1000 + replicate_number
Neuronal fallback data: seed = 99
Results are bitwise identical across runs on the same NumPy version and platform. The underlying PCG64 generator is stable across NumPy ≥1.17, but minor floating-point differences may occur across architectures or major NumPy versions.

Allen Brain Atlas API results depend on the database state at query time and are not guaranteed to be identical across runs, though the pipeline output is deterministic given the same input records. The fallback synthetic neuronal data is fully reproducible.

Perturbation conditions
Label	Coupling (C)	Drive (I_ext)	Recovery (ε)	a	Regime
Exc-WC	0.08	0.20	0.04	0.7	Excitable, weak coupling
Exc-SC	1.20	0.20	0.04	0.7	Excitable, strong coupling
Osc-WC	0.08	0.50	0.08	0.7	Oscillatory, weak coupling
Osc-SC	1.20	0.50	0.08	0.7	Oscillatory, strong coupling
HD-WC	0.08	0.80	0.12	0.5	High-drive, weak coupling
HD-SC	1.20	0.80	0.12	0.5	High-drive, strong coupling
Model equations
FitzHugh-Nagumo with diffusive coupling and noise:

dV/dt = V - V³/3 - w + I_ext + C ∇²V + σ ξ(x,t)
dw/dt = ε (V + a - bw)
32×32 grid, periodic boundary conditions, Euler-Maruyama integration (dt = 0.02, T_warmup = 100.0, T_record = 120.0, 600 snapshots recorded). Noise amplitude fixed at σ = 0.05 across all replicates.

Coordinate extraction
V₀: spatiotemporal mean voltage
A_δV: RMS spatial deviation from instantaneous spatial mean
T: autocorrelation 1/e decay time (with linear interpolation for sub-sample precision)
C_eff: effective coupling from least-squares fit of ∂V/∂t ≈ C ∇²V (absolute value)
Allen Brain Atlas mapping
Electrophysiology features from the Cell Types Database are mapped to state-space coordinates:

State-space coordinate	Allen feature	Units
V₀	Resting potential (v_rest)	mV
A_δV	|Peak − Trough| voltage	mV
T	Membrane time constant (τ)	ms
C_eff	Inverse input resistance (1/Ri)	1/MΩ
Dependencies
Python ≥ 3.8
NumPy ≥ 1.17
SciPy ≥ 1.4
Matplotlib ≥ 3.2
Requests ≥ 2.20 (optional, for --with-api)
License
MIT

Citation
@article{rodriguez2026statespace,
  title={An Operational Bioelectric State Space with Fisher--Rao Geometric Structure},
  author={Rodriguez, Anderson M.},
  journal={BioSystems},
  year={2026},
  note={Under revision}
}
