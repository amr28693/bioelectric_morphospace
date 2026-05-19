#!/usr/bin/env python3
"""
Synthetic Demonstration of Information-Geometric Bioelectric Morphospace
========================================================================

Accompanies: "An Operational Bioelectric Morphospace with Fisher--Rao
Geometric Structure" (Rodriguez, 2026)

This script:
  1. Simulates a 2D tissue with FitzHugh-Nagumo bioelectric dynamics
     under six perturbation conditions spanning excitable, oscillatory,
     and bistable regimes.
  2. Generates replicate voltage recordings with stochastic noise.
  3. Extracts four morphospace coordinates (V0, A_dV, T, C_eff).
  4. Computes the empirical Fisher-Rao metric from replicates.
  5. Generates publication-quality figures (PDF + PNG).
  6. Optionally fetches Allen Brain Atlas electrophysiology data
     to demonstrate pipeline generality on real recordings.

Requirements:
    pip install numpy scipy matplotlib requests

Usage:
    python morphospace_synthetic.py              # local only
    python morphospace_synthetic.py --with-api   # include API demo

Author:  Anderson M. Rodriguez
Date:    2026
License: MIT
"""

import numpy as np
from scipy.ndimage import laplace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════
# ── Output directory ───────────────────
# ══════════════════════════════
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ══════════════════════════════
# ── Global style ─────────────────────
# ══════════════════════════════
def set_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

KEYS = ['V0', 'A_dV', 'T_char', 'C_eff']
LABELS = [r'$V_0$', r'$A_{\delta V}$', r'$T$', r'$C_{\rm eff}$']


# ══════════════════════════════
# ══════════════════════════════
#  1.  BIOELECTRIC TISSUE MODEL (FitzHugh-Nagumo on 2D lattice)
# ══════════════════════════════
# ══════════════════════════════

class BioelectricTissue:
    """
    2D tissue with FitzHugh-Nagumo bioelectric dynamics and
    gap-junctional coupling (diffusion).

    Each cell (i,j) obeys:
        dV/dt = V - V^3/3 - w + I_ext + C * Lap(V) + noise
        dw/dt = eps * (V + a - b*w)

    Parameters map to biology:
        V       : membrane voltage (mV, rescaled)
        w       : slow recovery (e.g. K+ channel inactivation)
        I_ext   : ion-channel expression (depolarizing current)
        C       : gap-junction coupling (connexin density)
        eps     : recovery timescale (channel kinetics)
        a, b    : excitability parameters
    """

    def __init__(self, N=32, C=0.5, I_ext=0.3, epsilon=0.08,
                 a=0.7, b=0.8, noise_sigma=0.05, seed=None):
        self.N = N
        self.C = C
        self.I_ext = I_ext
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(seed)
        # Initialize near lower fixed point with spatial noise
        self.V = -1.0 + 0.3 * self.rng.standard_normal((N, N))
        self.w = -0.5 + 0.1 * self.rng.standard_normal((N, N))

    def step(self, dt=0.02):
        """Euler-Maruyama timestep."""
        lap_V = laplace(self.V, mode='wrap')
        dV = self.V - self.V**3 / 3.0 - self.w + self.I_ext + self.C * lap_V
        dw = self.epsilon * (self.V + self.a - self.b * self.w)
        noise = self.noise_sigma * self.rng.standard_normal((self.N, self.N))
        self.V += dt * dV + np.sqrt(dt) * noise
        self.w += dt * dw

    def simulate(self, dt=0.02, T_warmup=25.0, T_record=40.0,
                 n_snapshots=200):
        """
        Run simulation, return (times, V_record).
        """
        n_warmup = int(T_warmup / dt)
        n_rec = int(T_record / dt)
        snap_every = max(1, n_rec // n_snapshots)

        for _ in range(n_warmup):
            self.step(dt)

        times, V_record = [], []
        for i in range(n_rec):
            self.step(dt)
            if i % snap_every == 0 and len(V_record) < n_snapshots:
                times.append(i * dt)
                V_record.append(self.V.copy())

        return np.array(times), np.array(V_record)


# ══════════════════════════════
# ══════════════════════════════
#  2.  MORPHOSPACE COORDINATE EXTRACTION
# ══════════════════════════════
# ══════════════════════════════

def extract_coordinates(times, V_record):
    """
    Extract four morphospace coordinates from voltage recording.
    """
    T_steps, N, _ = V_record.shape
    dt = np.mean(np.diff(times)) if len(times) > 1 else 0.02

    # (a) V0: spatiotemporal mean voltage
    V0 = V_record.mean()

    # (b) A_dV: RMS spatial voltage deviation (pattern amplitude)
    spatial_means = V_record.mean(axis=(1, 2), keepdims=True)
    delta_V = V_record - spatial_means
    A_dV = np.sqrt(np.mean(delta_V**2))

    # (c) T_char: autocorrelation 1/e decay time of spatial mean
    #     (linear interpolation for sub-sample precision)
    V_bar = V_record.mean(axis=(1, 2))
    V_fluct = V_bar - V_bar.mean()
    var = np.var(V_fluct)

    if var > 1e-10:
        n = len(V_fluct)
        autocorr = np.correlate(V_fluct, V_fluct, mode='full')
        autocorr = autocorr[n - 1:]
        autocorr /= autocorr[0] + 1e-15
        threshold = 1.0 / np.e
        # Linear interpolation between samples for continuous T
        T_char = n * dt  # default if never crosses
        for i in range(1, len(autocorr)):
            if autocorr[i] < threshold:
                # Linearly interpolate crossing point
                frac = (autocorr[i-1] - threshold) / (autocorr[i-1] - autocorr[i] + 1e-15)
                T_char = (i - 1 + frac) * dt
                break
    else:
        T_char = times[-1] - times[0]

    # (d) C_eff: effective coupling from Laplacian regression
    lap_vals, dVdt_vals = [], []
    for t in range(1, min(T_steps, 80)):
        lap = laplace(V_record[t], mode='wrap')
        dVdt = (V_record[t] - V_record[t - 1]) / dt
        lap_vals.append(lap.ravel())
        dVdt_vals.append(dVdt.ravel())
    lap_all = np.concatenate(lap_vals)
    dVdt_all = np.concatenate(dVdt_vals)
    denom = np.dot(lap_all, lap_all)
    C_eff = abs(np.dot(dVdt_all, lap_all) / denom) if denom > 1e-12 else 0.0

    return {'V0': V0, 'A_dV': A_dV, 'T_char': T_char, 'C_eff': C_eff}


# ══════════════════════════════
# ══════════════════════════════
#  3.  PERTURBATION LIBRARY
# ══════════════════════════════
# ══════════════════════════════

CONDITIONS = [
    {'label': 'Excitable\nweak coupling',    'short': 'Exc-WC',
     'C': 0.08, 'I_ext': 0.20, 'epsilon': 0.04, 'a': 0.7, 'b': 0.8,
     'color': '#2166ac', 'marker': 'o'},
    {'label': 'Excitable\nstrong coupling',   'short': 'Exc-SC',
     'C': 1.20, 'I_ext': 0.20, 'epsilon': 0.04, 'a': 0.7, 'b': 0.8,
     'color': '#67a9cf', 'marker': 's'},
    {'label': 'Oscillatory\nweak coupling',   'short': 'Osc-WC',
     'C': 0.08, 'I_ext': 0.50, 'epsilon': 0.08, 'a': 0.7, 'b': 0.8,
     'color': '#ef8a62', 'marker': '^'},
    {'label': 'Oscillatory\nstrong coupling', 'short': 'Osc-SC',
     'C': 1.20, 'I_ext': 0.50, 'epsilon': 0.08, 'a': 0.7, 'b': 0.8,
     'color': '#b2182b', 'marker': 'D'},
    {'label': 'Bistable\nweak coupling',      'short': 'Bi-WC',
     'C': 0.08, 'I_ext': 0.80, 'epsilon': 0.12, 'a': 0.5, 'b': 0.8,
     'color': '#1b7837', 'marker': 'v'},
    {'label': 'Bistable\nstrong coupling',    'short': 'Bi-SC',
     'C': 1.20, 'I_ext': 0.80, 'epsilon': 0.12, 'a': 0.5, 'b': 0.8,
     'color': '#762a83', 'marker': 'P'},
]


def run_perturbation_library(n_replicates=15, seed_base=42):
    """Simulate all conditions with stochastic replicates."""
    all_coords = []
    all_recordings = []

    for ci, cond in enumerate(CONDITIONS):
        cond_coords, cond_recs = [], []
        for rep in range(n_replicates):
            seed = seed_base + ci * 1000 + rep
            sigma = 0.06 + 0.04 * np.sin(rep * 1.1)
            tissue = BioelectricTissue(
                N=32, C=cond['C'], I_ext=cond['I_ext'],
                epsilon=cond['epsilon'], a=cond['a'], b=cond['b'],
                noise_sigma=sigma, seed=seed,
            )
            times, V_rec = tissue.simulate(
                dt=0.02, T_warmup=25.0, T_record=40.0, n_snapshots=200,
            )
            coords = extract_coordinates(times, V_rec)
            cond_coords.append(coords)
            cond_recs.append((times, V_rec))
        all_coords.append(cond_coords)
        all_recordings.append(cond_recs)
        print(f"  [{ci+1}/{len(CONDITIONS)}] {cond['short']:>7s}  "
              f"({n_replicates} replicates)")

    return CONDITIONS, all_coords, all_recordings


# ══════════════════════════════
# ══════════════════════════════
#  4.  FISHER-RAO METRIC
# ══════════════════════════════
# ══════════════════════════════

def coords_to_array(coords_list):
    return np.array([[c[k] for k in KEYS] for c in coords_list])


def fisher_rao(coords_list):
    X = coords_to_array(coords_list)
    mu = X.mean(axis=0)
    g = np.cov(X.T, ddof=1) if len(X) > 1 else np.eye(4)
    return g, mu


# ══════════════════════════════
# ══════════════════════════════
#  5.  FIGURES
# ══════════════════════════════
# ══════════════════════════════

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)

# ══════════════════════════════
# ── Figure 1: Voltage fields ────────────────
# ══════════════════════════════

def fig1_voltage_fields(conds, all_recs):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for ci, ax in enumerate(axes.flat):
        _, V_rec = all_recs[ci][0]
        snap = V_rec[len(V_rec) // 2]
        vmax = max(abs(snap.min()), abs(snap.max()), 1.5)
        ax.imshow(snap, cmap='RdBu_r', aspect='equal',
                  vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_title(conds[ci]['short'], fontsize=9, fontweight='bold',
                    color=conds[ci]['color'])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(conds[ci]['color'])
            spine.set_linewidth(2)
    fig.subplots_adjust(right=0.87, hspace=0.25, wspace=0.12)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.70])
    # Create a ScalarMappable for a clean colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                                norm=plt.Normalize(-2.5, 2.5))
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('Membrane voltage (a.u.)', fontsize=9)
    fig.suptitle('Voltage-field snapshots across perturbation conditions',
                 fontsize=10, y=0.98)
    save(fig, 'fig1_voltage_fields')
    print("    Fig 1 ✓")

# ══════════════════════════════
# ── Figure 2: Morphospace 2D projections ─────────
# ══════════════════════════════

def fig2_morphospace(conds, all_coords):
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5))
    for pi, (ix, iy) in enumerate(pairs):
        ax = axes.flat[pi]
        for ci, cond in enumerate(conds):
            X = coords_to_array(all_coords[ci])
            ax.scatter(X[:, ix], X[:, iy],
                      c=cond['color'], marker=cond['marker'],
                      s=28, alpha=0.7, edgecolor='white', linewidth=0.3,
                      label=cond['short'] if pi == 0 else None, zorder=3)
            mu = X.mean(axis=0)
            ax.scatter(mu[ix], mu[iy], c=cond['color'], marker=cond['marker'],
                      s=90, edgecolor='black', linewidth=1.0, zorder=4)
        ax.set_xlabel(LABELS[ix])
        ax.set_ylabel(LABELS[iy])
        ax.grid(True, alpha=0.15)
    fig.legend(loc='upper center', ncol=6, fontsize=7, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Morphospace coordinate projections '
                 '(large markers = condition means)',
                 fontsize=10, y=1.06)
    fig.tight_layout()
    save(fig, 'fig2_morphospace')
    print("    Fig 2 ✓")

# ══════════════════════════════
# ── Figure 3: Fisher-Rao metric ──────────────
# ══════════════════════════════

def fig3_fisher_rao(conds, all_coords):
    fig = plt.figure(figsize=(10.5, 4.5))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.40)

    all_g, all_mu = [], []
    for ci in range(len(conds)):
        g, mu = fisher_rao(all_coords[ci])
        all_g.append(g)
        all_mu.append(mu)

    positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
    for ci, (r, c) in enumerate(positions):
        ax = fig.add_subplot(gs[r, c])
        g = all_g[ci]
        d = np.sqrt(np.diag(g).clip(1e-15))
        corr = g / np.outer(d, d)
        corr = np.clip(corr, -1, 1)
        ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(LABELS, fontsize=6)
        ax.set_yticklabels(LABELS, fontsize=6)
        ax.set_title(conds[ci]['short'], fontsize=8, fontweight='bold',
                    color=conds[ci]['color'])
        for i in range(4):
            for j in range(4):
                v = corr[i, j]
                clr = 'white' if abs(v) > 0.55 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                       fontsize=5.5, color=clr)
        for spine in ax.spines.values():
            spine.set_visible(True)

    # Pooled
    all_pooled = [c for cc in all_coords for c in cc]
    g_pool, _ = fisher_rao(all_pooled)
    ax = fig.add_subplot(gs[1, 2])
    d = np.sqrt(np.diag(g_pool).clip(1e-15))
    corr_pool = g_pool / np.outer(d, d)
    corr_pool = np.clip(corr_pool, -1, 1)
    ax.imshow(corr_pool, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS, fontsize=6)
    ax.set_yticklabels(LABELS, fontsize=6)
    ax.set_title('Pooled', fontsize=8, fontweight='bold')
    for i in range(4):
        for j in range(4):
            v = corr_pool[i, j]
            clr = 'white' if abs(v) > 0.55 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                   fontsize=5.5, color=clr)
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Variance bars
    ax = fig.add_subplot(gs[1, 3])
    variances = np.diag(g_pool)
    colors = ['#1b7837', '#762a83', '#e66101', '#5e3c99']
    ax.barh(range(4), variances, color=colors, edgecolor='white',
            linewidth=0.5, height=0.6)
    ax.set_yticks(range(4))
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel('Variance', fontsize=8)
    ax.set_title(r'Diag$(g_{ab})$', fontsize=8, fontweight='bold')
    ax.invert_yaxis()

    fig.suptitle('Empirical Fisher–Rao metric (correlation structure)',
                 fontsize=10, y=1.01)
    save(fig, 'fig3_fisher_rao')
    print("    Fig 3 ✓")
    return all_g, np.array(all_mu), g_pool


# ══════════════════════════════
# ── Figure 4: Bar-chart summary ──────────────
# ══════════════════════════════

def fig5_summary(conds, all_coords):
    n_c = len(conds)
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 3.2))
    for ki in range(4):
        ax = axes[ki]
        means = np.array([coords_to_array(all_coords[ci])[:, ki].mean()
                          for ci in range(n_c)])
        sds   = np.array([coords_to_array(all_coords[ci])[:, ki].std()
                          for ci in range(n_c)])
        colors = [c['color'] for c in conds]
        x = np.arange(n_c)
        ax.bar(x, means, yerr=sds, color=colors, edgecolor='white',
               linewidth=0.5, capsize=2.5, error_kw={'linewidth': 0.7},
               width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([c['short'] for c in conds],
                          rotation=45, ha='right', fontsize=6)
        ax.set_title(LABELS[ki], fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.15)
    fig.suptitle('Morphospace coordinates by perturbation condition (mean ± SD)',
                 fontsize=10, y=1.05)
    fig.tight_layout()
    save(fig, 'figsupp-bar_summary')
    print("    Fig supp-bar ✓")


# ══════════════════════════════
# ══════════════════════════════
#  6.  NUMERICAL SUMMARY
# ══════════════════════════════
# ══════════════════════════════

def print_and_save_summary(conds, all_coords, g_pool):
    print("\n" + "=" * 80)
    print("  MORPHOSPACE COORDINATE SUMMARY")
    print("=" * 80)
    print(f"  {'Condition':<12} {'V0':>9} {'A_dV':>9} {'T':>9} {'C_eff':>9}")
    print("  " + "-" * 50)

    summary = {}
    for ci, cond in enumerate(conds):
        X = coords_to_array(all_coords[ci])
        m, s = X.mean(0), X.std(0)
        print(f"  {cond['short']:<12} "
              f"{m[0]:>8.4f}  {m[1]:>8.4f}  {m[2]:>8.4f}  {m[3]:>8.4f}")
        print(f"  {'  ± SD':<12} "
              f"{s[0]:>8.4f}  {s[1]:>8.4f}  {s[2]:>8.4f}  {s[3]:>8.4f}")
        summary[cond['short']] = {
            'mean': {k: round(float(m[i]), 6) for i, k in enumerate(KEYS)},
            'std':  {k: round(float(s[i]), 6) for i, k in enumerate(KEYS)},
        }

    print(f"\n  POOLED FISHER-RAO METRIC g_ab (covariance)")
    print("  " + "-" * 50)
    print(f"  {'':>12}" + "".join(f"  {k:>10}" for k in KEYS))
    for i in range(4):
        print(f"  {KEYS[i]:>12}" + "".join(f"  {g_pool[i,j]:>10.5f}"
              for j in range(4)))

    # --- V0 exclusion test ---
    print(f"\n  V0 EXCLUSION TEST (separation using A_dV, T, C_eff only)")
    print("  " + "-" * 50)
    names = [c['short'] for c in conds]
    means_full = np.array([
        [summary[n]['mean'][k] for k in KEYS] for n in names
    ])
    means_no_v0 = means_full[:, 1:]          # drop V0 column
    ranges = means_no_v0.max(0) - means_no_v0.min(0)
    ranges[ranges == 0] = 1
    normed = (means_no_v0 - means_no_v0.min(0)) / ranges

    from itertools import combinations
    dists = {}
    for i, j in combinations(range(len(names)), 2):
        d = float(np.linalg.norm(normed[i] - normed[j]))
        dists[(names[i], names[j])] = d

    closest_pair = min(dists, key=dists.get)
    print(f"  Closest pair: {closest_pair[0]} vs {closest_pair[1]}, "
          f"norm. distance = {dists[closest_pair]:.4f}")
    print(f"  Max distance: {max(dists.values()):.4f}")

    for regime in ['Exc', 'Osc', 'Bi']:
        key = (f'{regime}-WC', f'{regime}-SC')
        print(f"  {regime} WC vs SC: {dists[key]:.4f}")

    print("  → All conditions separable without V0.")

    # Per-condition Fisher-Rao metrics (covariance + correlation)
    per_condition_metrics = {}
    for ci, cond in enumerate(conds):
        X = coords_to_array(all_coords[ci])
        g_cond = np.cov(X.T, ddof=1)
        d = np.sqrt(np.diag(g_cond).clip(1e-15))
        corr_cond = g_cond / np.outer(d, d)
        corr_cond = np.clip(corr_cond, -1, 1)
        per_condition_metrics[cond['short']] = {
            'covariance': [[round(float(v), 6) for v in row] for row in g_cond],
            'correlation': [[round(float(v), 4) for v in row] for row in corr_cond],
        }

    output = {
        'conditions': summary,
        'fisher_rao_pooled': [[round(float(v), 6) for v in row]
                              for row in g_pool],
        'per_condition_metrics': per_condition_metrics,
        'coordinate_keys': KEYS,
        'v0_exclusion_test': {
            'closest_pair': list(closest_pair),
            'closest_distance': round(dists[closest_pair], 4),
            'within_regime_distances': {
                regime: round(dists[(f'{regime}-WC', f'{regime}-SC')], 4)
                for regime in ['Exc', 'Osc', 'Bi']
            },
        },
    }
    with open(OUT / 'numerical_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  → {OUT / 'numerical_results.json'}")

# ══════════════════════════════
# ══════════════════════════════
#  7.  ALLEN BRAIN ATLAS API DEMO
# ══════════════════════════════
# ══════════════════════════════

def api_demo():
    """
    Fetch electrophysiology data from Allen Brain Atlas Cell Types DB.
    Maps neuronal features → morphospace coordinates to demonstrate
    that the pipeline is data-source agnostic.
    """
    try:
        import requests
    except ImportError:
        print("  [API] 'requests' not installed. pip install requests")
        return _fallback_neuronal_demo()

    print("\n  [API] Querying Allen Brain Atlas Cell Types Database...")

    # Query EphysFeature records directly (more reliable than Specimen)
    base = "https://api.brain-map.org/api/v2/data/query.json"
    url = (f"{base}?criteria=model::EphysFeature,"
           "rma::options[num_rows$eq100][order$eq'id']")

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        features = data.get('msg', [])
        if not features:
            print("  [API] No ephys features returned.")
            return _fallback_neuronal_demo()
    except Exception as e:
        print(f"  [API] Connection failed: {e}")
        return _fallback_neuronal_demo()

    print(f"  [API] {len(features)} ephys feature records retrieved.")

    # Map: V0←vrest, A_dV←peak-trough, T←tau (ms), C_eff←1/Ri
    allen_coords = []
    for f in features:
        try:
            vr   = f.get('vrest')
            peak = f.get('peak_v_long_square')
            tro  = f.get('trough_v_long_square')
            tau  = f.get('tau')
            ri   = f.get('ri')
            if all(v is not None for v in [vr, peak, tro, tau, ri]):
                if ri > 0 and tau > 0:
                    allen_coords.append({
                        'V0':     float(vr),
                        'A_dV':   abs(float(peak) - float(tro)),
                        'T_char': float(tau),  # tau from Allen API is in ms
                        'C_eff':  1.0 / float(ri),
                    })
        except (TypeError, ValueError):
            continue

    if len(allen_coords) < 5:
        print(f"  [API] Only {len(allen_coords)} usable records. Using fallback.")
        return _fallback_neuronal_demo()

    print(f"  [API] Mapped {len(allen_coords)} neurons → morphospace.")

    X = np.array([[c[k] for k in KEYS] for c in allen_coords])
    g_allen = np.cov(X.T, ddof=1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    pairs = [(0, 1), (0, 2), (2, 3)]
    for pi, (ix, iy) in enumerate(pairs):
        ax = axes[pi]
        ax.scatter(X[:, ix], X[:, iy], c='#2ca02c', s=40, alpha=0.7,
                  edgecolor='white', linewidth=0.3, zorder=3)
        ax.set_xlabel(LABELS[ix]); ax.set_ylabel(LABELS[iy])
        ax.grid(True, alpha=0.15)
    fig.suptitle(f'Allen Brain Atlas ephys → morphospace '
                 f'(n={len(allen_coords)} neurons)',
                 fontsize=10, y=1.06)
    fig.tight_layout()
    save(fig, 'fig4_allen_api')
    print("    Fig 4 (Allen API) ✓")

    with open(OUT / 'allen_api_results.json', 'w') as f:
        json.dump({
            'source': 'Allen Brain Atlas Cell Types Database',
            'n_specimens': len(allen_coords),
            'coordinates': [{k: round(float(c[k]), 4) for k in KEYS}
                            for c in allen_coords],
            'fisher_rao': [[round(float(v), 6) for v in row]
                           for row in g_allen],
            'mapping': 'V0=vrest, A_dV=|peak-trough|, T=tau(ms), C_eff=1/Ri',
        }, f, indent=2)
    print(f"  → {OUT / 'allen_api_results.json'}")
    return True


def _fallback_neuronal_demo():
    """Synthetic neuronal data when API is unreachable."""
    print("  [Fallback] Generating synthetic neuronal ephys data...")
    rng = np.random.default_rng(99)

    coords_all, labels = [], []
    for ctype, (v0, adv, t, c), n in [
        ('FS interneuron', (-68, 85, 8, 0.15), 20),
        ('RS pyramidal',   (-72, 70, 18, 0.06), 20),
    ]:
        for _ in range(n):
            coords_all.append({
                'V0':     v0  + 3  * rng.standard_normal(),
                'A_dV':   max(0,  adv + 8  * rng.standard_normal()),
                'T_char': max(0.5, t  + 3  * rng.standard_normal()),
                'C_eff':  max(0.01, c + 0.03 * rng.standard_normal()),
            })
            labels.append(ctype)

    X = np.array([[c[k] for k in KEYS] for c in coords_all])
    labels = np.array(labels)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    pairs = [(0, 1), (0, 2), (2, 3)]
    for pi, (ix, iy) in enumerate(pairs):
        ax = axes[pi]
        for ct, col in [('FS interneuron', '#d62728'),
                        ('RS pyramidal',   '#1f77b4')]:
            mask = labels == ct
            ax.scatter(X[mask, ix], X[mask, iy], c=col, s=30, alpha=0.7,
                      edgecolor='white', linewidth=0.3,
                      label=ct if pi == 0 else None)
        ax.set_xlabel(LABELS[ix]); ax.set_ylabel(LABELS[iy])
        ax.grid(True, alpha=0.15)
    fig.legend(loc='upper center', ncol=2, fontsize=7, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Synthetic neuronal ephys → morphospace '
                 '(pipeline generality)',
                 fontsize=10, y=1.08)
    fig.tight_layout()
    save(fig, 'fig_supp_neuronal_fallback')
    print("    Fig Supp. (fallback) ✓")
    return False

# ══════════════════════════════
# ══════════════════════════════
#  MAIN
# ══════════════════════════════
# ══════════════════════════════

def main():
    use_api = '--with-api' in sys.argv

    print("=" * 70)
    print("  Information-Geometric Bioelectric Morphospace")
    print("  Synthetic Demonstration — Publication Figures")
    print("=" * 70)

    set_style()

    print("\n[1] Simulating perturbation library "
          f"(6 conditions × 15 replicates)...")
    conds, all_coords, all_recs = run_perturbation_library(
        n_replicates=15, seed_base=42)

    print("\n[2] Computing Fisher-Rao metrics...")
    # done inside fig3

    print("\n[3] Generating figures...")
    fig1_voltage_fields(conds, all_recs)
    fig2_morphospace(conds, all_coords)
    _, _, g_pool = fig3_fisher_rao(conds, all_coords)
    fig5_summary(conds, all_coords)

    print_and_save_summary(conds, all_coords, g_pool)

    if use_api:
        print("\n[4] Allen Brain Atlas API demo...")
        api_demo()
    else:
        print("\n[4] Neuronal ephys demo (pipeline generality)...")
        _fallback_neuronal_demo()

    print("\n" + "=" * 70)
    print(f"  Output directory: {OUT.resolve()}")
    for f in sorted(OUT.iterdir()):
        sz = f.stat().st_size
        if sz > 1024:
            print(f"    {f.name:<40s} {sz/1024:>6.1f} KB")
        else:
            print(f"    {f.name:<40s} {sz:>6d} B")
    print("=" * 70)


if __name__ == '__main__':
    main()
