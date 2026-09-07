from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = 20260906
N_SIM = 20_000
N_MON = 30
PRE_CHANGE = 10  # first 10 monitoring observations are in control
TRUE_CP_BOUNDARY = PRE_CHANGE  # boundary is between observations 10 and 11
MIN_SEGMENT = 5
RULE_SETS = {
    "Rule 1 only": (1,),
    "Rules 1-3": (1, 2, 3),
    "Rules 1-8": tuple(range(1, 9)),
}
SCENARIOS = {
    "Stable": {"kind": "stable"},
    "Abrupt +1 SD": {"kind": "mean", "shift": 1.0},
    "Abrupt +2 SD": {"kind": "mean", "shift": 2.0},
    "Abrupt +3 SD": {"kind": "mean", "shift": 3.0},
    "Gradual drift to +3 SD": {"kind": "drift", "max_shift": 3.0},
    "SD x2": {"kind": "variance", "sd_mult": 2.0},
}

OUTDIR = Path(__file__).resolve().parent
RESULTS = OUTDIR / "results"
FIGURES = OUTDIR / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def _endpoint_signal_matrix(z: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Return Boolean matrices (replicate x time) showing when each Nelson rule
    first becomes evaluable and is satisfied at that observation.

    Definitions use the classic eight Nelson special-cause tests:
    R1: 1 point > 3 SD from center line
    R2: 9 points in a row on same side of center line
    R3: 6 points in a row all increasing or all decreasing
    R4: 14 points in a row alternating up and down
    R5: 2 of 3 points > 2 SD from center line on same side
    R6: 4 of 5 points > 1 SD from center line on same side
    R7: 15 points in a row within 1 SD of center line
    R8: 8 points in a row outside 1 SD, with points on both sides
    """
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[None, :]
    n, t = z.shape
    out = {r: np.zeros((n, t), dtype=bool) for r in range(1, 9)}

    # R1
    out[1] = np.abs(z) > 3.0

    # R2: 9 on same side (zeros count as neither side)
    for end in range(8, t):
        w = z[:, end - 8:end + 1]
        out[2][:, end] = np.all(w > 0, axis=1) | np.all(w < 0, axis=1)

    # R3: 6 monotonically increasing/decreasing points => 5 same-sign diffs
    dz = np.diff(z, axis=1)
    for end in range(5, t):
        w = dz[:, end - 5:end]
        out[3][:, end] = np.all(w > 0, axis=1) | np.all(w < 0, axis=1)

    # R4: 14 alternating points => 13 successive differences alternate signs
    for end in range(13, t):
        w = dz[:, end - 13:end]
        sign = np.sign(w)
        nonzero = np.all(sign != 0, axis=1)
        alternating = np.all(sign[:, 1:] * sign[:, :-1] < 0, axis=1)
        out[4][:, end] = nonzero & alternating

    # R5: 2 of 3 beyond 2 SD, same side
    for end in range(2, t):
        w = z[:, end - 2:end + 1]
        out[5][:, end] = (np.sum(w > 2.0, axis=1) >= 2) | (np.sum(w < -2.0, axis=1) >= 2)

    # R6: 4 of 5 beyond 1 SD, same side
    for end in range(4, t):
        w = z[:, end - 4:end + 1]
        out[6][:, end] = (np.sum(w > 1.0, axis=1) >= 4) | (np.sum(w < -1.0, axis=1) >= 4)

    # R7: 15 within 1 SD
    for end in range(14, t):
        w = z[:, end - 14:end + 1]
        out[7][:, end] = np.all(np.abs(w) < 1.0, axis=1)

    # R8: 8 outside 1 SD, both sides represented
    for end in range(7, t):
        w = z[:, end - 7:end + 1]
        outside = np.all(np.abs(w) > 1.0, axis=1)
        both = np.any(w > 1.0, axis=1) & np.any(w < -1.0, axis=1)
        out[8][:, end] = outside & both

    return out


def rule_set_signals(z: np.ndarray, rules: Iterable[int]) -> np.ndarray:
    matrices = _endpoint_signal_matrix(z)
    selected = [matrices[r] for r in rules]
    return np.logical_or.reduce(selected)


def first_true_index(mask: np.ndarray, start: int = 0) -> np.ndarray:
    """Return first True index at/after start for each row, or -1 if none."""
    sub = mask[:, start:]
    any_true = np.any(sub, axis=1)
    idx = np.argmax(sub, axis=1) + start
    return np.where(any_true, idx, -1)


def estimate_single_change_boundary(x: np.ndarray, min_segment: int = MIN_SEGMENT) -> np.ndarray:
    """
    Offline least-squares estimator of one mean-change boundary.
    For each row, chooses k minimizing SSE of x[:k] and x[k:].
    Returns k, where k is the number of observations in the first segment.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    n, t = x.shape
    candidates = np.arange(min_segment, t - min_segment + 1)
    csum = np.cumsum(x, axis=1)
    csum2 = np.cumsum(x * x, axis=1)
    total_sum = csum[:, -1]
    total_sum2 = csum2[:, -1]
    all_sse = []
    for k in candidates:
        sum1 = csum[:, k - 1]
        ss1 = csum2[:, k - 1]
        n1 = k
        sse1 = ss1 - (sum1 * sum1) / n1
        sum2 = total_sum - sum1
        ss2 = total_sum2 - ss1
        n2 = t - k
        sse2 = ss2 - (sum2 * sum2) / n2
        all_sse.append(sse1 + sse2)
    sse = np.column_stack(all_sse)
    best = np.argmin(sse, axis=1)
    return candidates[best]


# -----------------------------------------------------------------------------
# Data-generating mechanisms
# -----------------------------------------------------------------------------
def generate_monitoring(rng: np.random.Generator, scenario: str, n_sim: int = N_SIM) -> np.ndarray:
    cfg = SCENARIOS[scenario]
    kind = cfg["kind"]
    z = rng.normal(0.0, 1.0, size=(n_sim, N_MON))
    if kind == "stable":
        return z
    if kind == "mean":
        z[:, PRE_CHANGE:] += cfg["shift"]
        return z
    if kind == "drift":
        drift = np.linspace(0.0, cfg["max_shift"], N_MON - PRE_CHANGE)
        z[:, PRE_CHANGE:] += drift
        return z
    if kind == "variance":
        z[:, PRE_CHANGE:] *= cfg["sd_mult"]
        return z
    raise ValueError(kind)


def standardize_with_estimated_phase1(
    rng: np.random.Generator,
    scenario: str,
    n_phase1: int,
    n_sim: int = N_SIM,
) -> np.ndarray:
    """Generate Phase I N(0,1), estimate mean/sample SD, then standardize monitoring values."""
    phase1 = rng.normal(size=(n_sim, n_phase1))
    mu = phase1.mean(axis=1, keepdims=True)
    sd = phase1.std(axis=1, ddof=1, keepdims=True)
    mon = generate_monitoring(rng, scenario, n_sim)
    return (mon - mu) / sd


def generate_stress(rng: np.random.Generator, stress: str, n_sim: int = N_SIM) -> np.ndarray:
    if stress == "Normal IID (true parameters)":
        return rng.normal(size=(n_sim, N_MON))
    if stress == "Student t(5), unit variance":
        df = 5
        return rng.standard_t(df, size=(n_sim, N_MON)) * math.sqrt((df - 2) / df)
    if stress == "AR(1), rho=0.3, unit variance":
        rho = 0.3
        x = np.empty((n_sim, N_MON), dtype=float)
        x[:, 0] = rng.normal(size=n_sim)
        eps_sd = math.sqrt(1 - rho * rho)
        for t in range(1, N_MON):
            x[:, t] = rho * x[:, t - 1] + rng.normal(0, eps_sd, size=n_sim)
        return x
    if stress == "Normal IID, estimated Phase I n=18":
        return standardize_with_estimated_phase1(rng, "Stable", 18, n_sim)
    raise ValueError(stress)


# -----------------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------------
def run_unit_tests() -> None:
    # Rule 1
    z = np.array([0, 0, 3.1], dtype=float)
    assert _endpoint_signal_matrix(z)[1][0, 2]
    # Rule 2
    z = np.repeat(0.5, 9)
    assert _endpoint_signal_matrix(z)[2][0, 8]
    # Rule 3
    z = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])
    assert _endpoint_signal_matrix(z)[3][0, 5]
    # Rule 4
    z = np.array([0.5 if i % 2 == 0 else -0.5 for i in range(14)])
    assert _endpoint_signal_matrix(z)[4][0, 13]
    # Rule 5
    z = np.array([2.1, 2.2, 0.0])
    assert _endpoint_signal_matrix(z)[5][0, 2]
    # Rule 6
    z = np.array([1.1, 1.2, 1.3, 1.4, 0.0])
    assert _endpoint_signal_matrix(z)[6][0, 4]
    # Rule 7
    z = np.repeat(0.5, 15)
    assert _endpoint_signal_matrix(z)[7][0, 14]
    # Rule 8
    z = np.array([1.2, -1.2] * 4)
    assert _endpoint_signal_matrix(z)[8][0, 7]
    # Change-point sanity check: deterministic step
    x = np.r_[np.zeros(10), np.ones(20) * 2.0]
    assert estimate_single_change_boundary(x)[0] == 10


# -----------------------------------------------------------------------------
# Simulation analyses
# -----------------------------------------------------------------------------
def primary_oracle_results() -> pd.DataFrame:
    rows: List[dict] = []
    for s_idx, scenario in enumerate(SCENARIOS):
        rng = np.random.default_rng(SEED + 1000 * s_idx + 17)
        z = generate_monitoring(rng, scenario)
        for method, rules in RULE_SETS.items():
            sig = rule_set_signals(z, rules)
            if scenario == "Stable":
                detected = np.any(sig, axis=1)
                k = int(detected.sum())
                lo, hi = wilson_ci(k, N_SIM)
                rows.append({
                    "Scenario": scenario,
                    "Method": method,
                    "N": N_SIM,
                    "Prechange_false_signal_pct": np.nan,
                    "Eligible_no_prechange_N": np.nan,
                    "Detection_probability_pct": 100 * k / N_SIM,
                    "Detection_CI_low_pct": 100 * lo,
                    "Detection_CI_high_pct": 100 * hi,
                    "Median_detection_delay": np.nan,
                    "Mean_detection_delay": np.nan,
                })
            else:
                pre = np.any(sig[:, :PRE_CHANGE], axis=1)
                eligible = ~pre
                first_after = first_true_index(sig, start=PRE_CHANGE)
                detected_after = (first_after >= PRE_CHANGE) & eligible
                n_elig = int(eligible.sum())
                k = int(detected_after.sum())
                lo, hi = wilson_ci(k, n_elig)
                delays = first_after[detected_after] - PRE_CHANGE
                rows.append({
                    "Scenario": scenario,
                    "Method": method,
                    "N": N_SIM,
                    "Prechange_false_signal_pct": 100 * pre.mean(),
                    "Eligible_no_prechange_N": n_elig,
                    "Detection_probability_pct": 100 * k / n_elig,
                    "Detection_CI_low_pct": 100 * lo,
                    "Detection_CI_high_pct": 100 * hi,
                    "Median_detection_delay": float(np.median(delays)) if len(delays) else np.nan,
                    "Mean_detection_delay": float(np.mean(delays)) if len(delays) else np.nan,
                })
    return pd.DataFrame(rows)


def phase1_sensitivity_results() -> pd.DataFrame:
    rows = []
    phase1_sizes = [18, 30, 50, 100]
    for i, n1 in enumerate(phase1_sizes):
        rng = np.random.default_rng(SEED + 20_000 + i * 101)
        z = standardize_with_estimated_phase1(rng, "Stable", n1)
        for method, rules in RULE_SETS.items():
            sig = rule_set_signals(z, rules)
            alarm = np.any(sig, axis=1)
            k = int(alarm.sum())
            lo, hi = wilson_ci(k, N_SIM)
            rows.append({
                "Phase_I_reference": f"Estimated n={n1}",
                "Phase_I_n": n1,
                "Method": method,
                "N": N_SIM,
                "False_alarm_probability_pct": 100 * k / N_SIM,
                "CI_low_pct": 100 * lo,
                "CI_high_pct": 100 * hi,
            })
    # Known parameter benchmark with independent seed
    rng = np.random.default_rng(SEED + 17)
    z = generate_monitoring(rng, "Stable")
    for method, rules in RULE_SETS.items():
        alarm = np.any(rule_set_signals(z, rules), axis=1)
        k = int(alarm.sum())
        lo, hi = wilson_ci(k, N_SIM)
        rows.append({
            "Phase_I_reference": "True mean and SD",
            "Phase_I_n": np.nan,
            "Method": method,
            "N": N_SIM,
            "False_alarm_probability_pct": 100 * k / N_SIM,
            "CI_low_pct": 100 * lo,
            "CI_high_pct": 100 * hi,
        })
    return pd.DataFrame(rows)


def phase1_n18_detection_results() -> pd.DataFrame:
    rows = []
    for s_idx, scenario in enumerate(SCENARIOS):
        if scenario == "Stable":
            continue
        rng = np.random.default_rng(SEED + 40_000 + 1000 * s_idx)
        z = standardize_with_estimated_phase1(rng, scenario, 18)
        for method, rules in RULE_SETS.items():
            sig = rule_set_signals(z, rules)
            pre = np.any(sig[:, :PRE_CHANGE], axis=1)
            eligible = ~pre
            first_after = first_true_index(sig, start=PRE_CHANGE)
            detected = (first_after >= PRE_CHANGE) & eligible
            n_elig = int(eligible.sum())
            k = int(detected.sum())
            lo, hi = wilson_ci(k, n_elig)
            delays = first_after[detected] - PRE_CHANGE
            rows.append({
                "Scenario": scenario,
                "Method": method,
                "N": N_SIM,
                "Prechange_false_signal_pct": 100 * pre.mean(),
                "Eligible_no_prechange_N": n_elig,
                "Detection_probability_pct": 100 * k / n_elig,
                "Detection_CI_low_pct": 100 * lo,
                "Detection_CI_high_pct": 100 * hi,
                "Median_detection_delay": float(np.median(delays)) if len(delays) else np.nan,
                "Mean_detection_delay": float(np.mean(delays)) if len(delays) else np.nan,
            })
    return pd.DataFrame(rows)


def stress_test_results() -> pd.DataFrame:
    stresses = [
        "Normal IID (true parameters)",
        "Student t(5), unit variance",
        "AR(1), rho=0.3, unit variance",
        "Normal IID, estimated Phase I n=18",
    ]
    rows = []
    for i, stress in enumerate(stresses):
        rng = np.random.default_rng(SEED + 60_000 + i * 997)
        z = generate_stress(rng, stress)
        for method, rules in RULE_SETS.items():
            alarm = np.any(rule_set_signals(z, rules), axis=1)
            k = int(alarm.sum())
            lo, hi = wilson_ci(k, N_SIM)
            rows.append({
                "Stress_condition": stress,
                "Method": method,
                "N": N_SIM,
                "False_alarm_probability_pct": 100 * k / N_SIM,
                "CI_low_pct": 100 * lo,
                "CI_high_pct": 100 * hi,
            })
    return pd.DataFrame(rows)



def horizon_sensitivity_results() -> pd.DataFrame:
    """Stable-process probability of at least one signal across monitoring horizons."""
    rows = []
    for i, horizon in enumerate([15, 30, 60]):
        rng = np.random.default_rng(SEED + 70_000 + i * 503)
        z = rng.normal(size=(N_SIM, horizon))
        for method, rules in RULE_SETS.items():
            alarm = np.any(rule_set_signals(z, rules), axis=1)
            k = int(alarm.sum())
            lo, hi = wilson_ci(k, N_SIM)
            rows.append({
                "Monitoring_horizon_batches": horizon,
                "Method": method,
                "N": N_SIM,
                "False_alarm_probability_pct": 100 * k / N_SIM,
                "CI_low_pct": 100 * lo,
                "CI_high_pct": 100 * hi,
            })
    return pd.DataFrame(rows)

def change_point_results() -> pd.DataFrame:
    rows = []
    for s_idx, scenario in enumerate(["Abrupt +1 SD", "Abrupt +2 SD", "Abrupt +3 SD", "Gradual drift to +3 SD"]):
        rng = np.random.default_rng(SEED + 80_000 + s_idx * 733)
        z = generate_monitoring(rng, scenario)
        k_hat = estimate_single_change_boundary(z)
        if scenario.startswith("Abrupt"):
            err = np.abs(k_hat - TRUE_CP_BOUNDARY)
            rows.append({
                "Scenario": scenario,
                "N": N_SIM,
                "True_boundary": TRUE_CP_BOUNDARY,
                "Mean_estimated_boundary": float(np.mean(k_hat)),
                "Median_estimated_boundary": float(np.median(k_hat)),
                "MAE_batches": float(np.mean(err)),
                "Median_abs_error_batches": float(np.median(err)),
                "Exact_pct": 100 * np.mean(err == 0),
                "Within_1_batch_pct": 100 * np.mean(err <= 1),
                "Within_2_batches_pct": 100 * np.mean(err <= 2),
            })
        else:
            rows.append({
                "Scenario": scenario,
                "N": N_SIM,
                "True_boundary": np.nan,
                "Mean_estimated_boundary": float(np.mean(k_hat)),
                "Median_estimated_boundary": float(np.median(k_hat)),
                "MAE_batches": np.nan,
                "Median_abs_error_batches": np.nan,
                "Exact_pct": np.nan,
                "Within_1_batch_pct": np.nan,
                "Within_2_batches_pct": np.nan,
            })
    return pd.DataFrame(rows)


def scenario_design_table() -> pd.DataFrame:
    return pd.DataFrame([
        ["Stable", "IID N(0,1) throughout", "No change", "False-signal probability"],
        ["Abrupt +1 SD", "Mean shifts from 0 to +1 SD after observation 10", "Boundary 10/11", "Small sustained shift"],
        ["Abrupt +2 SD", "Mean shifts from 0 to +2 SD after observation 10", "Boundary 10/11", "Moderate sustained shift"],
        ["Abrupt +3 SD", "Mean shifts from 0 to +3 SD after observation 10", "Boundary 10/11", "Large sustained shift"],
        ["Gradual drift to +3 SD", "Linear mean drift from 0 to +3 SD over observations 11-30", "No single true boundary", "Trend sensitivity"],
        ["SD x2", "SD doubles from 1 to 2 after observation 10; mean remains 0", "Boundary 10/11", "Variance instability"],
    ], columns=["Scenario", "Data-generating mechanism", "Change timing", "Primary purpose"])


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def make_figures(primary: pd.DataFrame, phase1: pd.DataFrame, cp: pd.DataFrame) -> None:
    # Fig 1 - stable false alarms (oracle)
    stable = primary[primary["Scenario"] == "Stable"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(stable["Method"], stable["Detection_probability_pct"])
    ax.set_ylabel("Probability of >=1 false signal over 30 batches (%)")
    ax.set_xlabel("Prospective rule set")
    ax.set_ylim(0, max(stable["Detection_probability_pct"]) * 1.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig1_Stable_False_Alarm_Tradeoff.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "Fig1_Stable_False_Alarm_Tradeoff.eps", format="eps", bbox_inches="tight")
    plt.close(fig)

    # Fig 2 - detection probabilities by changed scenario
    changed = primary[primary["Scenario"] != "Stable"].copy()
    piv = changed.pivot(index="Scenario", columns="Method", values="Detection_probability_pct")
    ordered_scenarios = ["Abrupt +1 SD", "Abrupt +2 SD", "Abrupt +3 SD", "Gradual drift to +3 SD", "SD x2"]
    piv = piv.reindex(ordered_scenarios)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    markers = ["o", "s", "^"]
    for marker, method in zip(markers, RULE_SETS.keys()):
        ax.plot(range(len(piv)), piv[method], marker=marker, linewidth=1.8, label=method)
    ax.set_xticks(range(len(piv)))
    ax.set_xticklabels(piv.index, rotation=20, ha="right")
    ax.set_ylabel("Conditional detection probability (%)")
    ax.set_xlabel("Simulated process change")
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig2_Detection_Probability.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "Fig2_Detection_Probability.eps", format="eps", bbox_inches="tight")
    plt.close(fig)

    # Fig 3 - baseline-size sensitivity
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    p = phase1.dropna(subset=["Phase_I_n"]).copy()
    markers = ["o", "s", "^"]
    for marker, method in zip(markers, RULE_SETS.keys()):
        q = p[p["Method"] == method].sort_values("Phase_I_n")
        ax.plot(q["Phase_I_n"], q["False_alarm_probability_pct"], marker=marker, linewidth=1.8, label=method)
    ax.set_xlabel("Phase I reference sample size")
    ax.set_ylabel("Probability of >=1 false signal over 30 batches (%)")
    ax.set_xticks([18, 30, 50, 100])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig3_PhaseI_Baseline_Sensitivity.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "Fig3_PhaseI_Baseline_Sensitivity.eps", format="eps", bbox_inches="tight")
    plt.close(fig)

    # Fig 4 - change-point localization
    abrupt = cp[cp["Scenario"].str.startswith("Abrupt")].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(abrupt))
    width = 0.25
    ax.bar(x - width, abrupt["Exact_pct"], width, label="Exact")
    ax.bar(x, abrupt["Within_1_batch_pct"], width, label="Within +/-1 batch")
    ax.bar(x + width, abrupt["Within_2_batches_pct"], width, label="Within +/-2 batches")
    ax.set_xticks(x)
    ax.set_xticklabels(abrupt["Scenario"])
    ax.set_ylabel("Localization accuracy (%)")
    ax.set_xlabel("Abrupt mean shift")
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig4_ChangePoint_Localization.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "Fig4_ChangePoint_Localization.eps", format="eps", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    run_unit_tests()
    design = scenario_design_table()
    primary = primary_oracle_results()
    phase1 = phase1_sensitivity_results()
    n18det = phase1_n18_detection_results()
    stress = stress_test_results()
    horizon = horizon_sensitivity_results()
    cp = change_point_results()

    design.to_csv(RESULTS / "Table1_Simulation_Design.csv", index=False)
    primary.to_csv(RESULTS / "Table2_Primary_Oracle_Performance.csv", index=False)
    phase1.to_csv(RESULTS / "Table3_PhaseI_Baseline_Sensitivity.csv", index=False)
    n18det.to_csv(RESULTS / "Supplement_PhaseI_n18_Detection.csv", index=False)
    stress.to_csv(RESULTS / "Table4_Robustness_Stress_Tests.csv", index=False)
    horizon.to_csv(RESULTS / "Supplement_Monitoring_Horizon_Sensitivity.csv", index=False)
    cp.to_csv(RESULTS / "Table5_ChangePoint_Localization.csv", index=False)

    make_figures(primary, phase1, cp)

    # Combined machine-readable output
    with pd.ExcelWriter(RESULTS / "CPV_JPI_Simulation_Results.xlsx", engine="openpyxl") as xw:
        design.to_excel(xw, sheet_name="Design", index=False)
        primary.to_excel(xw, sheet_name="Primary_Oracle", index=False)
        phase1.to_excel(xw, sheet_name="PhaseI_Sensitivity", index=False)
        n18det.to_excel(xw, sheet_name="PhaseI18_Detection", index=False)
        stress.to_excel(xw, sheet_name="Stress_Tests", index=False)
        horizon.to_excel(xw, sheet_name="Horizon_Sensitivity", index=False)
        cp.to_excel(xw, sheet_name="ChangePoint", index=False)

    print("Unit tests: PASSED")
    print("\nStable oracle false-alarm probabilities:")
    print(primary[primary.Scenario == "Stable"][["Method", "Detection_probability_pct", "Detection_CI_low_pct", "Detection_CI_high_pct"]].to_string(index=False))
    print("\nAbrupt +1 SD conditional detection:")
    print(primary[primary.Scenario == "Abrupt +1 SD"][["Method", "Prechange_false_signal_pct", "Detection_probability_pct", "Median_detection_delay"]].to_string(index=False))
    print("\nPhase I n=18 stable false-alarm probabilities:")
    print(phase1[phase1.Phase_I_n == 18][["Method", "False_alarm_probability_pct"]].to_string(index=False))
    print("\nChange-point localization:")
    print(cp.to_string(index=False))


if __name__ == "__main__":
    main()
