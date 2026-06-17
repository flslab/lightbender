#!/usr/bin/env python3
"""
================================
Design : Shape (3 levels) x Error (5 levels), fully within-subjects
N      : 11 participants
DV     : Quality rating (0 - 10)

Pipeline
--------
1. Load & preprocess  - average the 3 repetitions per (participant, shape, error)
2. Simple effects     - Friedman's test per shape (varying error) and per error (varying shape)
3. Post-hoc           - Pairwise Wilcoxon signed-rank + Bonferroni correction

Usage
-----
  python analyze_significance.py                      # uses default path
  python analyze_significance.py --file your_data.xlsx
"""

import argparse
import textwrap
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG PIPELIE
# ──────────────────────────────────────────────────────────────────────────────

SKIP_ANOVA = True

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

DIVIDER = "─" * 72

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_excel(path, sheet_name="All Shapes and Conditions")
    error_cols = ["0 mm", "3 mm", "10.1 mm", "30 mm", "100 mm"]

    # Exclude specific shapes from analysis
    df = df[~df['Shape'].isin(['Letter X', 'Blue Emoji'])]

    # Average the repetitions, one score per (participant, shape, error)
    avg = (
        df.groupby(["Participant", "Shape"])[error_cols]
        .mean()
        .reset_index()
    )

    shapes       = sorted(avg["Shape"].unique())
    participants = sorted(avg["Participant"].unique())
    n, a, b      = len(participants), len(shapes), len(error_cols)

    # Build 3-D array X[subject, shape, error]
    X = np.full((n, a, b), np.nan)
    for i, pid in enumerate(participants):
        for j, shape in enumerate(shapes):
            row = avg[(avg["Participant"] == pid) & (avg["Shape"] == shape)]
            if not row.empty:
                X[i, j, :] = row[error_cols].values[0]

    if np.isnan(X).any():
        raise ValueError("Missing cells detected - check that every participant "
                         "completed every shape condition.")

    print(X)

    return X, shapes, error_cols, participants


# ──────────────────────────────────────────────────────────────────────────────
# 2. SIMPLE EFFECTS  (Friedman's test – non-parametric, robust for ordinal data)
# ──────────────────────────────────────────────────────────────────────────────

def simple_effects(X, shapes, error_cols):
    """
    For each shape  : test whether error condition matters  (Friedman over errors)
    For each error  : test whether shape matters             (Friedman over shapes)
    """
    n, a, b = X.shape
    se_error_in_shape = []                # effect of error, per shape
    se_shape_in_error = []                # effect of shape, per error

    for j, shape in enumerate(shapes):
        data = X[:, j, :]                 # (n, b)
        stat, p = stats.friedmanchisquare(*data.T)
        se_error_in_shape.append({
            "Shape": shape, "χ²": stat, "df": b - 1, "p": p, "sig": sig_stars(p)
        })

    for k, err in enumerate(error_cols):
        data = X[:, :, k]                 # (n, a)
        stat, p = stats.friedmanchisquare(*data.T)
        se_shape_in_error.append({
            "Error": err, "χ²": stat, "df": a - 1, "p": p, "sig": sig_stars(p)
        })

    return (pd.DataFrame(se_error_in_shape),
            pd.DataFrame(se_shape_in_error))


# ──────────────────────────────────────────────────────────────────────────────
# 3. POST-HOC  (Pairwise Wilcoxon signed-rank + Bonferroni)
# ──────────────────────────────────────────────────────────────────────────────

def pairwise_wilcoxon(data, labels, alpha=0.05):
    """
    All pairwise Wilcoxon signed-rank tests with Bonferroni correction.

    Parameters
    ----------
    data   : (n, k) array  - columns are levels to compare
    labels : list of k strings

    Returns
    -------
    DataFrame sorted by Bonferroni-adjusted p-value
    """
    pairs = list(combinations(range(len(labels)), 2))
    n_comp = len(pairs)
    rows = []
    for i, j in pairs:
        stat, p = stats.wilcoxon(data[:, i], data[:, j],
                                 alternative="two-sided")
        p_bonf = min(p * n_comp, 1.0)
        rows.append({
            "Comparison":       f"{labels[i]}  vs  {labels[j]}",
            "W":                stat,
            "p (raw)":          p,
            "p (Bonferroni)":   p_bonf,
            "Sig":              sig_stars(p_bonf),
        })
    return (pd.DataFrame(rows)
              .sort_values("p (Bonferroni)")
              .reset_index(drop=True))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(path):
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)

    # ── Load ─────────────────────────────────────────────────────────────────
    X, shapes, error_cols, participants = load_data(path)
    n, a, b = X.shape
    print(f"\nLoaded: {n} participants, {a} shapes, {b} error levels")
    print(f"Shapes : {shapes}")
    print(f"Errors : {error_cols}")

    # ── Simple effects ───────────────────────────────────────────────────────
    section("SIMPLE EFFECTS  (Friedman's test)")
    se_err, se_shp = simple_effects(X, shapes, error_cols)

    print("\n  Effect of Error within each Shape:")
    print(se_err.to_string(index=False))

    print("\n  Effect of Shape within each Error level:")
    print(se_shp.to_string(index=False))

    # ── Post-hoc ─────────────────────────────────────────────────────────────
    section("POST-HOC  (Pairwise Wilcoxon + Bonferroni)")

    print("\n  ── Error comparisons within each Shape ──────────────────────────")
    for j, shape in enumerate(shapes):
        # Only run post-hoc if simple effect was significant
        p_simple = float(se_err[se_err["Shape"] == shape]["p"].values[0])
        print(f"\n  Shape: {shape}  (simple-effect p = {p_simple:.4f})")
        if p_simple >= 0.05:
            print("    Skipping post-hoc (simple effect not significant)")
            continue
        ph = pairwise_wilcoxon(X[:, j, :], error_cols)
        print(ph.to_string(index=False))

    print("\n  ── Shape comparisons within each Error level ────────────────────")
    for k, err in enumerate(error_cols):
        p_simple = float(se_shp[se_shp["Error"] == err]["p"].values[0])
        print(f"\n  Error: {err}  (simple-effect p = {p_simple:.4f})")
        if p_simple >= 0.05:
            print("    Skipping post-hoc (simple effect not significant)")
            continue
        ph = pairwise_wilcoxon(X[:, :, k], shapes)
        print(ph.to_string(index=False))

    print(f"\n{'─'*72}")
    print("  Stars: *** p<.001  ** p<.01  * p<.05  ns = not significant")
    print(f"{'─'*72}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Friedman's test - Post-hoc Wilcoxon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              python analyze_significance.py
              python analyze_significance.py --file path/to/study2_results.xlsx
        """)
    )
    parser.add_argument(
        "--file", "-f",
        default="study2_results_tabulated.xlsx",
        help="Path to the Excel file (default: study2_results_tabulated-1.xlsx)"
    )
    args = parser.parse_args()
    main(args.file)