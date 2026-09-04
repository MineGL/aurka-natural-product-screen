#!/usr/bin/env python
"""Recalibrate the GNINA CNN cutoff under the v4 protocol, with uncertainty.

Derived from `define_cnn_cutoff.py` (DeepAURKA, v3). Three changes, all of them
responses to specific defects in the original:

1. **No default paths.** The v3 script defaulted to `docking_coconut_v3/...`, so
   an unqualified invocation would silently recalibrate against v3 results. Both
   inputs are now required arguments.
2. **Bootstrap confidence intervals** on AUC *and* on the Youden threshold. At
   n = 40 + 40 the Youden point is a noisy statistic, and it is the number the
   production screen cuts on, so reporting it without an interval overstates its
   precision.
3. **Labels joined from the calibration input**, since the v4 pipeline's ranked
   CSV carries no `role` column.

Resampling is stratified: actives and inactives are resampled separately to
their original sizes, which preserves the 40/40 design and keeps every replicate
estimable. Replicates where either class becomes degenerate are discarded and
counted.

The verdict ladder is unchanged from v3 and is pre-committed: AUC <= 0.60 means
the score must not be used to rank at all; < 0.70 means coarse filter only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Higher is better for the CNN scores; Vina affinity is negated so one
# orientation covers all three.
SCORE_COLUMNS = {
    "top_CNNscore": ("CNNscore (pose quality)", 1.0),
    "top_CNNaffinity": ("CNNaffinity (predicted pK)", 1.0),
    "top_vina_affinity": ("Vina affinity (kcal/mol, negated)", -1.0),
}

AUC_USELESS = 0.60
AUC_WEAK = 0.70
V3_REFERENCE = {"score": "top_CNNaffinity", "auc": 0.8606, "threshold": 7.2983,
                "sensitivity": 0.875, "specificity": 0.825}


def log(msg: str) -> None:
    print(msg, flush=True)


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC via the Mann-Whitney U identity, ties counted as half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def youden_threshold(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float, float]:
    candidates = np.unique(np.concatenate([pos, neg]))
    best = (float("-inf"), np.nan, np.nan, np.nan)
    for t in candidates:
        sens = float((pos >= t).mean())
        spec = float((neg < t).mean())
        j = sens + spec - 1.0
        if j > best[0]:
            best = (j, float(t), sens, spec)
    return best[1], best[2], best[3], best[0]


def bootstrap_ci(pos, neg, n_boot, seed):
    rng = np.random.default_rng(seed)
    aucs, thrs = [], []
    for _ in range(n_boot):
        p = pos[rng.integers(0, len(pos), len(pos))]
        n = neg[rng.integers(0, len(neg), len(neg))]
        if len(np.unique(np.concatenate([p, n]))) < 2:
            continue
        aucs.append(roc_auc(p, n))
        thrs.append(youden_threshold(p, n)[0])
    aucs = np.asarray(aucs, dtype=float)
    thrs = np.asarray(thrs, dtype=float)
    return {
        "n_effective": int(len(aucs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "threshold_median": float(np.median(thrs)),
        "threshold_ci95": [float(np.percentile(thrs, 2.5)), float(np.percentile(thrs, 97.5))],
        "threshold_iqr": [float(np.percentile(thrs, 25)), float(np.percentile(thrs, 75))],
        "p_auc_above_0.70": float((aucs > AUC_WEAK).mean()),
    }


def evaluate(df, label_col_active, label_col_inactive, n_boot, seed, tag):
    actives = df[df["role"].eq(label_col_active)]
    inactives = df[df["role"].eq(label_col_inactive)]
    if len(actives) < 5 or len(inactives) < 5:
        raise SystemExit(f"[{tag}] too few calibration compounds: "
                         f"{len(actives)} actives / {len(inactives)} inactives")
    metrics = {}
    for col, (label, sign) in SCORE_COLUMNS.items():
        if col not in df.columns:
            continue
        pos = (sign * actives[col].astype(float)).dropna().to_numpy()
        neg = (sign * inactives[col].astype(float)).dropna().to_numpy()
        auc = roc_auc(pos, neg)
        thr, sens, spec, j = youden_threshold(pos, neg)
        boot = bootstrap_ci(pos, neg, n_boot, seed)
        metrics[col] = {
            "label": label,
            "orientation": "higher_is_better" if sign > 0 else "lower_is_better",
            "n_actives": int(len(pos)), "n_inactives": int(len(neg)),
            "auc": round(auc, 4),
            "auc_ci95": [round(v, 4) for v in boot["auc_ci95"]],
            "youden_threshold_oriented": round(thr, 4),
            "threshold_in_original_units": round(sign * thr, 4),
            "threshold_ci95_oriented": [round(v, 4) for v in boot["threshold_ci95"]],
            "threshold_iqr_oriented": [round(v, 4) for v in boot["threshold_iqr"]],
            "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "youden_j": round(j, 4),
            "active_median_oriented": round(float(np.median(pos)), 4),
            "inactive_median_oriented": round(float(np.median(neg)), 4),
            "bootstrap": {k: boot[k] for k in ("n_effective", "auc_mean", "p_auc_above_0.70")},
        }
        log(f"  [{tag}] {label:34s} AUC {auc:.4f} [{boot['auc_ci95'][0]:.3f}, {boot['auc_ci95'][1]:.3f}]"
            f"  thr {sign*thr:8.4f} [{boot['threshold_ci95'][0]:.3f}, {boot['threshold_ci95'][1]:.3f}]"
            f"  sens {sens:.3f} spec {spec:.3f}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranked-csv", type=Path, required=True,
                    help="production gnina_cuda_results_ranked.csv from the control run")
    ap.add_argument("--labels-csv", type=Path, required=True,
                    help="calibration input CSV carrying coconut_id, role, reference_pIC50")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=67)
    args = ap.parse_args()

    ranked = pd.read_csv(args.ranked_csv, low_memory=False)
    labels = pd.read_csv(args.labels_csv)[["coconut_id", "role", "reference_pIC50"]]
    df = ranked.merge(labels, on="coconut_id", how="left", validate="one_to_one")
    if df["role"].isna().any():
        raise SystemExit(f"{int(df['role'].isna().sum())} docked rows have no label")
    ok = df["status"].astype(str).eq("ok")
    log(f"docked rows {len(df)}; status ok {int(ok.sum())}; "
        f"roles {df['role'].value_counts().to_dict()}")
    df = df[ok].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log("\nall docked controls:")
    metrics_all = evaluate(df, "train_active", "train_inactive", args.n_boot, args.seed, "all")

    ready = df[df["pose1_posebusters_ready"].astype(str).str.lower().isin(("true", "1"))]
    log(f"\nPoseBusters-ready subset ({len(ready)} of {len(df)}):")
    metrics_ready = evaluate(ready, "train_active", "train_inactive", args.n_boot, args.seed, "pb-ready")

    best_col = max(metrics_all, key=lambda c: metrics_all[c]["auc"])
    best = metrics_all[best_col]
    if best["auc"] <= AUC_USELESS:
        discrimination = "none"
        verdict = (f"Best AUC is {best['auc']:.4f} ({best['label']}), at or below {AUC_USELESS}. "
                   "Docking does not separate known AURKA actives from inactives under the v4 "
                   "protocol. Do NOT use the CNN score to rank or to claim potency; treat docking "
                   "only as a pose-plausibility and binding-mode check.")
    elif best["auc"] < AUC_WEAK:
        discrimination = "weak"
        verdict = (f"Best AUC is {best['auc']:.4f} ({best['label']}), weak. Usable as a coarse "
                   "filter but must be reported with this AUC; it does not support potency claims.")
    else:
        discrimination = "usable"
        verdict = (f"Best AUC is {best['auc']:.4f} ({best['label']}). The cutoff separates measured "
                   "actives from inactives and is a defensible screening filter.")
    log("\n" + verdict)

    v3 = V3_REFERENCE
    same = metrics_all.get(v3["score"], {})
    drift = {
        "v3_score": v3["score"], "v3_auc": v3["auc"], "v3_threshold": v3["threshold"],
        "v4_auc": same.get("auc"), "v4_threshold": same.get("threshold_in_original_units"),
        "auc_delta": None if same.get("auc") is None else round(same["auc"] - v3["auc"], 4),
        "threshold_delta": None if same.get("threshold_in_original_units") is None
        else round(same["threshold_in_original_units"] - v3["threshold"], 4),
        "v3_threshold_inside_v4_ci": None if not same else bool(
            same["threshold_ci95_oriented"][0] <= v3["threshold"] <= same["threshold_ci95_oriented"][1]),
        "note": ("The v3 calibration used the older GNINA build and v3 preparation; a threshold "
                 "shift is expected. What matters is whether the v3 value lies inside the v4 "
                 "interval - if it does, the legacy cutoff was not wrong, merely unqualified."),
    }

    summary = {
        "stage": "cnn_cutoff_recalibration_v4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"ranked_csv": str(args.ranked_csv), "labels_csv": str(args.labels_csv)},
        "calibration": {
            "n_actives": best["n_actives"], "n_inactives": best["n_inactives"],
            "active_definition": "measured AURKA pIC50 >= 8 (AD reference pool)",
            "inactive_definition": "measured AURKA pIC50 <= 5 (AD reference pool)",
            "protocol": "v4 frozen: box -2.709/-31.973/4.694, 30 A cube, exhaustiveness 32, "
                        "9 modes, CNN rescore, seed 67, v3p prep at pH 7.4",
            "bootstrap_replicates": args.n_boot,
            "bootstrap_scheme": "stratified: actives and inactives resampled separately",
        },
        "score_metrics_all_controls": metrics_all,
        "score_metrics_posebusters_ready": metrics_ready,
        "selected_score": best_col,
        "selected_threshold": best["threshold_in_original_units"],
        "selected_threshold_ci95": best["threshold_ci95_oriented"],
        "discrimination": discrimination,
        "verdict": verdict,
        "comparison_with_v3": drift,
    }
    out = args.out_dir / "cnn_cutoff_v4_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    log("\n" + "=" * 78)
    log(f"selected score : {best['label']}")
    log(f"AUC            : {best['auc']:.4f}  95% CI [{best['auc_ci95'][0]:.4f}, {best['auc_ci95'][1]:.4f}]")
    log(f"threshold      : {best['threshold_in_original_units']:.4f}  "
        f"95% CI [{best['threshold_ci95_oriented'][0]:.4f}, {best['threshold_ci95_oriented'][1]:.4f}]")
    log(f"sens / spec    : {best['sensitivity']:.3f} / {best['specificity']:.3f}")
    log(f"discrimination : {discrimination}")
    log(f"v3 -> v4       : AUC {v3['auc']} -> {drift['v4_auc']} ({drift['auc_delta']:+.4f}); "
        f"threshold {v3['threshold']} -> {drift['v4_threshold']} ({drift['threshold_delta']:+.4f}); "
        f"v3 value inside v4 CI: {drift['v3_threshold_inside_v4_ci']}")
    log("=" * 78)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
