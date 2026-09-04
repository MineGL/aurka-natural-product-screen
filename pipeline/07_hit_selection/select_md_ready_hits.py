#!/usr/bin/env python
"""Select the MD-ready AURKA hit list from the v5 candidate docking round.

Produces a ranked compound list and stops there: no MD systems are built and no
trajectory is launched. Every surviving row carries the pose-1 SDF path, which is
the only input `20_build_system.sh <name> <ligand.sdf>` needs later.

Selection rule, fixed before any candidate score was read
---------------------------------------------------------
1. pose-1 PoseBusters-ready. RMSD alone admits physically invalid poses
   (Buttenschoen 2024), so physical validity gates first.
2. `top_CNNaffinity` >= the v4-recalibrated Youden threshold, read from
   `cnn_cutoff_v4_summary.json` rather than hardcoded.
3. A PLIP hydrogen bond to Glu211 or Ala213 - the hinge contacts the cognate
   ligand SKE itself makes in 5DPV (N1-Glu211 O 3.07 A, N3-Ala213 O 2.36 A).
4. Ranked by CNNaffinity.

Because the threshold carries a bootstrap 95% CI of roughly +/-0.18 log units, the
count is also reported at both CI bounds. The point estimate is not treated as
exact.

Set A compounds are labelled `positive_control`, never `hit`: all four are
synthetic kinase inhibitors leaked into COCONUT (NP-likeness -1.27 to -2.05) and
were selected by a similarity floor that later proved to be the wrong instrument
(ECFP4 activity-relevant Tanimoto centres near 0.15; Jasial 2016). They test the
pipeline; they are not discoveries.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CARRY = [
    "coconut_id", "set", "standard_inchi_key", "consensus_rank",
    "consensus_pred_pIC50", "consensus_pred_IC50_nM",
    "consensus_pred_pIC50_lower", "consensus_pred_pIC50_upper",
    "xgb_ensemble_pred_pIC50", "gat_ensemble_pred_pIC50", "member_spread_pIC50",
    "nearest_train_tanimoto", "novelty_tier", "np_likeness",
    "sa_MW", "sa_WLOGP", "sa_TPSA", "sa_RotB", "sa_esol_logS",
    "sa_n_druglikeness_passed", "has_structural_alert",
]
SCORES = ["top_CNNaffinity", "top_CNNscore", "top_vina_affinity",
          "best_CNNaffinity", "pose_count"]


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranked-csv", type=Path, required=True)
    ap.add_argument("--plip-json", type=Path, required=True,
                    help="interactions.json from the PLIP profiling run")
    ap.add_argument("--input-csv", type=Path, required=True,
                    help="aurka_setAB_1004.csv, for the set label")
    ap.add_argument("--panel-csv", type=Path, required=True,
                    help="coconut_v5_swissadme_panel.csv, for ML and ADME columns")
    ap.add_argument("--cutoff-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cut = json.loads(args.cutoff_json.read_text())
    score_col = cut["selected_score"]
    thr = float(cut["selected_threshold"])
    thr_lo, thr_hi = (float(v) for v in cut["selected_threshold_ci95"])
    log(f"threshold {thr:.4f} on {score_col}  (95% CI {thr_lo:.4f} - {thr_hi:.4f}), "
        f"AUC {cut['score_metrics_all_controls'][score_col]['auc']}")

    ranked = pd.read_csv(args.ranked_csv, low_memory=False)
    sets = pd.read_csv(args.input_csv)[["coconut_id", "set"]]
    df = ranked.merge(sets, on="coconut_id", how="left", validate="one_to_one")
    if df["set"].isna().any():
        raise SystemExit(f"{int(df['set'].isna().sum())} docked rows have no set label")

    panel_cols = [c for c in CARRY if c not in ("coconut_id", "set")]
    panel = pd.read_csv(args.panel_csv, low_memory=False)
    keep = ["coconut_id"] + [c for c in panel_cols if c in panel.columns]
    df = df.merge(panel[keep], on="coconut_id", how="left", validate="one_to_one")

    # PLIP: fail closed rather than silently dropping the hinge criterion.
    plip = json.loads(args.plip_json.read_text())
    if not plip:
        raise SystemExit("PLIP interactions file is empty; hinge criterion cannot be applied")
    prof = pd.DataFrame([
        {"coconut_id": k,
         "plip_hinge_hbond": bool(v.get("hinge_hbond")),
         "plip_hinge_present": bool(v.get("hinge_present")),
         "plip_n_interactions": int(v.get("count", 0))}
        for k, v in plip.items()
    ])
    df = df.merge(prof, on="coconut_id", how="left")
    df["plip_profiled"] = df["plip_hinge_hbond"].notna()
    for col in ("plip_hinge_hbond", "plip_hinge_present"):
        df[col] = df[col].fillna(False).astype(bool)
    df["plip_n_interactions"] = df["plip_n_interactions"].fillna(0).astype(int)

    ok = df["status"].astype(str).eq("ok")
    ready = df["pose1_posebusters_ready"].astype(str).str.lower().isin(("true", "1"))
    score = pd.to_numeric(df[score_col], errors="coerce")
    df["passes_cutoff"] = score >= thr
    df["passes_cutoff_at_ci_lower"] = score >= thr_lo
    df["passes_cutoff_at_ci_upper"] = score >= thr_hi
    df["md_ready"] = ok & ready & df["passes_cutoff"] & df["plip_hinge_hbond"]
    df["role"] = np.where(df["set"].eq("setA"), "positive_control", "hit")

    funnel = {
        "docked": int(len(df)),
        "status_ok": int(ok.sum()),
        "pose1_posebusters_ready": int((ok & ready).sum()),
        "plip_profiled": int(df["plip_profiled"].sum()),
        f"passes_{score_col}_ge_{thr:.4f}": int((ok & ready & df["passes_cutoff"]).sum()),
        "at_ci_lower_bound": int((ok & ready & df["passes_cutoff_at_ci_lower"]).sum()),
        "at_ci_upper_bound": int((ok & ready & df["passes_cutoff_at_ci_upper"]).sum()),
        "plus_hinge_hbond": int(df["md_ready"].sum()),
        "md_ready_hits_setB": int((df["md_ready"] & df["set"].eq("setB")).sum()),
        "md_ready_positive_controls_setA": int((df["md_ready"] & df["set"].eq("setA")).sum()),
    }

    hits = df[df["md_ready"]].copy()
    hits = hits.sort_values(score_col, ascending=False)
    hits["md_rank"] = range(1, len(hits) + 1)
    hits["md_system_name"] = ["HIT_%03d_%s" % (r, str(cid).replace(".", "_"))
                              for r, cid in zip(hits["md_rank"], hits["coconut_id"])]

    out_cols = (["md_rank", "md_system_name", "role"]
                + [c for c in CARRY if c in hits.columns]
                + SCORES
                + ["pose1_posebusters_ready", "plip_hinge_hbond", "plip_hinge_present",
                   "plip_n_interactions", "passes_cutoff", "passes_cutoff_at_ci_lower",
                   "passes_cutoff_at_ci_upper", "protonation_identity_changed",
                   "inchikey_source", "inchikey_docked", "poses", "ligand_sdf"])
    out_cols = [c for c in dict.fromkeys(out_cols) if c in hits.columns]
    hits_path = args.out_dir / "md_ready_hits.csv"
    hits[out_cols].to_csv(hits_path, index=False)
    df.to_csv(args.out_dir / "candidates_with_selection_flags.csv", index=False)

    summary = {
        "stage": "md_ready_hit_selection_v5",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "md_launched": False,
        "note": "Compound list only. No MD system was built and no trajectory was started.",
        "selection_rule": {
            "1_physical_validity": "pose-1 PoseBusters-ready",
            "2_score": f"{score_col} >= {thr} (v4 recalibrated Youden threshold)",
            "3_interaction": "PLIP hydrogen bond to Glu211 or Ala213",
            "ranking": score_col,
            "fixed_before_scores_were_read": True,
        },
        "threshold": {"score": score_col, "value": thr, "ci95": [thr_lo, thr_hi],
                      "auc": cut["score_metrics_all_controls"][score_col]["auc"],
                      "auc_ci95": cut["score_metrics_all_controls"][score_col]["auc_ci95"]},
        "funnel": funnel,
        "set_A_handling": ("labelled positive_control, not hit: all four are synthetic kinase "
                           "inhibitors leaked into COCONUT and were selected by a similarity "
                           "floor that proved to be the wrong instrument"),
        "outputs": {"md_ready_hits": str(hits_path),
                    "all_candidates_with_flags": str(args.out_dir / "candidates_with_selection_flags.csv")},
        "next_step_not_taken": ("20_build_system.sh <md_system_name> <ligand_sdf>, then "
                                "30_run_md.sh <md_system_name> <seed> 100 for three velocity seeds"),
    }
    (args.out_dir / "md_ready_hit_selection_summary.json").write_text(json.dumps(summary, indent=2))

    log("\n" + "=" * 74)
    for k, v in funnel.items():
        log(f"  {k:42s} {v:6d}")
    log("=" * 74)
    if len(hits):
        cols = [c for c in ("md_rank", "coconut_id", "role", score_col,
                            "consensus_pred_pIC50", "nearest_train_tanimoto",
                            "np_likeness", "plip_n_interactions") if c in hits.columns]
        log(hits[cols].head(25).to_string(index=False))
    else:
        log("no compound satisfies all three criteria; nothing is MD-ready")
    log(f"\nwrote {hits_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
