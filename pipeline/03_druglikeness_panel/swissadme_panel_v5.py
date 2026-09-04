#!/usr/bin/env python
"""SwissADME-equivalent druglikeness panel + validation-support floor (v5 screen).

Applied AFTER the sub-100 nM consensus criterion and the existing triage
(`triage_coconut_consensus_hits.py`), on its full 46,107-row output. Nothing is
dropped: every row keeps every flag, and the candidate sets are filtered views.

SwissADME itself is a web service with no bulk API, so its published rule set is
reimplemented here with RDKit. One substitution is unavoidable and is recorded in
the summary: SwissADME uses MLOGP for Lipinski and XLOGP3 for Muegge, neither of
which RDKit provides, so the Wildman-Crippen WLOGP is used throughout. WLOGP is
what SwissADME itself uses for Ghose and Egan, so those two are exact.

Rules implemented (all thresholds as published):
  Lipinski  MW <= 500, WLOGP <= 4.15 (MLOGP substituted), HBA(N+O) <= 10,
            HBD(NH+OH) <= 5; pass = at most one violation
  Ghose     160 <= MW <= 480, -0.4 <= WLOGP <= 5.6, 40 <= MR <= 130,
            20 <= heavy atoms <= 70
  Veber     rotatable bonds <= 10, TPSA <= 140
  Egan      WLOGP <= 5.88, TPSA <= 131.6
  Muegge    200 <= MW <= 600, -2 <= WLOGP <= 5 (XLOGP3 substituted), TPSA <= 150,
            rings <= 7, carbons > 4, heteroatoms > 1, rotatable bonds <= 15,
            HBA <= 10, HBD <= 5
  ESOL      Delaney log S; "soluble" taken as log S > -6

Gastrointestinal absorption is represented by the **Egan** criterion rather than
the BOILED-Egg ellipse: the ellipse parameters are not reproduced here, and Egan
is the published absorption filter SwissADME plots that ellipse over. Stated
plainly rather than approximated.

The bioavailability score is not reproduced. For neutral molecules Martin's
score of 0.55 is equivalent to passing Lipinski, which is already reported.

Validation-support floor
------------------------
The nearest-train ECFP4 Tanimoto floor is not a convention, it is measured. In
the 425-compound scaffold frozen test the *minimum* nearest-train similarity is
0.403 -- there is no held-out compound below it -- and R2 for the XGB+GAT pair by
similarity band is:

    [0.40,0.50)  n= 29   R2 0.013   Spearman 0.542
    [0.50,0.60)  n= 74   R2 0.334   Spearman 0.592
    [0.60,0.75)  n=219   R2 0.582   Spearman 0.720
    [0.75,1.01)  n=103   R2 0.628   Spearman 0.747

So 0.40 is the boundary of any validation evidence at all, and 0.50 is where
value-level accuracy becomes non-trivial. Both are reported; the primary set uses
0.40 because below it the model has never been tested.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski

RDLogger.DisableLog("rdApp.*")

# Measured on the 425-row v3 scaffold frozen test; see module docstring.
VALIDATION_SUPPORT_FLOOR = 0.403
ACCURACY_FLOOR = 0.50
FLOORS_REPORTED = (0.20, 0.25, 0.30, 0.35, 0.403, 0.50)
ESOL_SOLUBLE_LOGS = -6.0


def log(msg: str) -> None:
    print(msg, flush=True)


def esol_logs(mol: Chem.Mol, mw: float, clogp: float, rotb: int) -> float:
    """Delaney (2004) ESOL. AP = fraction of heavy atoms that are aromatic."""
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ap = aromatic / heavy if heavy else 0.0
    return 0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rotb - 0.74 * ap


def panel_row(mol: Chem.Mol) -> dict:
    mw = Descriptors.MolWt(mol)
    wlogp = Crippen.MolLogP(mol)
    mr = Crippen.MolMR(mol)
    tpsa = Descriptors.TPSA(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    # SwissADME counts HBA as N+O and HBD as NH+OH.
    hba = Descriptors.NOCount(mol)
    hbd = Descriptors.NHOHCount(mol)
    heavy = mol.GetNumHeavyAtoms()
    rings = Lipinski.RingCount(mol)
    n_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    n_het = sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ("C", "H"))

    lip_viol = int(mw > 500) + int(wlogp > 4.15) + int(hba > 10) + int(hbd > 5)
    ghose = bool(160 <= mw <= 480 and -0.4 <= wlogp <= 5.6 and 40 <= mr <= 130 and 20 <= heavy <= 70)
    veber = bool(rotb <= 10 and tpsa <= 140)
    egan = bool(wlogp <= 5.88 and tpsa <= 131.6)
    muegge = bool(
        200 <= mw <= 600 and -2 <= wlogp <= 5 and tpsa <= 150 and rings <= 7
        and n_c > 4 and n_het > 1 and rotb <= 15 and hba <= 10 and hbd <= 5
    )
    logs = esol_logs(mol, mw, wlogp, rotb)
    row = {
        "sa_MW": round(mw, 2), "sa_WLOGP": round(wlogp, 3), "sa_MR": round(mr, 2),
        "sa_TPSA": round(tpsa, 2), "sa_RotB": rotb, "sa_HBA_NO": hba, "sa_HBD_NHOH": hbd,
        "sa_heavy_atoms": heavy, "sa_rings": rings, "sa_n_carbon": n_c, "sa_n_heteroatom": n_het,
        "sa_lipinski_violations": lip_viol,
        "sa_lipinski_pass": lip_viol <= 1,
        "sa_ghose_pass": ghose, "sa_veber_pass": veber, "sa_egan_pass": egan,
        "sa_muegge_pass": muegge,
        "sa_esol_logS": round(logs, 3),
        "sa_esol_soluble": bool(logs > ESOL_SOLUBLE_LOGS),
    }
    row["sa_n_druglikeness_passed"] = int(row["sa_lipinski_pass"]) + int(ghose) + int(veber) + int(egan) + int(muegge)
    row["sa_all5_pass"] = row["sa_n_druglikeness_passed"] == 5
    # Egan stands in for GI absorption; see module docstring.
    row["sa_gi_absorption_high"] = egan
    row["sa_panel_pass"] = bool(row["sa_all5_pass"] and row["sa_esol_soluble"])
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triage-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--docking-capacity", type=int, default=1000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[1/4] reading {args.triage_csv.name}")
    df = pd.read_csv(args.triage_csv, low_memory=False)
    log(f"  {len(df)} rows, consensus pIC50 {df['consensus_pred_pIC50'].min():.2f}-{df['consensus_pred_pIC50'].max():.2f}")
    for col in ("canonical_smiles", "nearest_train_tanimoto", "consensus_rank",
                "has_disqualifying_alert", "is_skeleton_representative"):
        if col not in df.columns:
            raise SystemExit(f"triage CSV is missing required column {col!r}")

    log("[2/4] computing the SwissADME-equivalent panel")
    records, failed = [], 0
    for i, smi in enumerate(df["canonical_smiles"].astype(str), 1):
        if i % 5000 == 0:
            log(f"    {i}/{len(df)}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            records.append({})
            continue
        records.append(panel_row(mol))
    panel = pd.DataFrame(records, index=df.index)
    out = pd.concat([df, panel], axis=1)
    if failed:
        log(f"  {failed} SMILES failed RDKit parsing (panel columns left empty)")

    log("[3/4] applying gates")
    base = (
        out["sa_panel_pass"].fillna(False).astype(bool)
        & ~out["has_disqualifying_alert"].astype(bool)
        & out["is_skeleton_representative"].astype(bool)
    )
    out["v5_adme_eligible"] = base
    sim = out["nearest_train_tanimoto"]
    floor_counts = {f"{f:.3f}": int((base & (sim >= f)).sum()) for f in FLOORS_REPORTED}
    out["within_validation_support"] = base & (sim >= VALIDATION_SUPPORT_FLOOR)
    out["within_accuracy_support"] = base & (sim >= ACCURACY_FLOOR)

    primary = out[out["within_validation_support"]].sort_values("consensus_rank").copy()
    primary["set_rank"] = range(1, len(primary) + 1)
    extrap = out[base & ~out["within_validation_support"]].sort_values("consensus_rank").head(args.docking_capacity).copy()
    extrap["set_rank"] = range(1, len(extrap) + 1)

    panel_path = args.out_dir / "coconut_v5_swissadme_panel.csv"
    primary_path = args.out_dir / "docking_set_A_within_validation_support.csv"
    extrap_path = args.out_dir / "docking_set_B_extrapolation.csv"
    out.to_csv(panel_path, index=False)
    primary.to_csv(primary_path, index=False)
    extrap.to_csv(extrap_path, index=False)

    log("[4/4] summarising")
    summary = {
        "stage": "swissadme_equivalent_panel_v5",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.triage_csv),
        "input_rows": int(len(df)),
        "smiles_parse_failures": failed,
        "implementation": {
            "tool": "RDKit reimplementation of the published SwissADME rule set",
            "logp_substitution": "Wildman-Crippen WLOGP used for Lipinski (SwissADME uses MLOGP) and Muegge (XLOGP3); Ghose and Egan use WLOGP natively and are exact",
            "gi_absorption": "represented by the Egan criterion, not the BOILED-Egg ellipse",
            "bioavailability_score": "not reproduced; for neutral molecules a score of 0.55 is equivalent to passing Lipinski, which is reported",
            "solubility": f"Delaney ESOL log S; soluble taken as > {ESOL_SOLUBLE_LOGS}",
        },
        "individual_rule_pass_counts": {
            k: int(out[f"sa_{k}_pass"].fillna(False).sum())
            for k in ("lipinski", "ghose", "veber", "egan", "muegge")
        },
        "esol_soluble": int(out["sa_esol_soluble"].fillna(False).sum()),
        "all5_druglikeness_pass": int(out["sa_all5_pass"].fillna(False).sum()),
        "panel_pass_all5_and_soluble": int(out["sa_panel_pass"].fillna(False).sum()),
        "panel_pass_and_alerts_and_dedup": int(base.sum()),
        "validation_support": {
            "frozen_test_min_nearest_train_tanimoto": VALIDATION_SUPPORT_FLOOR,
            "note": "no held-out compound exists below this similarity, so predictions below it are untested rather than merely uncertain",
            "eligible_at_floor": floor_counts,
            "set_A_within_validation_support": int(len(primary)),
            "set_B_extrapolation_capped_at": int(len(extrap)),
        },
        "outputs": {
            "panel_csv": str(panel_path),
            "set_A_csv": str(primary_path),
            "set_B_csv": str(extrap_path),
        },
    }
    (args.out_dir / "swissadme_panel_summary.json").write_text(json.dumps(summary, indent=2))

    log("\n" + "=" * 70)
    log(f"input {len(df)} sub-100 nM triaged candidates")
    for k in ("lipinski", "ghose", "veber", "egan", "muegge"):
        log(f"  {k:10s} pass {summary['individual_rule_pass_counts'][k]:6d}")
    log(f"  ESOL soluble    {summary['esol_soluble']:6d}")
    log(f"  all 5 rules     {summary['all5_druglikeness_pass']:6d}")
    log(f"  panel pass      {summary['panel_pass_all5_and_soluble']:6d}")
    log(f"  + alerts/dedup  {summary['panel_pass_and_alerts_and_dedup']:6d}")
    log("  eligible by similarity floor:")
    for f, n in floor_counts.items():
        log(f"    Tanimoto >= {f}  {n:6d}")
    log(f"  -> set A (within validation support) {len(primary)}")
    log(f"  -> set B (extrapolation, capped)     {len(extrap)}")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
