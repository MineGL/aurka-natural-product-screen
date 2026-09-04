# AURKA natural-product virtual screening — code availability

Code accompanying a computational screen of the COCONUT natural-product library
against Aurora kinase A (AURKA). A ligand-based consensus QSAR model (equal-weight
XGBoost + graph-attention network, dual applicability domain, split-conformal
prediction intervals) was applied to 711,906 COCONUT compounds, of which 403,154 fell
inside both applicability domains and 403,019 were predicted; 46,107 compounds met the
sub-100 nM consensus criterion, and 4,363 survived a SwissADME-equivalent druglikeness
panel together with structural-alert and skeleton filters. A candidate round of 1,004
compounds was docked into the AURKA ATP site (PDB 5DPV) with GNINA 1.3.3 on GPU; 1,000
docking runs completed and 891 produced a physically valid (PoseBusters-ready) pose 1.
Poses were profiled for protein–ligand interactions with PLIP. Against a CNNaffinity
threshold of 7.3082 — calibrated by Youden's J on measured actives versus measured
inactives rather than chosen by hand — 5 compounds cleared the cut, and all 5 also
formed a hydrogen bond to the hinge residues Glu211 or Ala213. Three of the five are
deliberately retained synthetic positive controls (kinase inhibitors present in the
COCONUT release) and two are natural-product candidates. The molecular-dynamics
protocol was validated on the cognate ligand of 5DPV; **molecular dynamics of the
candidates has not been run**, and no code for it is included here. Nothing in this
repository, and nothing in the outputs it produced, constitutes a measurement or a
claim of biological activity for any compound: every quantity reported is a model
score, a docking score, or a geometric interaction count.

## Pipeline stages, scripts and status

Directory numbering follows pipeline order. The gaps (02, 04, 06) are the stages whose
code lives only on the author's workstation and is **not** included — see
[Not included in this repository](#not-included-in-this-repository).

| # | Stage | Script | Status |
|---|-------|--------|--------|
| 01 | Library prediction: equal-weight XGBoost + GAT consensus, dual applicability domain, 90% split-conformal intervals, bit-exact reproduction gates against saved reference predictions | `pipeline/01_consensus_prediction/predict_coconut_xgb_gat_consensus_v5.py`, `pipeline/01_consensus_prediction/gat_model_def.py` | complete |
| 02 | Medicinal-chemistry triage and novelty tiering of the 46,107-compound criterion set | `triage_coconut_consensus_hits.py` | complete — **code not included** |
| 03 | Druglikeness panel (RDKit reimplementation of the published SwissADME rule sets) and nearest-train-similarity validation-support floor | `pipeline/03_druglikeness_panel/swissadme_panel_v5.py` | complete |
| 04 | GNINA-CUDA docking into 5DPV: calibration round (measured actives/inactives) and candidate round (1,004 compounds), plus queue submission wrappers | `aurka_gnina_cuda_pipeline_v5.py`, `run_aurka_*_jobq.sh` | complete — **code not included** |
| 05 | Docking-score calibration: ROC / Youden threshold on measured actives vs measured inactives, with stratified bootstrap intervals on both AUC and the threshold | `pipeline/05_docking_score_calibration/define_cnn_cutoff_v4.py` | complete |
| 06 | PLIP protein–ligand interaction profiling of the candidate poses | `aurka_plip_v5.py` | complete — **code not included** |
| 07 | Hit selection under the pre-registered rule (pose-1 physical validity → calibrated CNNaffinity threshold → hinge hydrogen bond → rank by CNNaffinity); fails closed if interaction profiling is absent | `pipeline/07_hit_selection/select_md_ready_hits.py` | complete |
| — | MD protocol validation on the cognate ligand (SKE / JNJ-7706621) of 5DPV, three replicates | `md_protocol_validation/stage0_gate_v2.py` | complete |
| — | Molecular dynamics of the selected candidates | — | **not run; no code in this repository** |

Stage 07 consumes the calibrated threshold from stage 05 by reading
`cnn_cutoff_v4_summary.json`; it is never hardcoded. The MD protocol validation is a
side branch that gates the MD protocol itself, not a step in the screening cascade.

## Repository layout

```
.
├── README.md
├── LICENSE                                  MIT
├── requirements.txt
├── .gitignore
├── pipeline
│   ├── 01_consensus_prediction
│   │   ├── predict_coconut_xgb_gat_consensus_v5.py
│   │   └── gat_model_def.py                 must stay beside the script above
│   ├── 03_druglikeness_panel
│   │   └── swissadme_panel_v5.py
│   ├── 05_docking_score_calibration
│   │   └── define_cnn_cutoff_v4.py
│   └── 07_hit_selection
│       └── select_md_ready_hits.py
└── md_protocol_validation
    └── stage0_gate_v2.py
```

`gat_model_def.py` is imported as `from gat_model_def import RobustGATModel`, so the two
files in `01_consensus_prediction` must remain in the same directory (or that directory
must be on `PYTHONPATH`). It is a verbatim copy of the training notebook's model class:
`deepchem.models.GATModel` is not a drop-in substitute, because the checkpoints were
written by this subclass.

## Dependencies

`requirements.txt` lists the third-party Python packages the included scripts actually
import. **No versions are pinned**: none of the scripts or their docstrings records the
version it was run with, so pinning would mean guessing. The one recorded external tool
version is GNINA 1.3.3, used for the docking rounds — whose pipeline code is not in this
repository. The GAT arm additionally requires the DGL backend for PyTorch, and the
checkpoints it restores are not distributed here.

External binaries, not installable from `requirements.txt`:

| Tool | Used by | Note |
|------|---------|------|
| GROMACS (`gmx`) | `md_protocol_validation/stage0_gate_v2.py` | invoked as `gmx`; must be on `PATH` |
| GNINA 1.3.3 (CUDA build) | stage 04 (not included) | GPU docking |
| PLIP | stage 06 (not included) | interaction profiling |

The consensus-prediction script imports both `torch` and `tensorflow` because DeepChem's
loader touches both; only the PyTorch path is exercised.

## Path literals redacted for publication

The scripts are included **verbatim** with one exception: six absolute path string
literals that pointed into personal home directories were replaced. No logic, argument
handling, threshold, or comment was otherwise changed, and line counts are unchanged.

| File | Original literal (redacted) | Replaced with |
|------|------------------------------|---------------|
| `predict_coconut_xgb_gat_consensus_v5.py` | `DEEPAURKA_ROOT` fallback default | `"./DeepAURKA"` |
| `gat_model_def.py` | notebook directory in the docstring | `DeepAURKA project tree` |
| `stage0_gate_v2.py` | absolute `gmx` path | `"gmx"` |
| `stage0_gate_v2.py` | `RUNS_DIR` | `Path("md_aurka_v1/runs")` |
| `stage0_gate_v2.py` | `SYSTEMS_DIR` | `Path("md_aurka_v1/systems")` |
| `stage0_gate_v2.py` | `OUT_DIR` | `Path("md_stage0_gate_20260903")` |

Consequences for reproduction: stage 01 reads its model tree from the `DEEPAURKA_ROOT`
environment variable (set it, as shown below, and the redacted fallback is never used),
and `stage0_gate_v2.py` takes no arguments, so its four path constants must be edited to
local values — or the script must be run from a directory in which the relative paths
above resolve.

## Reproduction

Commands below use only flags the scripts actually accept. All scripts are Python 3 and
are run directly; none installs anything.

### Stage 01 — consensus prediction

Accepts `--input`, `--output-dir`, `--chunk-size` (default 512), `--max-chunks`,
`--validate-only`. `DEEPAURKA_ROOT` (frozen model tree, read-only) and `RERUN_ROOT` (all
writes) are read from the environment; `RERUN_ROOT` defaults to
`~/AURKA_manuscript/rescreen_xgb_gat_20260903`.

```bash
cd pipeline/01_consensus_prediction
export DEEPAURKA_ROOT=/path/to/DeepAURKA
export RERUN_ROOT=/path/to/rescreen_xgb_gat_20260903

# reproduction gates only: restore checkpoints and reproduce the saved
# reference predictions bit-exactly, predict nothing
python predict_coconut_xgb_gat_consensus_v5.py --validate-only

# full library pass (chunked, atomically committed, resumable)
python predict_coconut_xgb_gat_consensus_v5.py \
    --input "$DEEPAURKA_ROOT/virtual_screening/coconut_2025_12/applicability_domain_v3/coconut_v3_both_models_in_domain.csv" \
    --output-dir "$RERUN_ROOT/runs/consensus_xgb_gat_v5" \
    --chunk-size 512

# smoke test: first two chunks only
python predict_coconut_xgb_gat_consensus_v5.py --max-chunks 2 \
    --output-dir /tmp/consensus_smoke
```

Writes `coconut_dual_domain_consensus_predictions.csv`,
`coconut_dual_domain_consensus_ranked.csv`, `consensus_prediction_summary.json`, and
per-chunk `chunks/prediction_chunk_*.csv` with `chunk_*.json` state files.

### Stage 03 — druglikeness panel

Accepts `--triage-csv` (required), `--out-dir` (required), `--docking-capacity`
(default 1000). Input is the full 46,107-row output of the stage-02 triage script.

```bash
python pipeline/03_druglikeness_panel/swissadme_panel_v5.py \
    --triage-csv /path/to/triage_coconut_consensus_hits_output.csv \
    --out-dir    /path/to/out/swissadme_panel_v5 \
    --docking-capacity 1000
```

Writes `coconut_v5_swissadme_panel.csv` (every input row, every flag; nothing dropped),
`docking_set_A_within_validation_support.csv`,
`docking_set_B_extrapolation.csv`, and `swissadme_panel_summary.json`. The four
substitutions relative to the SwissADME web service (WLOGP for MLOGP and XLOGP3; Egan in
place of the BOILED-Egg ellipse; bioavailability score not reproduced) are declared in
the module docstring and recorded in the summary JSON.

### Stage 05 — docking-score calibration

Accepts `--ranked-csv` (required), `--labels-csv` (required), `--out-dir` (required),
`--n-boot` (default 2000), `--seed` (default 67). There are deliberately no default
input paths. `--labels-csv` is the calibration input carrying `coconut_id`, `role` and
`reference_pIC50`; labels are joined from it because the ranked CSV carries no `role`
column.

```bash
python pipeline/05_docking_score_calibration/define_cnn_cutoff_v4.py \
    --ranked-csv /path/to/calibration_round/gnina_cuda_results_ranked.csv \
    --labels-csv /path/to/aurka_calibration_actives_inactives.csv \
    --out-dir    /path/to/out/cnn_cutoff_v4 \
    --n-boot 2000 --seed 67
```

Writes `cnn_cutoff_v4_summary.json`, which carries the Youden threshold (7.3082 in the
reported run), AUC, and stratified bootstrap 95% intervals on both. Stage 07 reads this
file.

### Stage 07 — hit selection

Accepts `--ranked-csv`, `--plip-json`, `--input-csv`, `--panel-csv`, `--cutoff-json`,
`--out-dir` — all required. Builds no MD systems and launches no trajectory.

```bash
python pipeline/07_hit_selection/select_md_ready_hits.py \
    --ranked-csv  /path/to/candidate_round/gnina_cuda_results_ranked.csv \
    --plip-json   /path/to/candidate_round/plip/interactions.json \
    --input-csv   /path/to/aurka_setAB_1004.csv \
    --panel-csv   /path/to/out/swissadme_panel_v5/coconut_v5_swissadme_panel.csv \
    --cutoff-json /path/to/out/cnn_cutoff_v4/cnn_cutoff_v4_summary.json \
    --out-dir     /path/to/out/md_ready_hits
```

Writes `md_ready_hits.csv`, `candidates_with_selection_flags.csv` (all 1,000 rows with
every selection flag) and `md_ready_hit_selection_summary.json`. The surviving count is
also reported at both bootstrap CI bounds of the threshold, so the point estimate is not
treated as exact. Set A compounds are labelled `positive_control`, never `hit`.

### MD protocol validation (cognate ligand)

Takes no command-line arguments. Edit `GMX`, `RUNS_DIR`, `SYSTEMS_DIR` and `OUT_DIR` at
the top of the file first (see the redaction table above); `gmx` must be on `PATH` or
given an absolute path. Inputs under the runs directory are treated as strictly
read-only and are sha256-pinned into the output JSON.

```bash
python md_protocol_validation/stage0_gate_v2.py
```

Writes `stage0_gate_v2.json` and per-replicate `timeseries_seed<N>.csv` under `OUT_DIR`.
This analysis asks only whether the production protocol holds the cognate ligand of
5DPV in its crystallographic pose across three replicates; the primary statistic
(Glu211 occupancy alone, block SEM per replicate, no maximum taken over residues) is
fixed in the code before any value is read. Two documented traps are guarded by
fail-closed assertions: `pdb4amber` renumbering (Glu211 → resid 85, Ala213 → resid 87)
and periodic-boundary wrapping. **It validates the protocol, not the candidates — no
candidate trajectory exists.**

## Result artifacts

Deposited with the manuscript rather than in this repository; the scripts above
regenerate them from the pipeline inputs.

| File | Rows / entries | Produced by | Contents |
|------|----------------|-------------|----------|
| `AURKA_candidate_docking_ranked.csv` | 1,000 rows × 39 cols | stage 04 (code not included) | Candidate-round GNINA-CUDA results, ranked: CNNaffinity, CNNscore, Vina affinity, pose counts, PoseBusters validity, preparation provenance per compound |
| `AURKA_candidate_plip_interactions.json` | 891 complexes | stage 06 (code not included) | PLIP interaction profiles for every physically valid pose-1 complex, keyed by COCONUT id |
| `AURKA_criterion_calibration.csv` | 8 rows | stage 05 | Potency-criterion calibration: AUC and Youden threshold with bootstrap 95% intervals, per active definition |
| `AURKA_hinge_criterion_calibration.csv` | 5 rows | stage 05 / 06 | Hinge hydrogen-bond criterion: prevalence in measured actives vs inactives, odds ratio, Fisher p, Bonferroni survival across 5 tests |
| `AURKA_candidates_with_selection_flags.csv` | 1,000 rows × 68 cols | stage 07 | All docked candidates with every selection flag, ML prediction, ADME panel column and interaction call |
| `AURKA_survivors_both_filters.csv` | 5 rows × 38 cols | stage 07 | The five compounds clearing both the calibrated CNNaffinity threshold and the hinge hydrogen-bond criterion — 3 labelled `positive_control` (set A), 2 labelled `hit` (set B) |

## Not included in this repository

The following pipeline files exist only on the author's workstation and are **not**
included here. They are listed with their exact on-host paths, relative to
`<project_root>/rescreen_xgb_gat_20260903/`. No substitute, stub, or reimplementation has
been written for any of them — an invented script would be worse than an absent one.

| On-host path | Stage | What it does |
|--------------|-------|--------------|
| `docking/code/aurka_gnina_cuda_pipeline_v5.py` | 04 | GNINA-CUDA docking pipeline (receptor/ligand preparation, GPU docking, pose export, ranking) |
| `docking/code/aurka_plip_v5.py` | 06 | PLIP protein–ligand interaction profiling of docked poses |
| `docking/code/run_aurka_*_jobq.sh` | 04 | Queue submission wrappers for the docking rounds |
| `scripts/triage_coconut_consensus_hits.py` | 02 | Medicinal-chemistry triage and novelty tiering; part of the author's pre-existing codebase |

Also not included: the trained XGBoost and GAT checkpoints, the frozen model tree
referenced by `DEEPAURKA_ROOT`, the prepared receptor and ligand files, and the MD input
files under the runs and systems directories. `stage0_gate_v2.py` is included because the
cognate-ligand protocol validation completed; **no candidate MD run script or MD analysis
code is included, because that stage has not been run.**

## Data availability

The compound library is COCONUT (COlleCtion of Open NatUral producTs), release
2025-12 as used here; 711,906 compounds entered the screen. The receptor is
**PDB 5DPV**, an AURKA structure whose cognate ligand SKE (JNJ-7706621) provides the
hinge reference contacts (N1–Glu211 O, N3–Ala213 O) and the redocking control for the MD
protocol validation. Both are publicly available from their respective sources; neither
is redistributed in this repository. Measured AURKA potency values used for the
docking-score calibration come from the curated training/validation set of the ligand-based
model and are reported with the manuscript.

## Pushing this repository to GitHub

This repository was assembled in an environment with **no GitHub credential**, so nothing
has been created or pushed. From the extracted tree:

```bash
# 1. fill in the copyright holder in LICENSE, then:
git init
git add .
git commit -m "AURKA natural-product virtual screening: code availability"
git branch -M main

# 2a. with the GitHub CLI (creates the remote and pushes in one step)
gh repo create <owner>/aurka-np-screening --public --source=. --remote=origin --push

# 2b. or, having created an empty repository in the GitHub web UI
git remote add origin git@github.com:<owner>/aurka-np-screening.git
git push -u origin main
```

Then paste the resulting URL, and a release DOI if one is minted (e.g. by enabling the
Zenodo integration and cutting a tag), into the manuscript's code-availability
statement.

## License

MIT — see `LICENSE`. The copyright holder line is a placeholder and must be filled in
before publication.
