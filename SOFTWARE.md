# Software versions

Every version below was read from the environments that actually executed this
pipeline, not from a specification. The `How determined` column records the
provenance of each entry so that any weakly-established version is visible as such.

## Stages 1, 3, 5, 7 — consensus prediction, druglikeness panel, score calibration, hit selection

Single conda environment, **Python 3.9.23**.

| Package | Version | How determined |
|---|---|---|
| numpy | 1.23.5 | `__version__` |
| pandas | 1.5.3 | `__version__` |
| scipy | 1.10.1 | `__version__` |
| rdkit | 2023.09.6 | `__version__` |
| xgboost | 2.1.4 | `__version__` |
| scikit-learn | 1.2.2 | `__version__` |
| pytorch | 2.1.2+cu121 (CUDA build 12.1) | `__version__`, `torch.version.cuda` |
| dgl | 1.1.2 | `__version__` |
| deepchem | 2.8.0 | `__version__` |
| tensorflow | 2.15.0 | `__version__` (present as a DeepChem dependency; not used by this pipeline) |

The GAT arm requires `torch`, `dgl` and `deepchem` together: its checkpoints were written
by the custom model class in `pipeline/01_consensus_prediction/gat_model_def.py`, and the
stock DeepChem GAT class will not restore them.

## Stage 6 — docking

| Software | Version | How determined |
|---|---|---|
| gnina | v1.3.3, `sm120` CUDA build | binary path and hash pinned in the pipeline |
| — binary SHA-256 | `9f50749a48ce9e1f1f90fba4284b8678964e5fef3a7f0568b0b15fe6f11b6bf1` | `sha256sum`, verified against the value asserted in the pipeline |
| — binary size | 141,008,328 bytes | `ls` |
| PoseBusters | 0.6.5 | `__version__` |
| — its environment | Python 3.12.3, RDKit 2026.03.5 | `__version__` |
| Open Babel | 3.2.0 | build-directory name in the pipeline source |

The gnina hash is published so that a reader can confirm they hold the identical build.
`sm120` is an architecture-specific CUDA build and will not load on a different GPU
architecture.

## Interaction profiling

| Software | Version | How determined |
|---|---|---|
| PLIP | 3.0.1 | environment name as provisioned — **weakly established**: the installed package exposes neither `__version__` nor importlib metadata, so this was not queried from the package itself |
| — its environment | Python 3.12.3 | `sys.version` |
| Open Babel | 3.2.0 | build-directory name in the pipeline source |

## Molecular dynamics — protocol validation (and the candidate round, not reported here)

| Software | Version | How determined |
|---|---|---|
| GROMACS | 2026.3 | `gmx --version` |
| — CUDA driver / runtime | 12.90 / 12.90 | `gmx --version` |
| AmberTools | 26.0 | conda package metadata |
| ParmEd | 4.3.1 | conda package metadata |
| Open Babel | 3.2.1 | conda package metadata — **note this differs from the 3.2.0 build used for docking and profiling** |

Force field and solvation, from the build script: protein **ff14SB**, water **TIP3P**,
ligand **GAFF2** with **AM1-BCC** charges, truncated octahedron, 0.15 M NaCl, 2 fs timestep
with LINCS on hydrogen-containing bonds and no hydrogen mass repartitioning.

## Two Open Babel versions

This is not an error in the record. Ligand preparation and interaction profiling use a
standalone Open Babel 3.2.0 build; the MD system-building toolchain uses conda's Open Babel
3.2.1. Anyone reproducing the work should expect protonation output to depend on which is used.

## Chance-correlation controls and figures (added after the screening run)

### y-randomisation, both arms — the screening environment, unchanged

Both `validation/yscramble_xgb.py` and `validation/yscramble_gat.py` run in the **same
conda environment that produced the screen**, so neither carries any version
substitution. Read from that environment:

| Package | Version |
|---|---|
| python | 3.9.23 |
| numpy | 1.23.5 |
| pandas | 1.5.3 |
| scikit-learn | 1.2.2 |
| rdkit | 2023.9.6 |
| xgboost | 2.1.4 |
| pytorch | 2.1.2 |
| dgl | 1.1.2 |
| dgllife | 0.3.1 |
| deepchem | 2.8.0 |

Both run on **CPU**. The graph arm is trained, not merely inferred, so it is the heavier
of the two: ten permutations plus a control is 33 independent fold-fits, which the driver
distributes over parallel workers (`worker <id> <n_workers> <n_iter>`, then `reduce`).
Each fold-fit writes its own result file, so the run is resumable and a killed worker
loses at most one fit.

An earlier attempt to run the graph arm on a separate 8 GB machine — to keep load off the
workstation while molecular dynamics ran — was abandoned: the workers were killed by the
kernel during featurisation, twice, with no error written. That attempt would also have
introduced two version substitutions (rdkit 2026.03.5 and dgllife 0.3.2). Running in the
screening environment instead removes both, and costs the workstation nothing that
matters, because these fits use only CPU while the molecular dynamics is GPU-bound.

### Figures — two environments

| Tool | Version | Where | Note |
|---|---|---|---|
| PyMOL | 3.1.0 | workstation, existing conda env | headless (`pymol -cq`), ray-traced |
| rdkit | 2023.9.6 | workstation, screening env | 2D hit depictions use the pipeline's OWN rdkit, so the structures come from the same toolkit version that produced the screen |
| matplotlib | 3.11.1 | analysis machine, python 3.11.16 | analysis panels: funnel, calibration, rank, profiling, criteria, species |
| numpy | 2.4.6 | analysis machine | analysis panels only — NumPy 2.x, unlike every other environment recorded here |
| pandas | 2.3.3 | analysis machine | |
| scipy | 1.17.1 | analysis machine | bootstrap and rank statistics |

`render_binding_modes.py` requires PyMOL on `PATH`; the system-wide PyMOL package on the
workstation is broken under Python 3.12 (it imports the removed `imp` module), so the
conda copy was used.
