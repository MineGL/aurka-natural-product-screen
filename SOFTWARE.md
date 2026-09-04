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
