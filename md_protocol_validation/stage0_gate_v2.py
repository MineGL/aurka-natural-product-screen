#!/usr/bin/env python3
"""AURKA stage-0 MD validation gate, v2 -- all three SKE_holo replicates.

Does the production protocol hold the cognate ligand (SKE / JNJ-7706621) in its
crystallographic pose?  SKE redocks to 0.734 A from the crystal pose, so there
is a known right answer and the gate is meaningful.

WHY v2 EXISTS.  v1 (analysis/stage0_gate.json) was computed when only seed 1
existed, and its hinge-occupancy pass came from taking a MAXIMUM over two
residues (Glu211 61.4%, Ala213 31.7%) against a 60% threshold -- i.e. the
statistic was selected after seeing the numbers, it cleared the bar by 1.4
points, and it carried no uncertainty.  v2 fixes exactly that: every estimator
below is a constant fixed BEFORE any value was looked at, Glu211 alone is the
primary statistic, no maximum over residues is ever taken, and each replicate
carries a block SEM.  v1 also silently reused a cached md_whole.xtc; on seed 1
that cache held 1424 frames (14.2 ns) of a run that had not finished, so v1's
numbers describe 14.2 ns, not 100 ns.  v2 never reuses a cache.

Two documented traps, both guarded by fail-closed assertions rather than trust:

1. RESIDUE NUMBERING.  pdb4amber renumbered the 5DPV construct 127-389 -> 1-263,
   so the hinge is NOT at 211/213 in the simulated system: Glu211 is resid 85
   and Ala213 is resid 87.  Querying the wrong residue returns "no hydrogen
   bonds" rather than an error, so the residue NAME at each resid is asserted
   before any hinge measurement is taken.
2. PERIODIC BOUNDARIES.  GROMACS wraps coordinates into the box, so a protein
   straddling a boundary reads as unfolded (CA RMSD in the tens of angstroms).
   Every trajectory is passed through trjconv -pbc mol -center -ur compact
   before a single distance is measured, and the reference frame is given the
   same treatment.

Modelled (not observed) residues 284-289 -> MD 158-163 are excluded from the
protein CA RMSD.  Thr288 is unphosphorylated in this system.

Inputs under md_aurka_v1/ are treated as strictly read-only and are hash-pinned
(sha256) into the output JSON.  Nothing is written outside OUT_DIR.

    python stage0_gate_v2.py            # writes OUT_DIR/stage0_gate_v2.json
"""
from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# PRE-SPECIFIED ESTIMATORS AND CRITERIA -- fixed before any value was inspected
# ---------------------------------------------------------------------------
SEEDS = (1, 2, 3)

WINDOW_START_PS = 20_000.0        # analysis window = last 80 ns of each 100 ns
WINDOW_END_PS = 100_000.0
N_BLOCKS = 4                      # 4 x 20 ns blocks -> mean +/- SEM across blocks
BLOCK_PS = 20_000.0
MIN_WINDOW_FRAMES = 8_000         # fail closed if the window is short

LIG_RMSD_PASS_A = 2.0             # window-mean ligand heavy-atom RMSD must be <
LIG_UNBOUND_A = 3.0               # also report fraction of window frames above
GLU211_OCC_PASS_PCT = 60.0        # PRIMARY hinge statistic must be >
N_PASS_FOR_GATE = 2               # PASS if >= 2 of 3 replicates pass

# Hinge H-bond geometry (GROMACS geometric definition, gmx hbond 2026):
# donor-acceptor distance <= HBOND_R nm and A-D-H angle <= HBOND_A degrees,
# donors/acceptors restricted to N and O (gmx defaults).  Occupancy = percentage
# of window frames carrying at least one such bond between LIG and the residue.
HBOND_R_NM = 0.35
HBOND_A_DEG = 30.0

# Ligand/protein RMSD: unweighted (-nomw) heavy-atom RMSD after a rot+trans
# least-squares fit.  Ligand is fitted on ALL protein CA (263 atoms); the
# protein CA RMSD is fitted on its own core selection (loop-excluded).
RMSD_FIT = "rot+trans"

# pdb4amber renumbering: MD resid = crystal resid - RESID_OFFSET
RESID_OFFSET = 126
GLU211_MD_RESID = 211 - RESID_OFFSET          # 85  -- PRIMARY
ALA213_MD_RESID = 213 - RESID_OFFSET          # 87  -- reported separately only
REBUILT_LOOP_CRYSTAL = (284, 289)             # modelled, not observed
REBUILT_LOOP_MD = (284 - RESID_OFFSET, 289 - RESID_OFFSET)   # 158-163
N_PROTEIN_RESIDUES = 263
LIGAND_RESNAME = "LIG"

# Expected selection sizes -- asserted, not assumed (fail closed on mismatch)
EXPECT = {
    "protein_atoms": 4322,
    "ca_all": N_PROTEIN_RESIDUES,                       # 263
    "ca_core": N_PROTEIN_RESIDUES - 6,                  # 257
    "lig_heavy": 27,
    "glu211_atoms": 15,                                 # GLU, Amber, with H
    "ala213_atoms": 10,                                 # ALA, Amber, with H
    "loop_ca": 6,
}

GMX = "gmx"
RUNS_DIR = Path("md_aurka_v1/runs")
SYSTEMS_DIR = Path("md_aurka_v1/systems")
OUT_DIR = Path("md_stage0_gate_20260903")
WORK_DIR = OUT_DIR / "work"
N_PARALLEL = 3

ENV = dict(os.environ, GMX_MAXBACKUP="-1", OMP_NUM_THREADS="2")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def run_gmx(args: list[str], stdin: str = "", cwd: Path | None = None,
            timeout: int = 14_400) -> subprocess.CompletedProcess:
    proc = subprocess.run([GMX, "--quiet", *args], input=stdin, text=True,
                          capture_output=True, cwd=str(cwd) if cwd else None,
                          env=ENV, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError("gmx %s failed (rc=%d): %s"
                           % (args[0], proc.returncode, proc.stderr[-1500:]))
    return proc


def jsonable(obj):
    """numpy scalars (np.bool_/integer/floating) are not JSON-serialisable."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError("not JSON serialisable: %r" % type(obj))


def read_xvg(path: Path) -> np.ndarray:
    rows = [[float(x) for x in line.split()]
            for line in path.read_text().splitlines()
            if line and line[0] not in "#@"]
    if not rows:
        raise RuntimeError("empty xvg: %s" % path)
    return np.asarray(rows, dtype=float)


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_sizes(structure: Path, selections: list[str], work: Path,
                    tag: str) -> list[int]:
    """Atom count of each selection, via gmx select -os."""
    out = work / ("sizes_%s.xvg" % tag)
    run_gmx(["select", "-s", str(structure), "-select", "; ".join(selections),
             "-os", str(out), "-on", str(work / ("groups_%s.ndx" % tag))],
            cwd=work)
    return [int(round(v)) for v in read_xvg(out)[0][1:]]


def block_stats(times: np.ndarray, values: np.ndarray) -> tuple[float, float, list[float]]:
    """Window mean and SEM across N_BLOCKS consecutive blocks of BLOCK_PS."""
    means = []
    for index in range(N_BLOCKS):
        low = WINDOW_START_PS + index * BLOCK_PS
        high = low + BLOCK_PS
        mask = (times >= low) & (times < high) if index < N_BLOCKS - 1 else \
               (times >= low) & (times <= high)
        if not mask.any():
            raise RuntimeError("block %d (%g-%g ps) is empty" % (index, low, high))
        means.append(float(values[mask].mean()))
    grand = float(np.mean(means))
    sem = float(np.std(means, ddof=1) / np.sqrt(N_BLOCKS))
    return grand, sem, means


# ---------------------------------------------------------------------------
# per-replicate analysis
# ---------------------------------------------------------------------------
def analyse_seed(seed: int) -> dict:
    run_dir = RUNS_DIR / ("SKE_holo_seed%d" % seed)
    work = WORK_DIR / ("seed%d" % seed)
    work.mkdir(parents=True, exist_ok=True)
    record: dict = {"seed": seed, "run": run_dir.name, "status": "OK",
                    "assertions": {}, "inputs": {}}
    assertions = record["assertions"]

    tpr, xtc, npt2 = run_dir / "md.tpr", run_dir / "md.xtc", run_dir / "npt2.gro"
    for path in (tpr, xtc, npt2):
        if not path.is_file():
            record["status"] = "MISSING_INPUT"
            record["reason"] = "missing %s" % path.name
            return record

    # Hash-pin the read-only inputs.
    for path in (tpr, xtc, npt2):
        record["inputs"][path.name] = {
            "path": str(path), "bytes": path.stat().st_size,
            "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime(path.stat().st_mtime)),
            "sha256": sha256(path)}

    # --- TRAP 1: assert the renumbering by residue NAME before measuring ------
    sel_full = [
        "resid 1 to %d" % N_PROTEIN_RESIDUES,
        "resid 1 to %d or resname %s" % (N_PROTEIN_RESIDUES, LIGAND_RESNAME),
        "resid %d" % GLU211_MD_RESID,
        "resid %d and resname GLU" % GLU211_MD_RESID,
        "resid %d" % ALA213_MD_RESID,
        "resid %d and resname ALA" % ALA213_MD_RESID,
    ]
    (n_prot, n_protlig, n_e85, n_e85_glu,
     n_a87, n_a87_ala) = selection_sizes(tpr, sel_full, work, "full")

    assertions["protein_atoms_4322"] = (n_prot == EXPECT["protein_atoms"], n_prot)
    assertions["glu211_is_resid85_GLU"] = (
        n_e85 > 0 and n_e85 == n_e85_glu == EXPECT["glu211_atoms"], [n_e85, n_e85_glu])
    assertions["ala213_is_resid87_ALA"] = (
        n_a87 > 0 and n_a87 == n_a87_ala == EXPECT["ala213_atoms"], [n_a87, n_a87_ala])

    # --- TRAP 2: make molecules whole and centre the protein, no cache reuse --
    full_ndx = work / "groups_full.ndx"          # group 0 = protein, 1 = prot+LIG
    whole = work / "md_whole_protlig.xtc"
    ref = work / "ref_protlig.gro"
    if whole.is_file():
        whole.unlink()                            # never reuse a cached whole traj
    run_gmx(["trjconv", "-s", str(tpr), "-f", str(xtc), "-n", str(full_ndx),
             "-o", str(whole), "-pbc", "mol", "-center", "-ur", "compact"],
            stdin="0\n1\n", cwd=work)
    run_gmx(["trjconv", "-s", str(tpr), "-f", str(npt2), "-n", str(full_ndx),
             "-o", str(ref), "-pbc", "mol", "-center", "-ur", "compact"],
            stdin="0\n1\n", cwd=work)
    record["pbc_correction"] = ("trjconv -pbc mol -center -ur compact, "
                                "centred on protein, protein+LIG written; "
                                "reference frame given the same treatment")

    # Subset topology (protein + ligand) so hbond/rms carry real masses+elements.
    protlig_tpr = work / "protlig.tpr"
    if protlig_tpr.is_file():
        protlig_tpr.unlink()
    run_gmx(["convert-tpr", "-s", str(tpr), "-n", str(full_ndx),
             "-o", str(protlig_tpr)], stdin="1\n", cwd=work)

    # Re-assert on the subset topology used for every measurement.
    sel_sub = [
        "name CA",
        "name CA and not resid %d to %d" % REBUILT_LOOP_MD,
        "resname %s and mass > 2" % LIGAND_RESNAME,
        "resid %d and resname GLU" % GLU211_MD_RESID,
        "resid %d and resname ALA" % ALA213_MD_RESID,
        "name CA and resid %d to %d" % REBUILT_LOOP_MD,
    ]
    n_ca, n_core, n_lig, n_e85s, n_a87s, n_loop = selection_sizes(
        protlig_tpr, sel_sub, work, "sub")
    assertions["ca_all_263"] = (n_ca == EXPECT["ca_all"], n_ca)
    assertions["ca_core_257_loop_excluded"] = (n_core == EXPECT["ca_core"], n_core)
    assertions["rebuilt_loop_ca_6"] = (n_loop == EXPECT["loop_ca"], n_loop)
    assertions["ligand_heavy_27"] = (n_lig == EXPECT["lig_heavy"], n_lig)
    assertions["subset_glu211_GLU"] = (n_e85s == EXPECT["glu211_atoms"], n_e85s)
    assertions["subset_ala213_ALA"] = (n_a87s == EXPECT["ala213_atoms"], n_a87s)

    failed = [key for key, (ok, _) in assertions.items() if not ok]
    if failed:
        record["status"] = "ASSERTION_FAILED"
        record["failed_assertions"] = failed
        return record

    sub_ndx = work / "groups_sub.ndx"    # 0 CA, 1 CA_core, 2 LIG_heavy, 3.. hinge
    # Ligand heavy-atom RMSD, fitted on all protein CA (group 0 -> group 2).
    lig_xvg = work / "rmsd_ligand.xvg"
    run_gmx(["rms", "-s", str(ref), "-f", str(whole), "-n", str(sub_ndx),
             "-o", str(lig_xvg), "-nomw", "-fit", RMSD_FIT, "-ng", "1"],
            stdin="0\n2\n", cwd=work)
    # Protein CA RMSD on the loop-excluded core, fitted on the same core.
    ca_xvg = work / "rmsd_protein_ca_core.xvg"
    run_gmx(["rms", "-s", str(ref), "-f", str(whole), "-n", str(sub_ndx),
             "-o", str(ca_xvg), "-nomw", "-fit", RMSD_FIT, "-ng", "1"],
            stdin="1\n1\n", cwd=work)

    # Hinge H-bonds: LIG vs Glu211 (PRIMARY) and LIG vs Ala213 (reported only).
    hb_paths = {}
    for label, resid in (("glu211", GLU211_MD_RESID), ("ala213", ALA213_MD_RESID)):
        out = work / ("hbnum_%s.xvg" % label)
        run_gmx(["hbond", "-s", str(protlig_tpr), "-f", str(whole),
                 "-r", "resname %s" % LIGAND_RESNAME, "-t", "resid %d" % resid,
                 "-num", str(out), "-hbr", str(HBOND_R_NM),
                 "-hba", str(HBOND_A_DEG)], cwd=work)
        hb_paths[label] = out

    # --- assemble time series on a common frame grid --------------------------
    lig = read_xvg(lig_xvg)
    cac = read_xvg(ca_xvg)
    hb_g = read_xvg(hb_paths["glu211"])
    hb_a = read_xvg(hb_paths["ala213"])
    times = lig[:, 0]
    for name, arr in (("protein_ca", cac), ("hb_glu211", hb_g), ("hb_ala213", hb_a)):
        if arr.shape[0] != times.shape[0]:
            record["status"] = "FRAME_MISMATCH"
            record["reason"] = "%s has %d frames, ligand RMSD has %d" % (
                name, arr.shape[0], times.shape[0])
            return record

    lig_a = lig[:, 1] * 10.0          # nm -> A
    ca_a = cac[:, 1] * 10.0
    g_bound = (hb_g[:, 1] > 0).astype(float)
    a_bound = (hb_a[:, 1] > 0).astype(float)

    record["frames_total"] = int(times.size)
    record["t_first_ps"], record["t_last_ps"] = float(times[0]), float(times[-1])
    assertions["trajectory_reaches_100ns"] = (times[-1] >= WINDOW_END_PS - 50.0,
                                              float(times[-1]))

    window = (times >= WINDOW_START_PS) & (times <= WINDOW_END_PS)
    n_window = int(window.sum())
    record["frames_in_window"] = n_window
    assertions["window_frames_ge_%d" % MIN_WINDOW_FRAMES] = (
        n_window >= MIN_WINDOW_FRAMES, n_window)

    failed = [key for key, (ok, _) in assertions.items() if not ok]
    if failed:
        record["status"] = "ASSERTION_FAILED"
        record["failed_assertions"] = failed
        return record

    # --- pre-specified statistics -------------------------------------------
    lig_mean, lig_sem, lig_blocks = block_stats(times, lig_a)
    ca_mean, ca_sem, ca_blocks = block_stats(times, ca_a)
    g_mean, g_sem, g_blocks = block_stats(times, g_bound * 100.0)
    a_mean, a_sem, a_blocks = block_stats(times, a_bound * 100.0)

    record.update({
        "ligand_rmsd_mean_A": round(lig_mean, 3),
        "ligand_rmsd_sem_A": round(lig_sem, 3),
        "ligand_rmsd_block_means_A": [round(v, 3) for v in lig_blocks],
        "ligand_rmsd_max_A": round(float(lig_a[window].max()), 3),
        "ligand_rmsd_window_max_over_full_traj_A": round(float(lig_a.max()), 3),
        "ligand_frac_frames_above_3A": round(
            float((lig_a[window] > LIG_UNBOUND_A).mean()), 5),
        "protein_ca_rmsd_mean_A": round(ca_mean, 3),
        "protein_ca_rmsd_sem_A": round(ca_sem, 3),
        "protein_ca_rmsd_block_means_A": [round(v, 3) for v in ca_blocks],
        "glu211_occupancy_pct": round(g_mean, 2),
        "glu211_occupancy_sem_pct": round(g_sem, 2),
        "glu211_occupancy_block_pct": [round(v, 2) for v in g_blocks],
        "ala213_occupancy_pct": round(a_mean, 2),
        "ala213_occupancy_sem_pct": round(a_sem, 2),
        "ala213_occupancy_block_pct": [round(v, 2) for v in a_blocks],
        "mean_hbonds_glu211": round(float(hb_g[window, 1].mean()), 3),
        "mean_hbonds_ala213": round(float(hb_a[window, 1].mean()), 3),
    })
    record["passes"] = bool(lig_mean < LIG_RMSD_PASS_A
                            and g_mean > GLU211_OCC_PASS_PCT)
    record["pass_criteria"] = ("ligand window-mean RMSD < %.1f A AND Glu211 "
                              "occupancy > %.0f%% (Ala213 never substituted, "
                              "no maximum over residues)"
                              % (LIG_RMSD_PASS_A, GLU211_OCC_PASS_PCT))

    csv = OUT_DIR / ("timeseries_seed%d.csv" % seed)
    header = "time_ps,ligand_rmsd_A,protein_ca_core_rmsd_A,hbonds_glu211,hbonds_ala213"
    np.savetxt(csv, np.column_stack([times, lig_a, ca_a, hb_g[:, 1], hb_a[:, 1]]),
               delimiter=",", header=header, comments="", fmt="%.4f")
    record["timeseries_csv"] = csv.name
    record["assertions"] = {k: {"ok": ok, "value": val}
                            for k, (ok, val) in assertions.items()}
    return record


def check_apo() -> dict:
    """The apo arm was assigned to a second host (KNIME).  Confirm, never fill in."""
    run_dirs = sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())
    apo_runs = [n for n in run_dirs if "apo" in n.lower()]
    apo_system = SYSTEMS_DIR / "apo"
    built = sorted(p.name for p in apo_system.iterdir()) if apo_system.is_dir() else []
    return {
        "status": "not assessed",
        "runs_dir_contents": run_dirs,
        "apo_run_dirs_on_this_host": apo_runs,
        "apo_system_built_on_this_host": bool(built),
        "apo_system_files": built,
        "note": ("The apo system is parameterised on this host (systems/apo/) but "
                 "no apo production run exists in runs/; the apo replicates were "
                 "assigned to the second GPU host (KNIME). Glu211 occupancy "
                 "therefore has NO measured background to be compared against in "
                 "this gate: the reported occupancies are absolute values against "
                 "a pre-specified 60% threshold, not an enrichment over apo. Not "
                 "substituted, simulated or estimated here."),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with futures.ThreadPoolExecutor(max_workers=N_PARALLEL) as pool:
        submitted = {pool.submit(analyse_seed, s): s for s in SEEDS}
        records = {}
        for future in futures.as_completed(submitted):
            seed = submitted[future]
            try:
                records[seed] = future.result()
            except Exception as exc:                     # fail closed, never guess
                records[seed] = {"seed": seed, "status": "ERROR",
                                 "reason": "%s: %s" % (type(exc).__name__,
                                                       str(exc)[:600])}
            print("  seed %d -> %s" % (seed, records[seed].get("status")), flush=True)

    replicates = [records[s] for s in SEEDS]
    analysed = [r for r in replicates if r.get("status") == "OK"]
    passing = [r for r in analysed if r.get("passes")]
    if len(analysed) < len(SEEDS):
        verdict = "INCOMPLETE"
    elif len(passing) >= N_PASS_FOR_GATE:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    report = {
        "gate": ("SKE_holo stage-0 protocol validation: ligand window-mean "
                 "heavy-atom RMSD < %.1f A AND Glu211 (MD resid %d) H-bond "
                 "occupancy > %.0f%% in >= %d of %d replicates"
                 % (LIG_RMSD_PASS_A, GLU211_MD_RESID, GLU211_OCC_PASS_PCT,
                    N_PASS_FOR_GATE, len(SEEDS))),
        "verdict": verdict,
        "n_analysed": len(analysed),
        "n_passing": len(passing),
        "generated_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "estimators_prespecified": {
            "analysis_window_ps": [WINDOW_START_PS, WINDOW_END_PS],
            "analysis_window": "last 80 ns of each 100 ns replicate",
            "block_averaging": "%d x %g ns blocks; mean +/- SEM across block means"
                               % (N_BLOCKS, BLOCK_PS / 1000.0),
            "primary_hinge_statistic": "Glu211 = MD resid %d" % GLU211_MD_RESID,
            "secondary_hinge_statistic": "Ala213 = MD resid %d, reported separately"
                                         % ALA213_MD_RESID,
            "no_maximum_over_residues": True,
            "hbond_definition": ("gmx hbond 2026 geometric: donor-acceptor <= %.2f nm, "
                                 "A-D-H angle <= %.0f deg, donors/acceptors N and O; "
                                 "occupancy = %% of window frames with >= 1 such bond"
                                 % (HBOND_R_NM, HBOND_A_DEG)),
            "ligand_rmsd": ("heavy atoms (27, mass > 2 amu), unweighted, after "
                            "rot+trans least-squares fit on all 263 protein CA"),
            "ligand_unbound_threshold_A": LIG_UNBOUND_A,
            "protein_rmsd": ("CA excluding modelled residues %d-%d (MD %d-%d), "
                             "fitted on the same core selection"
                             % (*REBUILT_LOOP_CRYSTAL, *REBUILT_LOOP_MD)),
            "reference": ("each replicate's own npt2.gro (= the coordinates grompp "
                          "wrote into md.tpr, i.e. the start of production), "
                          "PBC-corrected the same way as the trajectory"),
            "pass_condition": ("per replicate: ligand mean RMSD < %.1f A AND Glu211 "
                               "occupancy > %.0f%%; gate PASS if >= %d of %d pass"
                               % (LIG_RMSD_PASS_A, GLU211_OCC_PASS_PCT,
                                  N_PASS_FOR_GATE, len(SEEDS))),
        },
        "trap_notes": {
            "resid_mapping": ("pdb4amber renumbered 5DPV 127-389 -> 1-263: Glu211 is "
                              "MD resid %d, Ala213 is MD resid %d. Asserted by "
                              "residue NAME and atom count on both the production "
                              "tpr and the protein+ligand subset tpr before any "
                              "hinge measurement." % (GLU211_MD_RESID, ALA213_MD_RESID)),
            "pbc": ("trjconv -pbc mol -center -ur compact applied to every "
                    "trajectory AND to the reference frame before measurement; no "
                    "cached md_whole.xtc is ever reused (the v1 cache on seed 1 held "
                    "only 1424 frames / 14.2 ns of an unfinished run)."),
            "rebuilt_loop": ("Modelled residues %d-%d (MD %d-%d) excluded from the "
                             "protein CA RMSD; Thr288 unphosphorylated."
                             % (*REBUILT_LOOP_CRYSTAL, *REBUILT_LOOP_MD)),
        },
        "v1_comparison": {
            "v1_file": "md_aurka_v1/analysis/stage0_gate.json",
            "v1_defect": ("hinge pass came from max(Glu211 61.4%, Ala213 31.7%) vs a "
                          "60% threshold -- statistic chosen after seeing the values, "
                          "1.4 points of margin, no uncertainty; and it was computed "
                          "from a 14.2 ns cached whole-trajectory, not 100 ns."),
            "v2_fix": ("estimators fixed in advance, Glu211 alone is primary, block "
                       "SEMs reported, all three replicates, no cache reuse."),
        },
        "apo_baseline": check_apo(),
        "replicates": replicates,
        "software": {"gmx": GMX, "python": sys.version.split()[0],
                     "numpy": np.__version__},
    }
    report["gmx_version"] = subprocess.run(
        [GMX, "--version"], capture_output=True, text=True, env=ENV
    ).stdout.split("GROMACS version:")[-1].strip().splitlines()[0]

    out = OUT_DIR / "stage0_gate_v2.json"
    out.write_text(json.dumps(report, indent=2, default=jsonable) + "\n",
                   encoding="utf-8")

    for rec in replicates:
        if rec.get("status") != "OK":
            print("  seed %d  %s  %s" % (rec["seed"], rec["status"],
                                         rec.get("reason", rec.get("failed_assertions", ""))))
            continue
        print("  seed %d  lig %.2f+/-%.2f A  >3A %.1f%%  CA %.2f+/-%.2f A  "
              "Glu211 %.1f+/-%.1f%%  Ala213 %.1f%%  %s"
              % (rec["seed"], rec["ligand_rmsd_mean_A"], rec["ligand_rmsd_sem_A"],
                 100 * rec["ligand_frac_frames_above_3A"],
                 rec["protein_ca_rmsd_mean_A"], rec["protein_ca_rmsd_sem_A"],
                 rec["glu211_occupancy_pct"], rec["glu211_occupancy_sem_pct"],
                 rec["ala213_occupancy_pct"], "PASS" if rec["passes"] else "FAIL"))
    print("\nVERDICT: %s  (%d of %d replicates pass; apo baseline: not assessed)"
          % (verdict, len(passing), len(SEEDS)))
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
