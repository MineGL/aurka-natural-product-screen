#!/usr/bin/env python
"""y-randomisation of the AURKA graph-attention (GAT) arm.

Runs on the LOCAL machine so the docking workstation's GPU stays on the
molecular-dynamics replicates.

Why this is a reimplementation, and what that costs
---------------------------------------------------
The deployed pipeline never refits the graph arm: `predict_coconut_xgb_gat_
consensus_v5.py` RESTORES three checkpoints and validates them against saved
predictions. So unlike the XGBoost arm -- where y-randomisation could import
the pipeline's own `fit_xgb_ensemble` -- the graph arm has no validated refit
path, and its training had to be reconstructed.

Everything below that governs the fit is copied VERBATIM from the training
notebook (`GATmodelTraining_DeepAURKA_nested_split_balance_split_manifest.ipynb`)
rather than reconstructed: the model wrapper (`RobustGATModel`, huber loss),
the early-stopping loop, the normalisation transformer, the potency sample
weights and their constants, the prediction clipping, and the fold indexing.
Hyperparameters come from the deployed fold manifest.

The real-y control run is therefore the load-bearing check: the manifest records
each fold's stop-set R2 (0.4837, 0.5357, 0.5749) and the frozen-test R2 of the
deployed ensemble (0.5987). If the control reproduces those, the scrambled
baseline is comparable; the reported gap is what licenses the p value.

Scope limit, identical to the XGBoost arm: hyperparameters are held at their
selected values and only the FIT is repeated under permutation. The 48-trial
tuning is not re-run per permutation, so this bounds chance correlation in the
fit, not in model selection.

Usage: python yscramble_gat.py [n_iterations]   (0 = control only)
"""
from __future__ import annotations

import inspect
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

import dgl  # noqa: E402
import dgl.function as fn  # noqa: E402
import deepchem as dc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IN = json.load(open(os.path.join(HERE, "handoff/gat_inputs.json")))
OUT = os.path.join(HERE, "out")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# ---- notebook globals the verbatim functions close over -------------------
task_name = "pIC50"
smiles_column = "Smiles"
input_dataset = IN["gat_training_data.csv"]
device = "cpu"                      # dgl PyPI wheels are CPU-only
REGRESSION_LOSS_NAME = "huber"
ENABLE_POTENCY_SAMPLE_WEIGHTS = True
USE_SPLIT_MANIFEST = True
OUTER_SELECTION_SEEDS = [69, 101, 137]
EXTREME_WEIGHT_QUANTILES = (0.10, 0.90)
SHOULDER_WEIGHT_QUANTILES = (0.25, 0.75)
SHOULDER_WEIGHT_MULT = 1.15
EXTREME_WEIGHT_MULT = 1.45
MAX_SAMPLE_WEIGHT = 2.00
NORMALIZE_SAMPLE_WEIGHTS = True
FINAL_CV_FOLDS = 3
FINAL_MAX_EPOCHS = 220
FINAL_PATIENCE = 24
MIN_DELTA = 2e-3
PREDICTION_CLIP_STD_MULT = 0.75
PREDICTION_CLIP_MIN_MARGIN = 0.50
SEED = 69

split_manifest = pd.read_csv(IN["gat_split_manifest.csv"])

# ---- load the verbatim notebook code into this namespace -----------------
_g = globals()
for path in (IN["gat_model_def.py"], IN["gat_train_lib.py"], IN["gat_train_lib_extra.py"]):
    exec(compile(open(path).read(), path, "exec"), _g)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_folds(raw_pool, raw_test, fold_local_idx, manifest_rows, tag, only_folds=None):
    """Train the requested final-CV folds; return their frozen-test predictions."""
    per_fold = []
    members = []
    for row in manifest_rows.itertuples(index=False):
        f = int(row.fold_idx)
        if only_folds is not None and f not in only_folds:
            continue
        stop_idx = fold_local_idx[f - 1]
        train_idx = [i for i in range(len(raw_pool.y)) if i not in set(stop_idx)]
        train_raw = select_dataset(raw_pool, train_idx)
        stop_raw = select_dataset(raw_pool, stop_idx)
        # transform_from_train fits the NormalizationTransformer on the TRAIN
        # split and derives the potency sample weights from its raw targets, so
        # under permutation both follow the permuted response, as they must.
        bundle = transform_from_train(train_raw, stop_raw, raw_test)
        train_ds = bundle["train_dataset"]
        stop_ds = bundle["valid_dataset"]
        test_ds = bundle["test_dataset"]
        transformer = bundle["transformer"]
        feature_kwargs, _, _ = _gat_feature_kwargs_from_dataset(train_raw)
        params = params_from_mapping(row._asdict())
        seed_all(SEED + f)
        res = train_with_early_stopping(
            params=params, gat_feature_kwargs=feature_kwargs,
            train_dataset=train_ds, stop_dataset=stop_ds, transformer_obj=transformer,
            model_dir=os.path.join(OUT, "gat_ckpt", tag, "fold_%d" % f),
            max_epochs=FINAL_MAX_EPOCHS, patience=FINAL_PATIENCE, min_delta=MIN_DELTA,
        )
        bounds = prediction_clip_bounds(train_ds, transformer)
        pred = clip_predictions(
            transformer.untransform(res["model"].predict(test_ds)).flatten(), bounds)
        members.append(pred)
        per_fold.append({"fold": f, "best_epoch": res["best_epoch"],
                         "train_r2": res["train_r2"], "stop_r2": res["stop_r2"],
                         "clip_lower": bounds[0], "clip_upper": bounds[1]})
        print("    fold %d  epochs %3d  train_r2 %+.4f  stop_r2 %+.4f" % (
            f, res["best_epoch"], res["train_r2"], res["stop_r2"]), flush=True)
    return members, per_fold


def permuted_pool(raw_pool, y_pool, iter_idx):
    """Deterministic per-iteration permutation, identical across parallel workers."""
    rng = np.random.default_rng(SEED * 1000 + iter_idx)
    perm = rng.permutation(y_pool)
    ds = dc.data.NumpyDataset(X=raw_pool.X, y=perm.reshape(-1, 1),
                              w=raw_pool.w, ids=raw_pool.ids)
    _ensure_edge_features(ds)
    return ds


def prepare():
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True)
    loader = _build_csv_loader(featurizer)
    raw = _create_raw_dataset(loader, input_dataset)
    _ensure_edge_features(raw)
    pool_idx, _, test_idx = manifest_frozen_split_indices(raw)
    fold_local = manifest_final_fold_indices(pool_idx)
    raw_pool = select_dataset(raw, pool_idx)
    raw_test = select_dataset(raw, test_idx)
    man = pd.read_csv(IN["gat_fold_manifest.csv"]).sort_values("fold_idx")
    return raw_pool, raw_test, fold_local, man


def worker(worker_id, n_workers, n_iter):
    """Process the (iteration, fold) jobs whose index is ours, writing one file each."""
    jobs = [("control", f) for f in (1, 2, 3)]
    jobs += [(i, f) for i in range(n_iter) for f in (1, 2, 3)]
    mine = [j for k, j in enumerate(jobs) if k % n_workers == worker_id]
    jobdir = os.path.join(OUT, "gat_jobs")
    os.makedirs(jobdir, exist_ok=True)
    print("worker %d of %d: %d jobs" % (worker_id, n_workers, len(mine)), flush=True)
    raw_pool, raw_test, fold_local, man = prepare()
    y_pool = np.asarray(raw_pool.y, dtype=float).flatten()
    for tag, fold in mine:
        name = "%s_fold%d" % (tag, fold)
        dest = os.path.join(jobdir, name + ".json")
        if os.path.exists(dest):
            print("  skip %s (done)" % name, flush=True)
            continue
        t0 = time.time()
        pool = raw_pool if tag == "control" else permuted_pool(raw_pool, y_pool, int(tag))
        preds, per_fold = fit_folds(pool, raw_test, fold_local, man,
                                    tag=name, only_folds={fold})
        payload = dict(per_fold[0])
        payload["tag"] = str(tag)
        payload["prediction"] = [float(x) for x in preds[0]]
        payload["minutes"] = (time.time() - t0) / 60
        tmp = dest + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, dest)
        print("  wrote %s in %.1f min (stop_r2 %+.4f)" % (
            name, payload["minutes"], payload["stop_r2"]), flush=True)


def reduce_results(n_iter):
    jobdir = os.path.join(OUT, "gat_jobs")
    raw_pool, raw_test, fold_local, man = prepare()
    y_test = np.asarray(raw_test.y, dtype=float).flatten()
    rec = {"n_pool": int(len(raw_pool.y)), "n_test": int(len(y_test)), "device": device,
           "manifest_stop_r2": [float(x) for x in man["stop_r2"].tolist()],
           "deployed_ensemble_frozen_test_R2": 0.5987,
           "scope": ("hyperparameters held at their selected values; only the fit is repeated "
                     "under permutation, so this bounds chance correlation in the fit and not "
                     "in model selection -- the same scope as the XGBoost arm"),
           "reimplementation_note": ("the deployed pipeline never refits the graph arm (it restores "
                                     "checkpoints), so the fit was reconstructed: model wrapper, "
                                     "early-stopping loop, normalisation, potency sample weights and "
                                     "clipping copied verbatim from the training notebook, "
                                     "hyperparameters from the deployed fold manifest")}

    def gather(tag):
        out = []
        for f in (1, 2, 3):
            p = os.path.join(jobdir, "%s_fold%d.json" % (tag, f))
            if not os.path.exists(p):
                return None
            out.append(json.load(open(p)))
        return out

    ctrl = gather("control")
    if ctrl:
        pred = np.mean(np.vstack([np.array(c["prediction"]) for c in ctrl]), axis=0)
        rec["control_frozen_test_R2"] = float(r2_score(y_test, pred))
        rec["control_stop_r2"] = [c["stop_r2"] for c in ctrl]
        rec["control_best_epoch"] = [c["best_epoch"] for c in ctrl]
        rec["control_minus_deployed"] = rec["control_frozen_test_R2"] - 0.5987
    scr, done = [], []
    for i in range(n_iter):
        rows = gather(str(i))
        if not rows:
            continue
        pred = np.mean(np.vstack([np.array(r["prediction"]) for r in rows]), axis=0)
        scr.append(float(r2_score(y_test, pred)))
        done.append(i)
    if scr:
        s = np.array(scr)
        rec.update(scrambled_R2=s.tolist(), completed_iterations=len(s),
                   scrambled_R2_mean=float(s.mean()),
                   scrambled_R2_sd=float(s.std(ddof=1)) if len(s) > 1 else None,
                   scrambled_R2_min=float(s.min()), scrambled_R2_max=float(s.max()))
        if "control_frozen_test_R2" in rec:
            rec["p_value"] = float((s >= rec["control_frozen_test_R2"]).mean())
            rec["p_resolution"] = 1.0 / len(s)
    with open(os.path.join(OUT, "yscramble_gat.json"), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("scrambled_R2", "scope", "reimplementation_note")}, indent=1))


def main():
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "reduce"
    if mode == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "reduce":
        reduce_results(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        raise SystemExit("usage: yscramble_gat.py worker <id> <n_workers> <n_iter> | reduce <n_iter>")


if __name__ == "__main__":
    main()
