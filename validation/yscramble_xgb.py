#!/usr/bin/env python
"""True y-randomisation (y-scrambling) for the AURKA XGBoost arm.

Standard chance-correlation control: the response is permuted and the ENTIRE
model-building procedure is repeated on the permuted response, so the resulting
performance measures what the procedure can extract from noise. Permuting only
the test-set labels against fixed predictions is a different and weaker
statement; that is computed separately and reported as such.

The fit is NOT reimplemented here. `fit_xgb_ensemble` and `predict_xgb` are
imported from the deployed screening script, whose XGBoost arm reproduces the
saved library predictions bit-exactly, so the scrambled baseline is directly
comparable to the observed value. Reimplementing would have silently changed
`random_state` (42, not the pipeline seed) and dropped the per-fold
y-standardisation the manifest records.

Per iteration: permute pIC50 within the 2,106-compound training pool only,
refit all three folds, average, and score against the UNPERMUTED frozen test.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

RESCREEN = os.path.expanduser("RESCREEN_ROOT")
SCRIPT = RESCREEN + "/code/predict_coconut_xgb_gat_consensus_v5.py"
OUT = RESCREEN + "/runs/yscramble"
N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def load_pipeline():
    spec = importlib.util.spec_from_file_location("aurka_v5", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aurka_v5"] = mod
    spec.loader.exec_module(mod)
    return mod


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    v5 = load_pipeline()
    training, manifest = v5.load_and_validate_training_inputs()
    splits = manifest["frozen_split"].value_counts().to_dict()
    print("frozen_split values:", splits, flush=True)

    pool = manifest["frozen_split"].eq("train_valid_pool").to_numpy()
    test = ~pool
    test_smiles = training.loc[test, "Smiles"].astype(str).tolist()
    y_test = training.loc[test, "pIC50"].to_numpy(float)
    print("pool %d  test %d" % (int(pool.sum()), int(test.sum())), flush=True)

    models, means, stds = v5.fit_xgb_ensemble(training, manifest)
    real, _ = v5.predict_xgb(models, means, stds, test_smiles)
    observed = r2(y_test, real)
    print("refit-from-manifest frozen-test R2 = %.6f" % observed, flush=True)

    rec = {
        "observed_frozen_test_R2": observed,
        "n_pool": int(pool.sum()),
        "n_test": int(test.sum()),
        "n_iterations": N_ITER,
        "frozen_split_counts": {str(k): int(v) for k, v in splits.items()},
        "fit_source": "fit_xgb_ensemble imported from predict_coconut_xgb_gat_consensus_v5.py",
        "note": ("pIC50 permuted within the training pool only; frozen-test labels "
                 "untouched; all three folds refit per iteration"),
    }

    rng = np.random.default_rng(69)
    scrambled = []
    for i in range(N_ITER):
        perm = training.copy()
        vals = perm.loc[pool, "pIC50"].to_numpy(float)
        perm.loc[pool, "pIC50"] = rng.permutation(vals)
        m, mu, sd = v5.fit_xgb_ensemble(perm, manifest)
        pred, _ = v5.predict_xgb(m, mu, sd, test_smiles)
        scrambled.append(r2(y_test, pred))
        print("  iter %d/%d  scrambled R2 %.4f" % (i + 1, N_ITER, scrambled[-1]), flush=True)

    s = np.array(scrambled)
    rec.update(
        scrambled_R2=s.tolist(),
        scrambled_R2_mean=float(s.mean()),
        scrambled_R2_sd=float(s.std(ddof=1)),
        scrambled_R2_min=float(s.min()),
        scrambled_R2_max=float(s.max()),
        p_value=float((s >= observed).mean()),
    )
    with open(OUT + "/yscramble_xgb.json", "w") as fh:
        json.dump(rec, fh, indent=2)
    print("scrambled R2: mean %.4f sd %.4f max %.4f | p = %.4f" % (
        rec["scrambled_R2_mean"], rec["scrambled_R2_sd"],
        rec["scrambled_R2_max"], rec["p_value"]))


if __name__ == "__main__":
    main()
