#!/usr/bin/env python
"""Re-predict the dual-in-domain COCONUT library with an XGB + GAT consensus (v5).

Derived from ``predict_coconut_xgb_afp_consensus_v3.py`` by replacing the
AttentiveFP arm with GAT and the R2-proportional weighting with equal weights.
The XGBoost arm, the reproduction gates, the chunking/atomic-commit logic and
the seeding are unchanged from v3.

Why this differs from v3
------------------------
* **GAT replaces AFP.** On the 425-row v3 frozen test AFP scores R2 0.3283 --
  last of the five models trained -- and -0.020 on the 85 chemically distant
  compounds (worse than predicting the mean). GAT scores 0.5987 and 0.2892.
  Equal-weight XGB+GAT reaches 0.6153 against 0.5620 for the v3 XGB+AFP pair --
  and 0.5620 is below XGB's own 0.5947, i.e. the v3 consensus underperformed its
  stronger member (it still beat AFP alone at 0.3283).
* **Equal weights, not R2-proportional.** Variance-minimising weights for this
  pair are 0.477/0.523, so equal weighting is the optimum rather than a
  simplification -- and because it requires no fitting, no held-out data is
  consumed to set it. (v3 derived its weights from frozen-test R2, which
  compromised that set as an evaluation.)
* **Prediction intervals are emitted.** The 90% split-conformal half-width of
  this rule is 1.5789 pIC50 units, calibrated on the frozen test. A confident
  sub-100 nM claim would need a predicted pIC50 >= 8.579; nothing in the
  library reached 7.75 in v3. Downstream selection must therefore rank and take
  a capacity-driven depth, never threshold on absolute predicted potency.
* **Graph-featurization failures fall back to XGB** instead of dropping the
  compound (v3 lost 135 rows this way), recorded in a separate column so the
  two scales are never silently mixed.

Reads models and reference predictions from the frozen v3 tree read-only and
writes only under the rerun directory.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import deepchem as dc
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from xgboost import XGBRegressor

# Verbatim copy of the training notebook's GAT wrapper; see gat_model_def.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gat_model_def import RobustGATModel  # noqa: E402

# The notebook trained with a huber regression loss. Irrelevant at inference
# time, but construction must match what wrote the checkpoints.
REGRESSION_LOSS_NAME = "huber"


# Frozen v3 source tree: models, manifests, reference predictions, training
# data and the applicability-domain output. Read-only -- never written to.
PROJECT_ROOT = Path(
    os.environ.get("DEEPAURKA_ROOT", "./DeepAURKA")
).resolve()
# Rerun tree under the manuscript directory: all writes land here.
RERUN_ROOT = Path(
    os.environ.get(
        "RERUN_ROOT",
        str(Path.home() / "AURKA_manuscript" / "rescreen_xgb_gat_20260903"),
    )
).resolve()
AD_DIR = (
    PROJECT_ROOT
    / "virtual_screening"
    / "coconut_2025_12"
    / "applicability_domain_v3"
)
DEFAULT_INPUT = AD_DIR / "coconut_v3_both_models_in_domain.csv"
DEFAULT_OUTPUT_DIR = RERUN_ROOT / "runs" / "consensus_xgb_gat_v5"
TRAINING_DATA = (
    PROJECT_ROOT
    / "dataCleaningOutputs_parentStandardized_v3"
    / "aurka_training_parent_standardized_v3_20260611_v3_strict_model_ready.csv"
)
SPLIT_MANIFEST = (
    PROJECT_ROOT
    / "split_manifests"
    / "aurka_strict_20260611_v3_balanced_scaffold_split_manifest.csv"
)
XGB_MANIFEST = (
    PROJECT_ROOT
    / "externalValidationApplicability_mlxg_split_manifest"
    / "external_validation_applicability_model_manifest_20260629_091844.csv"
)
GAT_MANIFEST = (
    PROJECT_ROOT
    / "externalValidationApplicability_balance_split_manifest"
    / "external_validation_applicability_model_manifest_20260616_230349.csv"
)
XGB_EXTERNAL = (
    PROJECT_ROOT
    / "externalValidationApplicability_mlxg_split_manifest"
    / "external_validation_applicability_predictions_20260629_091844.csv"
)
GAT_EXTERNAL = (
    PROJECT_ROOT
    / "externalValidationApplicability_balance_split_manifest"
    / "external_validation_applicability_predictions_20260616_230349.csv"
)

SEED = 69
FP_RADIUS = 3
FP_SIZE = 1024

# Equal weights: the variance-minimising split for this pair is 0.477/0.523
# (from frozen-test residuals, resid correlation 0.908, sd 0.958/0.954), so
# 0.5/0.5 is the optimum and needs no fitting.
XGB_CONSENSUS_WEIGHT = 0.5
GAT_CONSENSUS_WEIGHT = 0.5

# Reference performance of THIS rule on the 425-row v3 frozen test, computed as
# 0.5*XGB + 0.5*GAT from the saved per-model test predictions
# (aurka_test_pred_vs_actual_{MLXG_split,GAT_balance}_manifest.csv).
FROZEN_TEST_N = 425
FROZEN_TEST_R2 = 0.615296
FROZEN_TEST_RMSE = 0.933800
FROZEN_TEST_SPEARMAN = 0.743367
# Split-conformal half-widths of |residual| on the same 425 rows.
CONFORMAL_HALF_WIDTH = {0.80: 1.138714263, 0.90: 1.578859995, 0.95: 1.907426396}
CONFORMAL_LEVEL = 0.90
# Low-similarity subset (nearest-train Tanimoto < 0.5860387, n = 85).
LOW_SIM_R2 = 0.335091
LOW_SIM_CONFORMAL_HALF_WIDTH_90 = 1.410448746


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    torch.manual_seed(SEED)
    dgl.seed(SEED)
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    if not hasattr(fn, "copy_edge") and hasattr(fn, "copy_e"):
        fn.copy_edge = fn.copy_e
    if not hasattr(fn, "src_mul_edge") and hasattr(fn, "u_mul_e"):
        fn.src_mul_edge = fn.u_mul_e


def circular_fingerprints(smiles: list[str]) -> np.ndarray:
    featurizer = dc.feat.CircularFingerprint(size=FP_SIZE, radius=FP_RADIUS)
    matrix = np.asarray(featurizer.featurize(smiles), dtype=np.float32)
    if matrix.shape != (len(smiles), FP_SIZE):
        raise RuntimeError(
            f"Expected fingerprint matrix {(len(smiles), FP_SIZE)}, got {matrix.shape}"
        )
    return matrix


def load_and_validate_training_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    training = pd.read_csv(TRAINING_DATA)
    manifest = pd.read_csv(SPLIT_MANIFEST)
    if len(training) != len(manifest):
        raise RuntimeError("Training data and split manifest row counts differ")
    if not training["Smiles"].astype(str).equals(manifest["Smiles"].astype(str)):
        raise RuntimeError("Training data and split manifest SMILES order differs")
    if not np.allclose(training["pIC50"], manifest["pIC50"], rtol=1e-8, atol=1e-8):
        raise RuntimeError("Training data and split manifest pIC50 values differ")
    return training, manifest


def fit_xgb_ensemble(
    training: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[list[XGBRegressor], list[float], list[float]]:
    model_manifest = pd.read_csv(XGB_MANIFEST).sort_values("fold_idx")
    if len(model_manifest) != 3:
        raise RuntimeError(f"Expected three XGBoost folds, found {len(model_manifest)}")

    pool_mask = manifest["frozen_split"].eq("train_valid_pool").to_numpy()
    pool_manifest = manifest.loc[pool_mask].reset_index(drop=True)
    pool_smiles = training.loc[pool_mask, "Smiles"].astype(str).tolist()
    pool_targets = training.loc[pool_mask, "pIC50"].to_numpy(dtype=float)
    pool_fps = circular_fingerprints(pool_smiles)

    first = model_manifest.iloc[0]
    params = {
        "n_estimators": int(first["n_estimators"]),
        "max_depth": int(first["max_depth"]),
        "learning_rate": float(first["learning_rate"]),
        "subsample": float(first["subsample"]),
        "colsample_bytree": float(first["colsample_bytree"]),
        "random_state": 42,
        "n_jobs": max(1, len(os.sched_getaffinity(0))),
    }

    models: list[XGBRegressor] = []
    means: list[float] = []
    stds: list[float] = []
    for row in model_manifest.itertuples(index=False):
        fold = int(row.fold_idx)
        train_mask = pool_manifest["final_cv3_fold"].astype(int).ne(fold).to_numpy()
        mean = float(row.transformer_y_mean)
        std = float(row.transformer_y_std)
        normalized_y = (pool_targets[train_mask] - mean) / std
        model = XGBRegressor(**params)
        log(f"Training XGBoost fold {fold} on {int(train_mask.sum()):,} rows")
        model.fit(pool_fps[train_mask], normalized_y)
        models.append(model)
        means.append(mean)
        stds.append(std)
    return models, means, stds


def predict_xgb(
    models: list[XGBRegressor],
    means: list[float],
    stds: list[float],
    smiles: list[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    features = circular_fingerprints(smiles)
    members = [
        model.predict(features).astype(float) * std + mean
        for model, mean, std in zip(models, means, stds)
    ]
    return np.mean(np.vstack(members), axis=0), members


def gat_feature_kwargs(graph: object) -> dict[str, int]:
    """Atom-feature width for GATModel, matching the installed signature."""
    n_atom = int(graph.node_features.shape[1])
    signature = inspect.signature(dc.models.GATModel.__init__)
    if "number_atom_features" in signature.parameters:
        return {"number_atom_features": n_atom}
    if "n_atom_feat" in signature.parameters:
        return {"n_atom_feat": n_atom}
    return {"number_atom_features": n_atom}


def build_gat_model(row: object, features: dict[str, int]) -> RobustGATModel:
    """Rebuild a GAT member exactly as the training notebook's builder did.

    Mirrors ``build_gat_model`` in
    GATmodelTraining_DeepAURKA_nested_split_balance_split_manifest.ipynb:
    ``graph_attention_layers = [hidden_channels] * num_layers`` (128 x 3 here),
    ``n_attention_heads = num_heads`` (8), and an Adam optimizer carrying the
    learning rate and weight decay. agg_modes, predictor_hidden_feats, residual,
    alpha and self_loop are left at the class defaults, exactly as in training.
    """
    graph_attention_layers = [int(row.hidden_channels)] * int(row.num_layers)
    optimizer = dc.models.optimizers.Adam(
        learning_rate=float(row.learning_rate),
        weight_decay=float(row.weight_decay),
    )
    kwargs = {
        "n_tasks": 1,
        "mode": "regression",
        "model_dir": str(PROJECT_ROOT / str(row.checkpoint_dir)),
        "device": "cpu",
        "graph_attention_layers": graph_attention_layers,
        "n_attention_heads": int(row.num_heads),
        "dropout": float(row.dropout),
        "batch_size": int(row.batch_size),
        "optimizer": optimizer,
        "regression_loss_name": REGRESSION_LOSS_NAME,
    }
    kwargs.update(features)
    return RobustGATModel(**kwargs)


def graph_dataset(
    smiles: list[str],
    featurizer: dc.feat.MolGraphConvFeaturizer,
) -> tuple[dc.data.NumpyDataset | None, np.ndarray, list[str]]:
    graphs = featurizer.featurize(smiles)
    valid_mask = np.asarray(
        [
            hasattr(graph, "node_features")
            and getattr(graph, "node_features", np.empty((0,))).size > 0
            and getattr(graph, "edge_features", None) is not None
            for graph in graphs
        ],
        dtype=bool,
    )
    errors = ["" if valid else "GAT graph featurization failed" for valid in valid_mask]
    if not valid_mask.any():
        return None, valid_mask, errors
    valid_graphs = np.asarray(graphs, dtype=object)[valid_mask]
    dataset = dc.data.NumpyDataset(
        X=valid_graphs,
        y=np.zeros((int(valid_mask.sum()), 1), dtype=np.float32),
        ids=np.asarray(smiles, dtype=object)[valid_mask],
    )
    return dataset, valid_mask, errors


def load_gat_ensemble(
    sample_smiles: str,
) -> tuple[
    dc.feat.MolGraphConvFeaturizer,
    list[RobustGATModel],
    pd.DataFrame,
]:
    model_manifest = pd.read_csv(GAT_MANIFEST).sort_values("fold_idx")
    if len(model_manifest) != 3:
        raise RuntimeError(f"Expected three GAT members, found {len(model_manifest)}")
    if not model_manifest["use_edges"].astype(bool).all():
        raise RuntimeError("Saved GAT members do not all use edge features")
    if not model_manifest["use_chirality"].astype(bool).all():
        raise RuntimeError("Saved GAT members do not all use chirality")
    # The saved run disabled both stacking and calibration; the ensemble is a
    # plain unweighted mean. This was verified numerically against the saved
    # external predictions (max |delta| 2.7e-15 for the plain mean, 7.4e-02 for
    # a stop_r2-weighted mean), so guard the assumption here.
    if model_manifest["calibration_enabled"].astype(str).str.lower().ne("false").any():
        raise RuntimeError("Saved GAT members expect calibration, which this script does not apply")
    if not np.allclose(model_manifest["family_stack_weight"].to_numpy(dtype=float), 1.0):
        raise RuntimeError("Saved GAT members carry non-unit family stack weights")

    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True)
    sample_graph = featurizer.featurize([sample_smiles])[0]
    features = gat_feature_kwargs(sample_graph)

    models: list[RobustGATModel] = []
    for row in model_manifest.itertuples(index=False):
        model = build_gat_model(row, features)
        checkpoint = PROJECT_ROOT / str(row.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model.restore(checkpoint=str(checkpoint))
        models.append(model)
        log(f"Restored GAT fold {int(row.fold_idx)} from {checkpoint.name}")
    return featurizer, models, model_manifest


def predict_gat(
    models: list[RobustGATModel],
    manifest: pd.DataFrame,
    featurizer: dc.feat.MolGraphConvFeaturizer,
    smiles: list[str],
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Per-fold denormalize, clip, then take the PLAIN UNWEIGHTED MEAN.

    Unlike the AFP arm in v3 (which weighted folds by stop_r2), the saved GAT
    run aggregates folds with an unweighted mean; verified against the saved
    external predictions to 2.7e-15, where a stop_r2-weighted mean is off by
    7.4e-02. ``load_gat_ensemble`` asserts the conditions that make this true.
    """
    dataset, valid_mask, errors = graph_dataset(smiles, featurizer)
    ensemble = np.full(len(smiles), np.nan, dtype=float)
    members_full = [np.full(len(smiles), np.nan, dtype=float) for _ in models]
    if dataset is None:
        return ensemble, members_full, errors

    valid_members: list[np.ndarray] = []
    for member_index, (model, row) in enumerate(
        zip(models, manifest.itertuples(index=False))
    ):
        normalized = np.asarray(model.predict(dataset), dtype=float).reshape(-1)
        raw = normalized * float(row.transformer_y_std) + float(row.transformer_y_mean)
        raw = np.clip(raw, float(row.clip_lower), float(row.clip_upper))
        members_full[member_index][valid_mask] = raw
        valid_members.append(raw)
    ensemble[valid_mask] = np.mean(np.vstack(valid_members), axis=0)
    return ensemble, members_full, errors


def validate_models(
    xgb_models: list[XGBRegressor],
    xgb_means: list[float],
    xgb_stds: list[float],
    gat_models: list[RobustGATModel],
    gat_manifest: pd.DataFrame,
    gat_featurizer: dc.feat.MolGraphConvFeaturizer,
) -> None:
    xgb_saved = pd.read_csv(XGB_EXTERNAL)
    gat_saved = pd.read_csv(GAT_EXTERNAL)
    if not xgb_saved["canonical_smiles"].equals(gat_saved["canonical_smiles"]):
        raise RuntimeError("Saved external prediction rows are not aligned")
    if not xgb_saved["Smiles"].equals(gat_saved["Smiles"]):
        raise RuntimeError("Saved external raw SMILES rows are not aligned")
    # Use the exact strings originally featurized by both validation notebooks.
    smiles = xgb_saved["Smiles"].astype(str).tolist()
    xgb_pred, xgb_members = predict_xgb(xgb_models, xgb_means, xgb_stds, smiles)
    gat_pred, gat_members, errors = predict_gat(
        gat_models, gat_manifest, gat_featurizer, smiles
    )
    if any(errors):
        raise RuntimeError("GAT failed to featurize a saved external validation row")

    xgb_delta = float(
        np.max(np.abs(xgb_pred - xgb_saved["ensemble_pred_pIC50"].to_numpy()))
    )
    gat_delta = float(
        np.max(np.abs(gat_pred - gat_saved["ensemble_pred_pIC50"].to_numpy()))
    )
    xgb_member_delta = max(
        float(
            np.max(
                np.abs(
                    prediction
                    - xgb_saved[f"pred_fold_{index}_raw"].to_numpy(dtype=float)
                )
            )
        )
        for index, prediction in enumerate(xgb_members, start=1)
    )
    gat_member_delta = max(
        float(
            np.max(
                np.abs(
                    prediction
                    - gat_saved[f"pred_fold_{index}_family_1_raw"].to_numpy(dtype=float)
                )
            )
        )
        for index, prediction in enumerate(gat_members, start=1)
    )
    log(
        "External prediction validation: "
        f"XGB ensemble max delta={xgb_delta:.3g}, member max={xgb_member_delta:.3g}; "
        f"GAT ensemble max delta={gat_delta:.3g}, member max={gat_member_delta:.3g}"
    )
    if max(xgb_delta, xgb_member_delta) > 1e-4:
        raise RuntimeError("Reconstructed XGBoost predictions do not match saved v3 output")
    if max(gat_delta, gat_member_delta) > 1e-4:
        raise RuntimeError("Restored GAT predictions do not match saved v3 output")


def concatenate_chunks(chunk_paths: list[Path], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as output_handle:
        for index, chunk_path in enumerate(chunk_paths):
            with chunk_path.open("rb") as input_handle:
                if index:
                    input_handle.readline()
                shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
    os.replace(temporary, destination)


def predict_library(
    input_path: Path,
    output_dir: Path,
    chunk_size: int,
    max_chunks: int | None,
    xgb_models: list[XGBRegressor],
    xgb_means: list[float],
    xgb_stds: list[float],
    gat_models: list[RobustGATModel],
    gat_manifest: pd.DataFrame,
    gat_featurizer: dc.feat.MolGraphConvFeaturizer,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.monotonic()
    seen_chunks = 0

    for chunk_index, source in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        seen_chunks += 1
        output_path = chunk_dir / f"prediction_chunk_{chunk_index:06d}.csv"
        state_path = chunk_dir / f"chunk_{chunk_index:06d}.json"
        if output_path.exists() and state_path.exists():
            log(f"Prediction chunk {chunk_index}: already complete")
            continue
        if "both_in_domain" not in source or not source["both_in_domain"].astype(bool).all():
            raise RuntimeError("Prediction input contains a compound outside the dual domain")

        chunk_started = time.monotonic()
        smiles = source["canonical_smiles"].astype(str).tolist()
        xgb_pred, xgb_members = predict_xgb(
            xgb_models, xgb_means, xgb_stds, smiles
        )
        gat_pred, gat_members, gat_errors = predict_gat(
            gat_models, gat_manifest, gat_featurizer, smiles
        )
        consensus = (
            XGB_CONSENSUS_WEIGHT * xgb_pred + GAT_CONSENSUS_WEIGHT * gat_pred
        )
        # Graph featurization failures keep their XGB prediction on a SEPARATE
        # column rather than being dropped (v3 lost 135 rows) or silently mixed
        # into the consensus column, which would put two scales in one field.
        graph_failed = ~np.isfinite(gat_pred)
        half_width = CONFORMAL_HALF_WIDTH[CONFORMAL_LEVEL]

        output = source.copy()
        for index, member_pred in enumerate(xgb_members, start=1):
            output[f"xgb_fold_{index}_pred_pIC50"] = member_pred
        output["xgb_ensemble_pred_pIC50"] = xgb_pred
        for index, member_pred in enumerate(gat_members, start=1):
            output[f"gat_fold_{index}_pred_pIC50"] = member_pred
        output["gat_ensemble_pred_pIC50"] = gat_pred
        output["consensus_xgb_weight"] = XGB_CONSENSUS_WEIGHT
        output["consensus_gat_weight"] = GAT_CONSENSUS_WEIGHT
        output["consensus_pred_pIC50"] = consensus
        output["consensus_pred_IC50_nM"] = np.power(10.0, 9.0 - consensus)
        output["absolute_model_disagreement_pIC50"] = np.abs(xgb_pred - gat_pred)
        # Spread across all six members: an uncertainty proxy independent of the
        # conformal band, which is a single global width.
        all_members = np.vstack(xgb_members + gat_members)
        output["member_spread_pIC50"] = np.nanstd(all_members, axis=0)
        output["conformal_level"] = CONFORMAL_LEVEL
        output["conformal_half_width_pIC50"] = half_width
        output["consensus_pred_pIC50_lower"] = consensus - half_width
        output["consensus_pred_pIC50_upper"] = consensus + half_width
        # Reported so no downstream step has to re-derive why an absolute
        # sub-100 nM claim is unavailable at this error level.
        output["confident_sub_100nM"] = (consensus - half_width) >= 7.0
        output["xgb_only_pred_pIC50"] = np.where(graph_failed, xgb_pred, np.nan)
        output["prediction_status"] = np.where(
            np.isfinite(consensus),
            "predicted",
            "graph_featurization_failed_xgb_only",
        )
        output["prediction_note"] = gat_errors

        output_tmp = output_path.with_suffix(".csv.partial")
        state_tmp = state_path.with_suffix(".json.partial")
        output.to_csv(output_tmp, index=False)
        state = {
            "chunk_index": chunk_index,
            "input_rows": int(len(output)),
            "predicted_rows": int(np.isfinite(consensus).sum()),
            "graph_featurization_failures": int(graph_failed.sum()),
            "elapsed_seconds": time.monotonic() - chunk_started,
            "completed_at_utc": utc_now(),
        }
        state_tmp.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(output_tmp, output_path)
        os.replace(state_tmp, state_path)
        log(
            f"Prediction chunk {chunk_index}: {len(output):,} rows, "
            f"{state['predicted_rows']:,} predicted, "
            f"{state['elapsed_seconds']:.1f} s"
        )

    state_paths = sorted(chunk_dir.glob("chunk_*.json"))
    if len(state_paths) != seen_chunks:
        raise RuntimeError(
            f"Expected {seen_chunks} prediction chunk states, found {len(state_paths)}"
        )
    states = [json.loads(path.read_text()) for path in state_paths]
    chunks = sorted(chunk_dir.glob("prediction_chunk_*.csv"))
    unsorted_output = output_dir / "coconut_dual_domain_consensus_predictions.csv"
    concatenate_chunks(chunks, unsorted_output)

    log("Sorting successful predictions by consensus pIC50")
    ranked = pd.read_csv(unsorted_output)
    ranked = ranked.sort_values(
        ["consensus_pred_pIC50", "absolute_model_disagreement_pIC50"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "consensus_rank", np.arange(1, len(ranked) + 1))
    ranked_output = output_dir / "coconut_dual_domain_consensus_ranked.csv"
    ranked.to_csv(ranked_output, index=False)

    total = sum(item["input_rows"] for item in states)
    predicted = sum(item["predicted_rows"] for item in states)
    summary = {
        "stage": "dual_in_domain_xgb_gat_consensus_prediction_v5",
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "input_file": str(input_path.resolve()),
        "input_rows": total,
        "predicted_rows": predicted,
        "graph_featurization_failures": sum(
            item["graph_featurization_failures"] for item in states
        ),
        "members": {"descriptor_model": "MLXG (XGBoost, Morgan r3/1024)", "graph_model": "GAT"},
        "weighting_rule": "equal weights (variance-minimising split for this pair is 0.477/0.523)",
        "weighting_requires_no_fitting": True,
        "xgb_consensus_weight": XGB_CONSENSUS_WEIGHT,
        "gat_consensus_weight": GAT_CONSENSUS_WEIGHT,
        "gat_fold_aggregation": "plain unweighted mean (verified vs saved external output to 2.7e-15)",
        "frozen_test_reference": {
            "n": FROZEN_TEST_N,
            "r2": FROZEN_TEST_R2,
            "rmse": FROZEN_TEST_RMSE,
            "spearman": FROZEN_TEST_SPEARMAN,
            "low_similarity_r2": LOW_SIM_R2,
            "low_similarity_n": 85,
        },
        "conformal": {
            "level": CONFORMAL_LEVEL,
            "half_width_pIC50": CONFORMAL_HALF_WIDTH[CONFORMAL_LEVEL],
            "half_widths_all_levels": CONFORMAL_HALF_WIDTH,
            "low_similarity_half_width_90": LOW_SIM_CONFORMAL_HALF_WIDTH_90,
            "calibration_set": "425-row v3 frozen test, equal-weight XGB+GAT",
            "note": (
                "A lower bound >= 7.0 requires a predicted pIC50 >= 8.579. The v3 "
                "library maximum was 7.743, so absolute sub-100 nM claims are not "
                "available at this error level; select by rank to a capacity-driven depth."
            ),
        },
        "superseded_v3_consensus": {
            "members": "XGB + AFP, R2-proportional 0.6443/0.3557",
            "frozen_test_r2": 0.5620,
            "reason_replaced": (
                "below XGB alone (0.5947); AFP is last of five models trained "
                "(0.3283) and -0.020 on low-similarity compounds"
            ),
        },
        "unranked_output": str(unsorted_output.resolve()),
        "ranked_output": str(ranked_output.resolve()),
        "elapsed_seconds": time.monotonic() - started,
        "cpu_affinity_visible_to_process": sorted(os.sched_getaffinity(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output_dir / "consensus_prediction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    log(f"Consensus prediction complete: {predicted:,}/{total:,} rows")
    log(f"Ranked output: {ranked_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything()
    log(
        f"Starting v3 predictor; CPU affinity={sorted(os.sched_getaffinity(0))}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
    )
    training, split_manifest = load_and_validate_training_inputs()
    xgb_models, xgb_means, xgb_stds = fit_xgb_ensemble(training, split_manifest)
    sample_smiles = training.iloc[0]["Smiles"]
    gat_featurizer, gat_models, gat_manifest = load_gat_ensemble(sample_smiles)
    validate_models(
        xgb_models,
        xgb_means,
        xgb_stds,
        gat_models,
        gat_manifest,
        gat_featurizer,
    )
    if args.validate_only:
        log("Validation-only predictor run complete")
        return 0
    predict_library(
        args.input,
        args.output_dir,
        args.chunk_size,
        args.max_chunks,
        xgb_models,
        xgb_means,
        xgb_stds,
        gat_models,
        gat_manifest,
        gat_featurizer,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
