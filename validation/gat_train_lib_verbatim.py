# ==== imports (verbatim) ====

from datetime import datetime

from deepchem.models.losses import HuberLoss, L2Loss, SparseSoftmaxCrossEntropy

from deepchem.models.torch_models.gat import GAT

from deepchem.models.torch_models.torch_model import TorchModel


from pathlib import Path

from rdkit import Chem, DataStructs

from rdkit.Chem import AllChem

from sklearn.metrics import r2_score


import deepchem as dc

import dgl

import dgl.function as fn


import glob

import inspect

import json


import numpy as np


import os

import pandas as pd

import random

import re


import shutil


import time

import torch

import warnings

# ==== constants (verbatim) ====

MIN_DELTA = 2e-3

FINAL_MAX_EPOCHS = 220

FINAL_PATIENCE = 24

FINAL_CV_FOLDS = 3

PREDICTION_CLIP_STD_MULT = 0.75

PREDICTION_CLIP_MIN_MARGIN = 0.50

# ==== functions (verbatim) ====

def _ensure_edge_features(dataset_obj):
    if dataset_obj is None:
        return
    for graph in dataset_obj.X:
        if getattr(graph, 'edge_features', None) is None:
            edge_index = getattr(graph, 'edge_index', None)
            n_edges = edge_index.shape[1] if edge_index is not None else 0
            graph.edge_features = np.zeros((n_edges, 1), dtype=np.float32)

def _build_csv_loader(featurizer):
    csv_loader_sig = inspect.signature(dc.data.CSVLoader.__init__)
    if 'feature_field' in csv_loader_sig.parameters:
        return dc.data.CSVLoader(tasks=[task_name], feature_field=smiles_column, featurizer=featurizer)
    return dc.data.CSVLoader(tasks=[task_name], smiles_field=smiles_column, featurizer=featurizer)

def _create_raw_dataset(loader, dataset_path):
    if hasattr(loader, 'create_dataset'):
        return loader.create_dataset(dataset_path)
    return loader.featurize(dataset_path)

def _gat_feature_kwargs_from_dataset(dataset_obj):
    sample_graph = dataset_obj.X[0]
    n_atom_feat = sample_graph.node_features.shape[1]
    n_bond_feat = sample_graph.edge_features.shape[1] if getattr(sample_graph, 'edge_features', None) is not None else 0
    gat_sig = inspect.signature(dc.models.GATModel.__init__)
    if 'number_atom_features' in gat_sig.parameters:
        feature_kwargs = {'number_atom_features': n_atom_feat}
    elif 'n_atom_feat' in gat_sig.parameters:
        feature_kwargs = {'n_atom_feat': n_atom_feat}
    else:
        feature_kwargs = {'number_atom_features': n_atom_feat}
    return feature_kwargs, n_atom_feat, n_bond_feat

def manifest_global_indices(column_name, label):
    manifest = ensure_split_manifest_ready()
    if column_name not in manifest.columns:
        raise KeyError(f"Split manifest column not found: {column_name}")
    mask = manifest[column_name].astype(str) == str(label)
    return manifest.loc[mask, "row_id"].astype(int).to_numpy()

def global_to_local_indices(global_indices, base_global_indices, context):
    global_indices = np.asarray(list(global_indices), dtype=int)
    base_global_indices = np.asarray(list(base_global_indices), dtype=int)
    lookup = {int(global_idx): pos for pos, global_idx in enumerate(base_global_indices)}
    missing = [int(global_idx) for global_idx in global_indices if int(global_idx) not in lookup]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"Split manifest indices for {context} are not contained in the requested base split. "
            f"Missing count={len(missing)} preview={preview}"
        )
    return np.asarray([lookup[int(global_idx)] for global_idx in global_indices], dtype=int)

def manifest_frozen_split_indices(raw_dataset):
    ensure_split_manifest_ready(raw_dataset)
    frozen_pool_idx = manifest_split_indices("frozen_split", "train_valid_pool")
    frozen_test_idx = manifest_split_indices("frozen_split", "frozen_test")
    return frozen_pool_idx, [], frozen_test_idx

def manifest_final_fold_indices(frozen_pool_global_indices):
    ensure_split_manifest_ready()
    fold_indices = []
    for fold_idx in range(1, FINAL_CV_FOLDS + 1):
        fold_global_idx = split_manifest.loc[
            split_manifest["final_cv3_fold"].astype(int) == int(fold_idx),
            "row_id",
        ].astype(int).to_numpy()
        fold_indices.append(
            global_to_local_indices(
                fold_global_idx,
                frozen_pool_global_indices,
                f"final_cv3_fold={fold_idx}",
            ).tolist()
        )
    return fold_indices

def select_dataset(raw_dataset, indices):
    indices = list(indices)
    if len(indices) == 0:
        return None
    selected = raw_dataset.select(indices)
    _ensure_edge_features(selected)
    return selected

def transform_from_train(train_raw, valid_raw=None, test_raw=None):
    _ensure_edge_features(train_raw)
    _ensure_edge_features(valid_raw)
    _ensure_edge_features(test_raw)
    local_transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train_raw)
    train_dataset = local_transformer.transform(train_raw)
    valid_dataset = local_transformer.transform(valid_raw) if valid_raw is not None else None
    test_dataset = local_transformer.transform(test_raw) if test_raw is not None else None
    sample_weight_summary = None
    if ENABLE_POTENCY_SAMPLE_WEIGHTS:
        sample_weight_summary = compute_potency_sample_weights(train_raw)
        train_dataset = _clone_dataset_with_weights(train_dataset, sample_weight_summary['weights'])
    else:
        train_dataset = _clone_dataset_with_weights(train_dataset)
    valid_dataset = _clone_dataset_with_weights(valid_dataset) if valid_dataset is not None else None
    test_dataset = _clone_dataset_with_weights(test_dataset) if test_dataset is not None else None
    return {'train_dataset': train_dataset, 'valid_dataset': valid_dataset, 'test_dataset': test_dataset, 'transformer': local_transformer, 'sample_weight_summary': sample_weight_summary}

def compute_dataset_r2(model_obj, dataset_obj, transformer_obj):
    y_true = transformer_obj.untransform(dataset_obj.y).flatten()
    y_pred = transformer_obj.untransform(model_obj.predict(dataset_obj)).flatten()
    if np.std(y_true) == 0:
        return np.nan
    return r2_score(y_true, y_pred)

def untransform_targets(dataset_obj, transformer_obj):
    return transformer_obj.untransform(dataset_obj.y).flatten()

def regression_summary(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mse = float(rmse ** 2)
    if np.std(y_true) == 0:
        r2 = np.nan
    else:
        r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson_r = np.nan
    else:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "pearson_r": pearson_r,
    }

def params_from_mapping(row):
    return {
        "num_layers": int(row["num_layers"]),
        "num_heads": int(row["num_heads"]),
        "hidden_channels": int(row["hidden_channels"]),
        "batch_size": int(row["batch_size"]),
        "dropout": float(row["dropout"]),
        "learning_rate": float(row["learning_rate"]),
        "weight_decay": float(row["weight_decay"]),
    }

def build_gat_model(params, gat_feature_kwargs, model_dir):
    graph_attention_layers = [int(params["hidden_channels"])] * int(params["num_layers"])
    optimizer = dc.models.optimizers.Adam(
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    model_kwargs = {
        "n_tasks": 1,
        "mode": "regression",
        "model_dir": model_dir,
        "device": device,
        "graph_attention_layers": graph_attention_layers,
        "n_attention_heads": int(params["num_heads"]),
        "dropout": float(params["dropout"]),
        "batch_size": int(params["batch_size"]),
        "optimizer": optimizer,
        "regression_loss_name": REGRESSION_LOSS_NAME,
    }
    model_kwargs.update(gat_feature_kwargs)
    return RobustGATModel(**model_kwargs)

def is_checkpoint_write_error(exc):
    message = str(exc)
    return (
        'PytorchStreamWriter' in message
        or 'unexpected pos' in message
        or 'file write failed' in message
    )

def safe_save_checkpoint(model_obj, model_dir, max_checkpoints_to_keep=1, retries=1):
    last_error = None
    for attempt in range(retries + 1):
        try:
            model_obj.save_checkpoint(
                max_checkpoints_to_keep=max_checkpoints_to_keep,
                model_dir=model_dir,
            )
            return
        except RuntimeError as exc:
            last_error = exc
            if not is_checkpoint_write_error(exc) or attempt >= retries:
                raise
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            os.makedirs(model_dir, exist_ok=True)
    if last_error is not None:
        raise last_error

def train_with_early_stopping(
    params,
    gat_feature_kwargs,
    train_dataset,
    stop_dataset,
    transformer_obj,
    model_dir,
    max_epochs,
    patience,
    min_delta,
):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    model_obj = build_gat_model(
        params=params,
        gat_feature_kwargs=gat_feature_kwargs,
        model_dir=model_dir,
    )

    best_stop_r2 = float("-inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model_obj.fit(train_dataset, nb_epoch=1, checkpoint_interval=0, max_checkpoints_to_keep=0)
        stop_r2 = compute_dataset_r2(model_obj, stop_dataset, transformer_obj)
        stop_r2_for_stop = stop_r2 if np.isfinite(stop_r2) else -1e9

        if stop_r2_for_stop > best_stop_r2 + min_delta:
            best_stop_r2 = stop_r2_for_stop
            best_epoch = epoch
            no_improve = 0
            safe_save_checkpoint(model_obj, model_dir=model_dir, max_checkpoints_to_keep=1, retries=1)
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    checkpoints = model_obj.get_checkpoints(model_dir=model_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint was saved for {model_dir}")

    model_obj.restore(checkpoint=checkpoints[-1])
    train_r2 = compute_dataset_r2(model_obj, train_dataset, transformer_obj)
    stop_r2 = compute_dataset_r2(model_obj, stop_dataset, transformer_obj)

    return {
        "model": model_obj,
        "best_epoch": int(best_epoch),
        "train_r2": float(train_r2),
        "stop_r2": float(stop_r2),
    }

def prediction_clip_bounds(train_dataset, transformer_obj):
    train_targets = untransform_targets(train_dataset, transformer_obj)
    train_std = float(np.std(train_targets))
    margin = max(PREDICTION_CLIP_MIN_MARGIN, PREDICTION_CLIP_STD_MULT * train_std)
    lower = float(np.min(train_targets) - margin)
    upper = float(np.max(train_targets) + margin)
    return lower, upper

def clip_predictions(predictions, clip_bounds=None):
    pred = np.asarray(predictions, dtype=float)
    if clip_bounds is None:
        return pred
    lower, upper = clip_bounds
    return np.clip(pred, lower, upper)

def aggregate_prediction_arrays(prediction_arrays, member_weights=None):
    stacked = np.vstack(prediction_arrays).astype(float)

    if member_weights is not None:
        weights = np.asarray(member_weights, dtype=float).flatten()
        if weights.shape[0] == stacked.shape[0]:
            weights = np.clip(weights, 0.0, None)
            if np.sum(weights) > 0:
                return np.average(stacked, axis=0, weights=weights)

    if stacked.shape[0] == 1:
        return stacked[0].copy()
    return np.mean(stacked, axis=0)