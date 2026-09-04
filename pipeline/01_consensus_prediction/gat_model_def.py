#!/usr/bin/env python
"""GAT model wrapper copied VERBATIM from the training notebook.

Source: GATmodelTraining_DeepAURKA_nested_split_balance_split_manifest.ipynb
        (DeepAURKA project tree), the notebook that produced the
        three checkpoints in output_DeepAURKA_GAT_balance_manifest/gat_best_model_balance/.

The class below is an unmodified copy of the notebook's definition. It is copied
rather than reconstructed so the module graph -- and therefore the state_dict
keys -- are byte-identical to what wrote the checkpoints. Do not "simplify" it:
dc.models.GATModel is NOT a drop-in substitute, because the checkpoints were
written by this subclass (huber regression loss, self-loop handling in
_prepare_batch).

The loss choice is irrelevant at inference time; it is retained so construction
matches training exactly.
"""

from __future__ import annotations

import dgl
import dgl.function as fn
import torch
from deepchem.models.losses import HuberLoss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.gat import GAT
from deepchem.models.torch_models.torch_model import TorchModel


def patch_dgl_compat() -> None:
    """DGL renamed these between 0.x and 1.x; the notebook applied the same shim."""
    if not hasattr(fn, "copy_edge") and hasattr(fn, "copy_e"):
        fn.copy_edge = fn.copy_e
    if not hasattr(fn, "src_mul_edge") and hasattr(fn, "u_mul_e"):
        fn.src_mul_edge = fn.u_mul_e


patch_dgl_compat()


class RobustGATModel(TorchModel):
    """DeepChem-compatible GAT wrapper with selectable regression loss."""

    def __init__(
        self,
        n_tasks,
        graph_attention_layers=None,
        n_attention_heads=8,
        agg_modes=None,
        activation=torch.nn.functional.elu,
        residual=True,
        dropout=0.0,
        alpha=0.2,
        predictor_hidden_feats=128,
        predictor_dropout=0.0,
        mode='regression',
        number_atom_features=30,
        n_classes=2,
        self_loop=True,
        regression_loss_name='huber',
        **kwargs,
    ):
        model = GAT(
            n_tasks=n_tasks,
            graph_attention_layers=graph_attention_layers,
            n_attention_heads=n_attention_heads,
            agg_modes=agg_modes,
            activation=activation,
            residual=residual,
            dropout=dropout,
            alpha=alpha,
            predictor_hidden_feats=predictor_hidden_feats,
            predictor_dropout=predictor_dropout,
            mode=mode,
            number_atom_features=number_atom_features,
            n_classes=n_classes,
        )
        if mode == 'regression':
            normalized_loss_name = str(regression_loss_name).strip().lower()
            if normalized_loss_name == 'huber':
                loss = HuberLoss()
            elif normalized_loss_name in {'l2', 'mse'}:
                loss = L2Loss()
            else:
                raise ValueError(f'Unsupported regression_loss_name: {regression_loss_name}')
            output_types = ['prediction']
        else:
            loss = SparseSoftmaxCrossEntropy()
            output_types = ['prediction', 'loss']
        super().__init__(model, loss=loss, output_types=output_types, **kwargs)
        self._self_loop = self_loop

    def _prepare_batch(self, batch):
        inputs, labels, weights = batch
        dgl_graphs = [graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]]
        inputs = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super()._prepare_batch(([], labels, weights))
        return inputs, labels, weights
