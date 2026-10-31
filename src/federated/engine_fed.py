"""
Phase 3 glue code between Flower and the plain-PyTorch pieces built in
Phase 0/1/2 (src/models/mlp.py, src/train/engine.py). engine.py itself is
NOT edited -- this file wraps it instead, per the plan's "vanilla first,
framework second" rule: the model and the metric code stay identical to
what Phase 1's centralized run and Phase 2's local-only run already used
and reported. Only the training LOOP shape changes here, because a
federated round is fundamentally different from a one-shot fit() call:

  - fit() (engine.py) trains to convergence with early stopping -- correct
    for "train this model once, as well as possible" (Phase 1, Phase 2).
  - A federated round instead does a FEW local epochs, then stops and
    sends the updated weights back -- early stopping on a round makes no
    sense (the model isn't meant to converge locally; it's meant to take
    one bounded step before the next aggregation). So local_train() below
    is a short, fixed-epoch loop, not a copy of fit().

FedProx's only difference from FedAvg happens here, client-side: it adds
the proximal term (mu/2)*||w - w_global||^2 to the loss, pulling local
updates back toward the global model so one client's local epochs can't
drag the shared model too far on non-IID data (Li et al. 2018,
arxiv.org/abs/1812.06127). The server-side aggregation (weighted average
of client updates by sample count) is identical for both -- Flower's own
FedProx strategy class is FedAvg's aggregation plus sending `proximal_mu`
in the fit config, which is exactly what this file reads.
"""
import sys
sys.path.insert(0, "src")
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flwr.client import NumPyClient
from flwr.common import Context

from models.mlp import MLP
from train.engine import class_weights, evaluate, make_loaders


def get_parameters(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(np.asarray(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def local_train(model, X, y, n_classes, epochs, batch_size, lr, proximal_mu, global_params):
    """Fixed-epoch local update for one federated round. class_weights()
    and the Adam+CE recipe are reused unchanged from engine.py -- only the
    lack of early stopping and the optional FedProx term are new."""
    device = "cpu"
    model.to(device)
    weights = class_weights(y, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    global_tensors = None
    if proximal_mu > 0:
        global_tensors = [torch.as_tensor(p, dtype=torch.float32) for p in global_params]

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if global_tensors is not None:
                prox = torch.zeros(1)
                for local_p, global_p in zip(model.parameters(), global_tensors):
                    prox = prox + (local_p - global_p).pow(2).sum()
                loss = loss + (proximal_mu / 2.0) * prox.squeeze()
            loss.backward()
            optimizer.step()
    return model


class FlowerClient(NumPyClient):
    """One client = one device (N-BaIoT) or one Dirichlet shard (TON_IoT).
    Holds its own local training data only -- never sees another
    client's rows, consistent with every other phase's privacy framing."""

    def __init__(self, model, X_train, y_train, n_classes):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.n_classes = n_classes

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        # Snapshot the GLOBAL weights as received, before any local step --
        # this is w^t in the proximal term, fixed for the whole round.
        global_params = [p.copy() for p in parameters]
        mu = float(config.get("proximal_mu", 0.0))
        epochs = int(config.get("local_epochs", 2))
        batch_size = int(config.get("batch_size", 256))
        lr = float(config.get("lr", 1e-3))
        local_train(self.model, self.X_train, self.y_train, self.n_classes,
                    epochs=epochs, batch_size=batch_size, lr=lr,
                    proximal_mu=mu, global_params=global_params)
        return get_parameters(self.model), len(self.y_train), {}

    def evaluate(self, parameters, config):
        # Federated (per-client) evaluation is disabled (fraction_evaluate=0
        # in the strategy) -- the metric we report is the server's
        # CENTRALIZED evaluation against the shared global test set, which
        # is directly comparable to Phase 1 and Phase 2's "global test"
        # numbers. This method exists only to satisfy the NumPyClient
        # interface and is not expected to be called.
        set_parameters(self.model, parameters)
        return 0.0, len(self.y_train), {}


def make_client_fn(dataset_tag, n_features, n_classes, data_dir="data/partitions/federated",
                    base_seed=42):
    def client_fn(context: Context):
        partition_id = context.node_config["partition-id"]
        X = np.load(f"{data_dir}/{dataset_tag}_client{partition_id}_X.npy")
        y = np.load(f"{data_dir}/{dataset_tag}_client{partition_id}_y.npy")
        torch.manual_seed(base_seed + partition_id)
        model = MLP(n_features=n_features, n_classes=n_classes)
        return FlowerClient(model, X, y, n_classes).to_client()
    return client_fn


def make_evaluate_fn(X_test, y_test, n_features, n_classes, history_sink, batch_size=1024,
                      checkpoint_path=None):
    """Centralized evaluation, run on the SERVER after each round's
    aggregation -- reuses engine.evaluate() unchanged, exactly as Phase 1
    and Phase 2 did, so macro-F1 / per-class recall / FPR numbers are
    computed the same way across every phase.

    If checkpoint_path is given, overwrites it with the current global
    model's state_dict every round, so the file left behind at the end is
    the FINAL round's global model -- needed for Phase 4, which has to
    load the actual federated model (not just its metrics) to test it
    against unseen-attack clients."""
    def evaluate_fn(server_round, parameters, config):
        model = MLP(n_features=n_features, n_classes=n_classes)
        set_parameters(model, parameters)
        _, loader = make_loaders(X_test[:1], y_test[:1], X_test, y_test, batch_size=batch_size)
        metrics = evaluate(model, loader, n_classes)
        history_sink.append({"round": server_round, **metrics})
        if checkpoint_path is not None:
            torch.save(model.state_dict(), checkpoint_path)
        return float(1.0 - metrics["macro_f1"]), {"macro_f1": metrics["macro_f1"]}
    return evaluate_fn
