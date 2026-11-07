# Author: Kai Fan kf5601
# Author: Tyriz Newton tn1207
# Class: CSCI 532 Introduction to Intelligent Security Systems
# Professor: Dr.Leon Reznik
# File: src/pid_strategy.py
#
# PID-based anomaly detection strategy for Federated Learning.
#
# Implements the adapted PID controller described in the project spec:
#
#   u(t) = Kp * D(t, mu) + Ki * sum_{i=1}^{t-1} D(i, mu) + Kd * (D(t, mu) - D(t-1, mu))
#
# Clients whose PID score exceeds the threshold are excluded from aggregation,
# global model distribution, and further communication.

import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple


def _flatten_params(parameters: Parameters) -> np.ndarray:
    """Concatenate all layer arrays in a Parameters object into one flat vector."""
    ndarrays = parameters_to_ndarrays(parameters)
    return np.concatenate([arr.ravel().astype(np.float64) for arr in ndarrays])


def _normalized_l2(vec: np.ndarray, centroid: np.ndarray) -> float:
    """
    Normalized L2 distance between a client vector and the centroid.

    Normalizing by ||centroid|| keeps distances on a comparable scale across
    rounds (model weights can grow/shrink substantially as training proceeds).
    """
    denom = np.linalg.norm(centroid)
    if denom < 1e-12:
        return 0.0
    return float(np.linalg.norm(vec - centroid) / denom)


class PIDStrategy(FedAvg):
    """
    FedAvg with PID-based client anomaly filtering.

    Parameters
    ----------
    Kp, Ki, Kd : PID coefficients
    pid_threshold : clients with PID score > threshold are excluded
    patience : number of consecutive rounds a client must exceed threshold
               before being permanently excluded (reduces false positives)
    min_clients_for_aggregation : safety floor; never aggregate fewer than
               this many clients (prevents total collapse if threshold is
               mis-tuned)
    """

    def __init__(
        self,
        Kp: float = 1.0,
        Ki: float = 0.05,
        Kd: float = 0.5,
        pid_threshold: float = 0.5,
        patience: int = 1,
        min_clients_for_aggregation: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.pid_threshold = pid_threshold
        self.patience = patience
        self.min_clients_for_aggregation = min_clients_for_aggregation

        # Per-client cumulative integral term: sum of past distances
        self._integral: Dict[str, float] = {}
        # Per-client previous distance (for derivative term)
        self._prev_dist: Dict[str, float] = {}
        # Per-client consecutive violation counter
        self._violation_count: Dict[str, int] = {}
        # Clients permanently excluded
        self.excluded_clients: set = set()
        # Logs: round -> {cid: pid_score}
        self.pid_log: Dict[int, Dict[str, float]] = {}
        # Logs: round -> list of excluded cids *at that round*
        self.removal_log: Dict[int, List[str]] = {}

    # ------------------------------------------------------------------
    # PID computation
    # ------------------------------------------------------------------
    def _compute_pid_scores(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> Dict[str, float]:
        """Compute PID score for each client in this round's results."""
        if not results:
            return {}

        # Build cid -> flattened weights mapping.
        # NOTE: Flower's ClientProxy.cid is a stable identifier; we also
        # read the integer cid from FitRes.metrics['cid'] because our client
        # returns it explicitly (Ray-safe).
        cid_to_vec: Dict[str, np.ndarray] = {}
        for proxy, fit_res in results:
            cid_val = fit_res.metrics.get("cid", proxy.cid)
            cid = str(cid_val)
            cid_to_vec[cid] = _flatten_params(fit_res.parameters)

        # Centroid = elementwise mean of all client vectors
        stack = np.stack(list(cid_to_vec.values()), axis=0)
        centroid = stack.mean(axis=0)

        pid_scores: Dict[str, float] = {}
        for cid, vec in cid_to_vec.items():
            dist = _normalized_l2(vec, centroid)

            # Round 1 (first observation for this client): only proportional term
            if cid not in self._integral:
                self._integral[cid] = 0.0
                self._prev_dist[cid] = dist
                # First-round PID is just the proportional distance
                pid_scores[cid] = self.Kp * dist
            else:
                self._integral[cid] += dist
                derivative = dist - self._prev_dist[cid]
                self._prev_dist[cid] = dist
                pid_scores[cid] = (
                    self.Kp * dist
                    + self.Ki * self._integral[cid]
                    + self.Kd * derivative
                )

        return pid_scores

    # ------------------------------------------------------------------
    # Aggregate fit: filter clients then call FedAvg's weighting
    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        if not results:
            return None, {}

        # 1. Compute PID scores for this round
        pid_scores = self._compute_pid_scores(results)
        self.pid_log[server_round] = dict(pid_scores)

        # 2. Determine which clients to exclude this round
        newly_excluded: List[str] = []
        kept_results: List[Tuple[ClientProxy, FitRes]] = []

        for proxy, fit_res in results:
            cid_val = fit_res.metrics.get("cid", proxy.cid)
            cid = str(cid_val)

            # Already permanently excluded -> skip silently
            if cid in self.excluded_clients:
                continue

            score = pid_scores.get(cid, 0.0)
            if score > self.pid_threshold:
                self._violation_count[cid] = self._violation_count.get(cid, 0) + 1
                if self._violation_count[cid] >= self.patience:
                    self.excluded_clients.add(cid)
                    newly_excluded.append(cid)
                    continue  # do not include in aggregation
            else:
                self._violation_count[cid] = 0

            kept_results.append((proxy, fit_res))

        # 3. Safety floor: if too few clients remain, fall back to full set
        if len(kept_results) < self.min_clients_for_aggregation:
            print(
                f"[PID] Round {server_round}: too few clients "
                f"({len(kept_results)}) after filtering; reverting to all clients."
            )
            kept_results = list(results)

        self.removal_log[server_round] = newly_excluded

        print(
            f"[PID] Round {server_round}: "
            f"{len(kept_results)}/{len(results)} clients kept; "
            f"newly excluded={newly_excluded}"
        )

        # 4. Delegate to standard FedAvg weighting/aggregation
        return super().aggregate_fit(server_round, kept_results, failures)

    # ------------------------------------------------------------------
    # Aggregate evaluate: honor exclusion so excluded clients are not
    # evaluated (spec: discard from further communication)
    # ------------------------------------------------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        if not results:
            return None, {}

        kept = []
        for proxy, eval_res in results:
            cid_val = eval_res.metrics.get("cid", proxy.cid)
            cid = str(cid_val)
            if cid in self.excluded_clients:
                continue
            kept.append((proxy, eval_res))

        if not kept:
            kept = list(results)

        return super().aggregate_evaluate(server_round, kept, failures)