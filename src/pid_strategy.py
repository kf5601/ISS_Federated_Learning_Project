# Author: Kai Fan kf5601
# Author: Tyriz Newton tn1207
# Class: CSCI 532 Introduction to Intelligent Security Systems
# Professor: Dr.Leon Reznik
# File: src/pid_strategy.py

import numpy as np
import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple


def _flatten_params(parameters: Parameters) -> np.ndarray:
    """Concatenate layer arrays in a Parameters object into one flat vector."""
    ndarrays = parameters_to_ndarrays(parameters)
    return np.concatenate([arr.ravel().astype(np.float64) for arr in ndarrays])


class PIDStrategy(FedAvg):
    """
    FedAvg with Adaptive Core-Density PID Client Exclusion.
    
    Uses pairwise distance density to construct a robust consensus centroid
    and dynamically threshold outliers via Z-score scaling.
    """

    def __init__(
        self,
        Kp: float = 1.0,
        Ki: float = 0.05,
        Kd: float = 0.5,
        z_multiplier: float = 1.25,
        patience: int = 2,
        min_clients_for_aggregation: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.z_multiplier = z_multiplier
        self.patience = patience
        self.min_clients_for_aggregation = min_clients_for_aggregation

        self._integral: Dict[str, float] = {}
        self._prev_dist: Dict[str, float] = {}
        self._violation_count: Dict[str, int] = {}
        self.excluded_clients: set = set()
        self.pid_log: Dict[int, Dict[str, float]] = {}
        self.removal_log: Dict[int, List[str]] = {}

    def _compute_core_centroid(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Finds the densest 50% cluster of client models to form a robust centroid."""
        n = len(vectors)
        if n <= 2:
            return np.mean(vectors, axis=0)

        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(vectors[i] - vectors[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        k = max(1, n // 2)
        knn_scores = np.array([np.sum(np.sort(dist_matrix[i])[1 : k + 1]) for i in range(n)])
        
        # Select the densest 50% of active models
        core_indices = np.argsort(knn_scores)[: max(1, n // 2)]
        return np.mean([vectors[idx] for idx in core_indices], axis=0)

    def _compute_pid_scores(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> Tuple[Dict[str, float], float]:
        if not results:
            return {}, 0.0

        cid_to_vec: Dict[str, np.ndarray] = {}
        for proxy, fit_res in results:
            cid_val = fit_res.metrics.get("cid", proxy.cid)
            cid = str(cid_val)
            cid_to_vec[cid] = _flatten_params(fit_res.parameters)

        cids = list(cid_to_vec.keys())
        vectors = [cid_to_vec[c] for c in cids]

        # 1. Robust Centroid
        centroid = self._compute_core_centroid(vectors)

        # 2. Distance Normalization
        raw_dists = {cid: float(np.linalg.norm(cid_to_vec[cid] - centroid)) for cid in cids}
        median_dist = np.median(list(raw_dists.values())) + 1e-12
        norm_dists = {cid: raw_dists[cid] / median_dist for cid in cids}

        # 3. PID Score Evaluation
        pid_scores: Dict[str, float] = {}
        for cid in cids:
            dist = norm_dists[cid]

            if cid not in self._integral:
                self._integral[cid] = 0.0
                self._prev_dist[cid] = dist
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

        # 4. Dynamic Z-Score Threshold
        scores_arr = np.array(list(pid_scores.values()))
        dynamic_threshold = float(np.mean(scores_arr) + self.z_multiplier * np.std(scores_arr))

        return pid_scores, dynamic_threshold

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        if not results:
            return None, {}

        pid_scores, dynamic_threshold = self._compute_pid_scores(results)
        self.pid_log[server_round] = dict(pid_scores)

        newly_excluded: List[str] = []
        kept_results: List[Tuple[ClientProxy, FitRes]] = []

        for proxy, fit_res in results:
            cid_val = fit_res.metrics.get("cid", proxy.cid)
            cid = str(cid_val)

            if cid in self.excluded_clients:
                continue

            score = pid_scores.get(cid, 0.0)
            if score > dynamic_threshold:
                self._violation_count[cid] = self._violation_count.get(cid, 0) + 1
                if self._violation_count[cid] >= self.patience:
                    self.excluded_clients.add(cid)
                    newly_excluded.append(cid)
                    continue
            else:
                self._violation_count[cid] = 0

            kept_results.append((proxy, fit_res))

        if len(kept_results) < self.min_clients_for_aggregation:
            kept_results = [r for r in results if str(r[1].metrics.get("cid", r[0].cid)) not in self.excluded_clients]

        self.removal_log[server_round] = newly_excluded

        print(
            f"[PID] Round {server_round}: Dynamic Threshold = {dynamic_threshold:.4f} | "
            f"{len(kept_results)}/{len(results)} clients kept | Excluded: {newly_excluded}"
        )

        return super().aggregate_fit(server_round, kept_results, failures)

    def aggregate_evaluate(self, server_round: int, results, failures):
        if not results:
            return None, {}
        kept = [r for r in results if str(r[1].metrics.get("cid", r[0].cid)) not in self.excluded_clients]
        return super().aggregate_evaluate(server_round, kept if kept else list(results), failures)