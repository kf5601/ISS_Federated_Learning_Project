# Author: Kai Fan kf5601
# Author: Tyriz Newton tn1207
# Class: CSCI 532 Introduction to Intelligent Security Systems
# Professor: Dr.Leon Reznik
# File: src/train_pid.py
#
# Runs the attacked simulation (50% poisoned clients) under the PID strategy,
# saves metrics + plots, logs client-removal dynamics, and produces the
# Part 4 comparison plots against the Conventional FL results.

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from collections import OrderedDict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import flwr as fl
from model import DigitCNN
from dataset import get_client_dataloaders
from pid_strategy import PIDStrategy

# ------------------------------------------------------------------
# Paths & hyperparameters
# ------------------------------------------------------------------
PROJECT_ROOT = Path(SRC_DIR).parent
DATASET_PATH = str(PROJECT_ROOT / "femnist_dataset")
ATTACK_DIR = str(PROJECT_ROOT / "results" / "attack")            # conventional
PID_DIR = str(PROJECT_ROOT / "results" / "pid_attack")           # PID
COMPARE_DIR = str(PROJECT_ROOT / "results" / "comparison")

for d in (ATTACK_DIR, PID_DIR, COMPARE_DIR):
    os.makedirs(d, exist_ok=True)

NUM_CLIENTS = 20
ROUNDS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# PID hyperparameters (spec suggests 1.0, 0.05, 0.5 for FEMNIST)
KP, KI, KD = 1.0, 0.05, 0.5
PID_THRESHOLD = 0.5      # tune this; see notes below
PATIENCE = 1             # exclude after 1 consecutive violation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Flower client (identical to Part 2, imported logic)
# ------------------------------------------------------------------
from train import FlowerClient  # reuse your existing client class


# ------------------------------------------------------------------
# Metrics-logging PID strategy (wraps PIDStrategy to also record
# per-client loss/accuracy histories, mirroring Part 2's logger)
# ------------------------------------------------------------------
class PIDLoggingStrategy(PIDStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history_loss = {}
        self.history_acc = {}

    def aggregate_evaluate(self, server_round, results, failures):
        loss_agg, metrics_agg = super().aggregate_evaluate(
            server_round, results, failures
        )
        r_loss, r_acc = {}, {}
        for proxy, eval_res in results:
            cid_val = eval_res.metrics.get("cid", proxy.cid)
            cid = str(cid_val)
            r_loss[cid] = eval_res.loss
            r_acc[cid] = eval_res.metrics.get("accuracy", 0.0)
        self.history_loss[server_round] = r_loss
        self.history_acc[server_round] = r_acc
        return loss_agg, metrics_agg


# ------------------------------------------------------------------
# Simulation runner
# ------------------------------------------------------------------
def run_pid_experiment(poisoned_cids, num_rounds=ROUNDS):
    client_dataloaders = {}
    for c_id in range(NUM_CLIENTS):
        cid_str = str(c_id)
        tr, te = get_client_dataloaders(
            DATASET_PATH, c_id, batch_size=BATCH_SIZE, poison=(c_id in poisoned_cids)
        )
        if tr is not None and te is not None:
            client_dataloaders[cid_str] = (tr, te)

    def client_fn(cid: str) -> fl.client.Client:
        tr, te = client_dataloaders[cid]
        return FlowerClient(cid, tr, te).to_client()

    strategy = PIDLoggingStrategy(
        Kp=KP,
        Ki=KI,
        Kd=KD,
        pid_threshold=PID_THRESHOLD,
        patience=PATIENCE,
        min_clients_for_aggregation=2,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args={"runtime_env": {"working_dir": SRC_DIR}, "include_dashboard": False},
    )

    loss_matrix = np.zeros((num_rounds, NUM_CLIENTS))
    acc_matrix = np.zeros((num_rounds, NUM_CLIENTS))
    for r in range(1, num_rounds + 1):
        for c in range(NUM_CLIENTS):
            cid_str = str(c)
            loss_matrix[r - 1, c] = strategy.history_loss.get(r, {}).get(cid_str, 0.0)
            acc_matrix[r - 1, c] = strategy.history_acc.get(r, {}).get(cid_str, 0.0)

    return loss_matrix, acc_matrix, strategy


# ------------------------------------------------------------------
# Saving metrics + plots (Part 3 requirements)
# ------------------------------------------------------------------
def save_pid_results_and_plots(loss_hist, acc_hist, strategy, output_dir, poisoned_ids):
    rounds, num_clients = loss_hist.shape
    r_axis = np.arange(1, rounds + 1)

    # CSVs
    pd.DataFrame(loss_hist, index=r_axis,
                 columns=[f"Client_{i}" for i in range(num_clients)]).to_csv(
        os.path.join(output_dir, "loss_per_client.csv"), index_label="Round")
    pd.DataFrame(acc_hist, index=r_axis,
                 columns=[f"Client_{i}" for i in range(num_clients)]).to_csv(
        os.path.join(output_dir, "acc_per_client.csv"), index_label="Round")
    pd.DataFrame(loss_hist.mean(axis=1), index=r_axis,
                 columns=["Avg_Loss"]).to_csv(
        os.path.join(output_dir, "avg_loss.csv"), index_label="Round")
    pd.DataFrame(acc_hist.mean(axis=1), index=r_axis,
                 columns=["Avg_Accuracy"]).to_csv(
        os.path.join(output_dir, "avg_acc.csv"), index_label="Round")

    # Client-removal dynamics CSV
    removal_rows = []
    for r in sorted(strategy.removal_log.keys()):
        for cid in strategy.removal_log[r]:
            removal_rows.append({"Round": r, "Excluded_Client": int(cid)})
    pd.DataFrame(removal_rows).to_csv(
        os.path.join(output_dir, "removal_dynamics.csv"), index=False)

    # PID score log CSV (wide format: Round x Client)
    pid_wide = {f"Client_{i}": [] for i in range(num_clients)}
    for r in r_axis:
        for c in range(num_clients):
            pid_wide[f"Client_{c}"].append(
                strategy.pid_log.get(int(r), {}).get(str(c), np.nan))
    pd.DataFrame(pid_wide, index=r_axis).to_csv(
        os.path.join(output_dir, "pid_scores.csv"), index_label="Round")

    # Plots
    def _plot_per_client(data, title, ylabel, fname, c_benign, c_poison):
        plt.figure(figsize=(9, 5))
        for c in range(num_clients):
            is_p = c in poisoned_ids
            plt.plot(r_axis, data[:, c],
                     color=c_poison if is_p else c_benign,
                     alpha=0.85 if is_p else 0.25,
                     linewidth=1.5 if is_p else 1.0)
        plt.title(title); plt.xlabel("Round"); plt.ylabel(ylabel)
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname)); plt.close()

    _plot_per_client(loss_hist, "PID FL: Loss Per Client Over Rounds",
                     "Loss", "history_loss_per_client.png", "blue", "red")
    _plot_per_client(acc_hist, "PID FL: Accuracy Per Client Over Rounds",
                     "Accuracy", "history_acc_per_client.png", "green", "red")

    # Average loss
    plt.figure(figsize=(9, 5))
    plt.plot(r_axis, loss_hist.mean(axis=1), color="black", marker="o", linewidth=2)
    plt.title("PID FL: Average Loss Among Clients Over Rounds")
    plt.xlabel("Round"); plt.ylabel("Average Loss"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "history_avg_loss.png")); plt.close()

    # Average accuracy
    plt.figure(figsize=(9, 5))
    plt.plot(r_axis, acc_hist.mean(axis=1), color="black", marker="s", linewidth=2)
    plt.title("PID FL: Average Accuracy Among Clients Over Rounds")
    plt.xlabel("Round"); plt.ylabel("Average Accuracy"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "history_avg_acc.png")); plt.close()

    # Client-removal dynamics: cumulative count of excluded clients per round
    cumulative = []
    removed_so_far = set()
    for r in range(1, rounds + 1):
        removed_so_far |= set(strategy.removal_log.get(r, []))
        cumulative.append(len(removed_so_far))

    plt.figure(figsize=(9, 5))
    plt.step(r_axis, cumulative, where="mid", color="purple", linewidth=2, marker="o")
    plt.axhline(len(poisoned_ids), color="red", linestyle="--",
                label=f"Ground-truth poisoned = {len(poisoned_ids)}")
    plt.title("PID FL: Cumulative Client-Removal Dynamics")
    plt.xlabel("Round"); plt.ylabel("# Excluded Clients")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "removal_dynamics.png")); plt.close()


# ------------------------------------------------------------------
# Comparison plots vs. Conventional FL (Part 4 artifacts)
# ------------------------------------------------------------------
def make_comparison_plots(pid_loss, pid_acc, conv_loss_path, conv_acc_path):
    conv_avg_loss = pd.read_csv(conv_loss_path)["Avg_Loss"].values
    conv_avg_acc = pd.read_csv(conv_acc_path)["Avg_Accuracy"].values
    pid_avg_loss = pid_loss.mean(axis=1)
    pid_avg_acc = pid_acc.mean(axis=1)
    r_axis = np.arange(1, len(pid_avg_loss) + 1)

    # Average loss comparison
    plt.figure(figsize=(9, 5))
    plt.plot(r_axis[:len(conv_avg_loss)], conv_avg_loss, marker="o",
             label="Conventional FL (attacked)", color="red")
    plt.plot(r_axis, pid_avg_loss, marker="s",
             label="FL with PID (attacked)", color="blue")
    plt.title("Average Loss: Conventional FL vs. FL with PID (Under Attack)")
    plt.xlabel("Round"); plt.ylabel("Average Loss"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARE_DIR, "comparison_avg_loss.png"))
    plt.close()

    # Average accuracy comparison
    plt.figure(figsize=(9, 5))
    plt.plot(r_axis[:len(conv_avg_acc)], conv_avg_acc, marker="o",
             label="Conventional FL (attacked)", color="red")
    plt.plot(r_axis, pid_avg_acc, marker="s",
             label="FL with PID (attacked)", color="blue")
    plt.title("Average Accuracy: Conventional FL vs. FL with PID (Under Attack)")
    plt.xlabel("Round"); plt.ylabel("Average Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARE_DIR, "comparison_avg_acc.png"))
    plt.close()


# ------------------------------------------------------------------
# Attack-severity sweep under PID defense
# ------------------------------------------------------------------
def run_severity_sweep():
    num_poisoned_list = [4, 10, 16]
    rows = []
    for n_p in num_poisoned_list:
        p_ids = list(range(n_p))
        l_h, a_h, _ = run_pid_experiment(p_ids, num_rounds=5)
        rows.append({
            "num_poisoned": n_p,
            "final_loss": float(l_h[-1].mean()),
            "final_acc": float(a_h[-1].mean()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PID_DIR, "attack_severity.csv"), index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(df["num_poisoned"], df["final_acc"], marker="o",
             color="blue", linewidth=2)
    plt.title("FL with PID: Final Accuracy vs. Attack Severity")
    plt.xlabel("Number of Poisoned Clients")
    plt.ylabel("Final Average Accuracy")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(PID_DIR, "attack_severity_dependence.png"))
    plt.close()
    return df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(f"Using device: {device}")

    # Group 5 settings: 10/20 poisoned clients = 50%
    poisoned_cids = list(range(10))

    # EXPERIMENT: PID defense under attack
    print("\n=== FL with PID defense — attacked run ===")
    pid_loss, pid_acc, strategy = run_pid_experiment(poisoned_cids, num_rounds=ROUNDS)
    save_pid_results_and_plots(pid_loss, pid_acc, strategy, PID_DIR, poisoned_cids)

    # Comparison vs. conventional FL (uses Part 2 outputs)
    conv_avg_loss_path = os.path.join(ATTACK_DIR, "avg_loss.csv")
    conv_avg_acc_path = os.path.join(ATTACK_DIR, "avg_acc.csv")
    if os.path.exists(conv_avg_loss_path) and os.path.exists(conv_avg_acc_path):
        make_comparison_plots(pid_loss, pid_acc,
                              conv_avg_loss_path, conv_avg_acc_path)
    else:
        print("[WARN] Conventional FL attack results not found; "
              "run Part 2 (train.py) first to generate comparison plots.")

    # Severity sweep under PID
    print("\n=== PID attack-severity sweep ===")
    run_severity_sweep()

    print("\nPart 3 simulation complete. Artifacts written to:", PID_DIR)
    print("Comparison plots written to:", COMPARE_DIR)


if __name__ == "__main__":
    main()