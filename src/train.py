# Author: Kai Fan kf5601 
# Author: Tyriz Newton tn1207
# Class: CSCI 532 Introduction to Intelligent Security Systems
# Professor: Dr.Leon Reznik
# File: src/train.py

import sys
import os
from pathlib import Path

# Force non-GUI Matplotlib backend BEFORE any pyplot import
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure Ray workers locate modules
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

# Auto-resolve project directories
PROJECT_ROOT = Path(SRC_DIR).parent
DATASET_PATH = str(PROJECT_ROOT / "femnist_dataset")
NO_ATTACK_DIR = str(PROJECT_ROOT / "results" / "no_attack")
ATTACK_DIR = str(PROJECT_ROOT / "results" / "attack")

NUM_CLIENTS = 20
ROUNDS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

os.makedirs(NO_ATTACK_DIR, exist_ok=True)
os.makedirs(ATTACK_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Flower Client Class ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, train_loader, test_loader):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = DigitCNN().to(device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        for images, labels in self.train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        # Explicitly pass self.cid inside metrics to bypass Ray proxy hash mismatch
        return float(avg_loss), total, {"accuracy": float(accuracy), "cid": int(self.cid)}


# --- Custom Strategy to Log Metrics ---
class MetricLoggingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history_loss = {}  # round -> {cid: loss}
        self.history_acc = {}   # round -> {cid: acc}

    def aggregate_evaluate(self, server_round, results, failures):
        loss_agg, metrics_agg = super().aggregate_evaluate(server_round, results, failures)

        r_loss, r_acc = {}, {}
        for client_proxy, eval_res in results:
            # Extract client ID from metrics dict
            cid_val = eval_res.metrics.get("cid", client_proxy.cid)
            cid = str(cid_val)
            r_loss[cid] = eval_res.loss
            r_acc[cid] = eval_res.metrics.get("accuracy", 0.0)

        self.history_loss[server_round] = r_loss
        self.history_acc[server_round] = r_acc
        return loss_agg, metrics_agg


# --- Simulation Function ---
def run_fl_experiment(poisoned_cids=[], flip_rate=1.0, num_rounds=10):
    client_dataloaders = {}
    for c_id in range(NUM_CLIENTS):
        cid_str = str(c_id)
        is_poisoned = c_id in poisoned_cids
        tr, te = get_client_dataloaders(
            DATASET_PATH, c_id, batch_size=BATCH_SIZE, poison=is_poisoned, flip_rate=flip_rate
        )
        if tr is not None and te is not None:
            client_dataloaders[cid_str] = (tr, te)

    def client_fn(cid: str) -> fl.client.Client:
        tr, te = client_dataloaders[cid]
        return FlowerClient(cid, tr, te).to_client()

    strategy = MetricLoggingStrategy(
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

    return loss_matrix, acc_matrix


# --- CSV & Plot Generation ---
def save_results_and_plots(loss_hist, acc_hist, output_dir, poisoned_ids=[]):
    rounds, num_clients = loss_hist.shape
    r_axis = np.arange(1, rounds + 1)

    pd.DataFrame(loss_hist, index=r_axis, columns=[f"Client_{i}" for i in range(num_clients)]).to_csv(
        os.path.join(output_dir, "loss_per_client.csv")
    )
    pd.DataFrame(acc_hist, index=r_axis, columns=[f"Client_{i}" for i in range(num_clients)]).to_csv(
        os.path.join(output_dir, "acc_per_client.csv")
    )
    pd.DataFrame(loss_hist.mean(axis=1), index=r_axis, columns=["Avg_Loss"]).to_csv(
        os.path.join(output_dir, "avg_loss.csv")
    )
    pd.DataFrame(acc_hist.mean(axis=1), index=r_axis, columns=["Avg_Accuracy"]).to_csv(
        os.path.join(output_dir, "avg_acc.csv")
    )

    # Plot 1: Loss Per Client
    plt.figure(figsize=(8, 5))
    for c in range(num_clients):
        is_p = c in poisoned_ids
        color = "red" if is_p else "blue"
        alpha = 0.8 if is_p else 0.3
        label = "Poisoned" if is_p else "Benign"
        plt.plot(r_axis, loss_hist[:, c], color=color, alpha=alpha, label=label if c in (poisoned_ids[:1] if poisoned_ids else [0]) else "")
    plt.title("History of Loss Per Client Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    if poisoned_ids: plt.legend()
    plt.savefig(os.path.join(output_dir, "history_loss_per_client.png"))
    plt.close()

    # Plot 2: Average Loss
    plt.figure(figsize=(8, 5))
    plt.plot(r_axis, loss_hist.mean(axis=1), color="black", linewidth=2, marker="o")
    plt.title("History of Average Loss Among Clients Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "history_avg_loss.png"))
    plt.close()

    # Plot 3: Accuracy Per Client
    plt.figure(figsize=(8, 5))
    for c in range(num_clients):
        is_p = c in poisoned_ids
        color = "red" if is_p else "green"
        alpha = 0.8 if is_p else 0.3
        label = "Poisoned" if is_p else "Benign"
        plt.plot(r_axis, acc_hist[:, c], color=color, alpha=alpha, label=label if c in (poisoned_ids[:1] if poisoned_ids else [0]) else "")
    plt.title("History of Evaluation Accuracy Per Client Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    if poisoned_ids: plt.legend()
    plt.savefig(os.path.join(output_dir, "history_acc_per_client.png"))
    plt.close()

    # Plot 4: Average Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(r_axis, acc_hist.mean(axis=1), color="black", linewidth=2, marker="s")
    plt.title("History of Average Evaluation Accuracy Among Clients Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "history_avg_acc.png"))
    plt.close()


def main():
    print(f"Using device: {device}")

    # 1. Clean FL Evaluation
    print("\n--- Running Clean Simulation (No Attack) ---")
    loss_clean, acc_clean = run_fl_experiment(poisoned_cids=[], num_rounds=ROUNDS)
    save_results_and_plots(loss_clean, acc_clean, NO_ATTACK_DIR)

    # 2. Group 5 Evaluation With Attack (10/20 Poisoned Clients = 50%)
    poisoned_cids = list(range(10))
    print(f"\n--- Running Group 5 Attack Simulation ({len(poisoned_cids)}/20 Poisoned Clients) ---")
    loss_attack, acc_attack = run_fl_experiment(poisoned_cids=poisoned_cids, flip_rate=1.0, num_rounds=ROUNDS)
    save_results_and_plots(loss_attack, acc_attack, ATTACK_DIR, poisoned_ids=poisoned_cids)

    # 3. Severity Sweep (Required 5th Attack Plot)
    print("\n--- Running Attack Severity Sweep ---")
    flip_rates = [0.2, 0.5, 0.8, 1.0]
    num_poisoned_list = [4, 10, 16]
    sweep_results = []

    for n_p in num_poisoned_list:
        p_ids = list(range(n_p))
        for fr in flip_rates:
            l_h, a_h = run_fl_experiment(poisoned_cids=p_ids, flip_rate=fr, num_rounds=5)
            sweep_results.append({
                "num_poisoned": n_p,
                "flip_rate": fr,
                "final_loss": l_h[-1].mean(),
                "final_acc": a_h[-1].mean(),
            })

    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(os.path.join(ATTACK_DIR, "attack_severity.csv"), index=False)

    plt.figure(figsize=(8, 5))
    for n_p in num_poisoned_list:
        sub = df_sweep[df_sweep["num_poisoned"] == n_p]
        plt.plot(sub["flip_rate"], sub["final_acc"], marker="o", label=f"{n_p} Poisoned Clients")
    plt.title("Final Accuracy Dependence on Attack Severity")
    plt.xlabel("Flipping Rate")
    plt.ylabel("Final Average Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(ATTACK_DIR, "attack_severity_dependence.png"))
    plt.close()

    print("\nAll simulations completed successfully! All CSV metrics and plot images generated in results/.")


if __name__ == "__main__":
    main()