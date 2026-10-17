# Author: Kai Fan kf5601 
# Author: Tyriz Newton tn1207
# Class: CSCI 532 Introduction to Intelligent Security Systems
# Professor: Dr.Leon Reznik
# File: src/train.py

import sys
import os
from pathlib import Path

# DO NOT TOUCH!!! FIXES FOR WINDOWS MULTI-THREADING CRASHES since multiple clients spawn by ray according to StackOverflow
# Force Matplotlib to use the non-GUI 'Agg' backend before importing pyplot. otherwise GUI gets all 0s in graph
# Prevents background Ray client threads from invoking Tkinter GUI calls that crash C-extensions.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Insert src directory into Python module search path so Ray worker processes locate dataset and model
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

# DIRECTORY and HYPERPARAMETER SETUP 
PROJECT_ROOT = Path(SRC_DIR).parent
DATASET_PATH = str(PROJECT_ROOT / "femnist_dataset")
NO_ATTACK_DIR = str(PROJECT_ROOT / "results" / "no_attack")
ATTACK_DIR = str(PROJECT_ROOT / "results" / "attack")

NUM_CLIENTS = 20 # Total virtual clients in simulation
ROUNDS = 10 # Federated communication rounds
BATCH_SIZE = 32 # Local training batch size
LEARNING_RATE = 0.001 

# making sure target results folders exist before running experiments
os.makedirs(NO_ATTACK_DIR, exist_ok=True)
os.makedirs(ATTACK_DIR, exist_ok=True)

# Hardware acceleration setup (NVIDIA CUDA GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Flower Virtual Client
class FlowerClient(fl.client.NumPyClient):
    """
    Flower client implementation representing a single virtual node.
    Handles local model training (`fit`) and evaluation (`evaluate`).
    """

    # Initializer
    def __init__(self, cid: str, train_loader, test_loader):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = DigitCNN().to(device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """
        Updates local model weights with global weights received from server.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Trains model locally on client data for 1 epoch.
        
        Returns: tuple of (updated_weights, num_train_samples, metric_dict)
        """
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
        """
        Evaluates global model performance on client's local test subset.
        Returns: tuple of (loss_float, num_test_samples, metrics_dict)
        """
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

        # DO NOT TOUCH: FIX FOR RAY ACTOR HASHING according to stack overflow
        # Explicitly pass integer client ID ('cid') in metrics dict to bypass Ray string proxy hash mismatches.
        return float(avg_loss), total, {"accuracy": float(accuracy), "cid": int(self.cid)}


# Metrics logging
class MetricLoggingStrategy(fl.server.strategy.FedAvg):
    """
    Subclasses standard Federated Averaging (FedAvg) strategy to record
    individual client loss and evaluation accuracy at the end of every round.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history_loss = {}  # Round index -> {cid: loss}
        self.history_acc = {}   # Round index -> {cid: accuracy}

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Called automatically after all clients complete local evaluation.
        """
        loss_agg, metrics_agg = super().aggregate_evaluate(server_round, results, failures)

        r_loss, r_acc = {}, {}
        for client_proxy, eval_res in results:
            # Extract client ID from metrics dict (fallback to proxy ID if missing)
            cid_val = eval_res.metrics.get("cid", client_proxy.cid)
            cid = str(cid_val)
            r_loss[cid] = eval_res.loss
            r_acc[cid] = eval_res.metrics.get("accuracy", 0.0)

        self.history_loss[server_round] = r_loss
        self.history_acc[server_round] = r_acc
        return loss_agg, metrics_agg


# Experiment Simulation Runner
def run_fl_experiment(poisoned_cids=[], num_rounds=10):
    """
    Do a full FL simulation run using Flower Virtual Client Engine.

    Args:
        poisoned_cids (list): List of client IDs designated as poisoned (e.g. [0, 1, ..., 9])
        num_rounds (int): Total training rounds

    Returns:
        tuple of (loss_matrix_2d, accuracy_matrix_2d) of shape (num_rounds, NUM_CLIENTS)
    """
    client_dataloaders = {}
    for c_id in range(NUM_CLIENTS):
        cid_str = str(c_id)
        is_poisoned = c_id in poisoned_cids
        tr, te = get_client_dataloaders(DATASET_PATH, c_id, batch_size=BATCH_SIZE, poison=is_poisoned)
        if tr is not None and te is not None:
            client_dataloaders[cid_str] = (tr, te)

    # Factory callback function spawned by Flower for each virtual client
    def client_fn(cid: str) -> fl.client.Client:
        tr, te = client_dataloaders[cid]
        return FlowerClient(cid, tr, te).to_client()

    # FedAvg strategy tracking all 20 clients
    strategy = MetricLoggingStrategy(
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    # Launch Flower simulation using Ray backend
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args={"runtime_env": {"working_dir": SRC_DIR}, "include_dashboard": False},
    )

    # Convert dictionary round logs into 2D NumPy matrices (Rounds x Clients)
    loss_matrix = np.zeros((num_rounds, NUM_CLIENTS))
    acc_matrix = np.zeros((num_rounds, NUM_CLIENTS))

    for r in range(1, num_rounds + 1):
        for c in range(NUM_CLIENTS):
            cid_str = str(c)
            loss_matrix[r - 1, c] = strategy.history_loss.get(r, {}).get(cid_str, 0.0)
            acc_matrix[r - 1, c] = strategy.history_acc.get(r, {}).get(cid_str, 0.0)

    return loss_matrix, acc_matrix


# Export CSV and make MatPlot graphs
def save_results_and_plots(loss_hist, acc_hist, output_dir, poisoned_ids=[]):
    """
    Saves CSV metrics and generates Matplotlib line plots.
    
    Outputs: Round,Client_0,Client_1,...
    """
    rounds, num_clients = loss_hist.shape
    r_axis = np.arange(1, rounds + 1)

    # Save CSV metric files (index_label="Round" removes leading CSV comma issue)
    pd.DataFrame(loss_hist, index=r_axis, columns=[f"Client_{i}" for i in range(num_clients)]).to_csv(
        os.path.join(output_dir, "loss_per_client.csv"), index_label="Round"
    )
    pd.DataFrame(acc_hist, index=r_axis, columns=[f"Client_{i}" for i in range(num_clients)]).to_csv(
        os.path.join(output_dir, "acc_per_client.csv"), index_label="Round"
    )
    pd.DataFrame(loss_hist.mean(axis=1), index=r_axis, columns=["Avg_Loss"]).to_csv(
        os.path.join(output_dir, "avg_loss.csv"), index_label="Round"
    )
    pd.DataFrame(acc_hist.mean(axis=1), index=r_axis, columns=["Avg_Accuracy"]).to_csv(
        os.path.join(output_dir, "avg_acc.csv"), index_label="Round"
    )

    # Plot 1: History of Loss Per Client Over Rounds
    plt.figure(figsize=(8, 5))
    for c in range(num_clients):
        is_p = c in poisoned_ids
        color = "red" if is_p else "blue"
        alpha = 0.8 if is_p else 0.3
        plt.plot(r_axis, loss_hist[:, c], color=color, alpha=alpha)
    plt.title("History of Loss Per Client Over Rounds")
    plt.xlabel("Round"); plt.ylabel("Loss"); plt.grid(True)
    plt.savefig(os.path.join(output_dir, "history_loss_per_client.png"))
    plt.close()

    # Plot 2: History of Average Loss Among Clients Over Rounds
    plt.figure(figsize=(8, 5))
    plt.plot(r_axis, loss_hist.mean(axis=1), color="black", linewidth=2, marker="o")
    plt.title("History of Average Loss Among Clients Over Rounds")
    plt.xlabel("Round"); plt.ylabel("Average Loss"); plt.grid(True)
    plt.savefig(os.path.join(output_dir, "history_avg_loss.png"))
    plt.close()

    # Plot 3: History of Evaluation Accuracy Per Client Over Rounds
    plt.figure(figsize=(8, 5))
    for c in range(num_clients):
        is_p = c in poisoned_ids
        color = "red" if is_p else "green"
        alpha = 0.8 if is_p else 0.3
        plt.plot(r_axis, acc_hist[:, c], color=color, alpha=alpha)
    plt.title("History of Evaluation Accuracy Per Client Over Rounds")
    plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.grid(True)
    plt.savefig(os.path.join(output_dir, "history_acc_per_client.png"))
    plt.close()

    # Plot 4: History of Average Evaluation Accuracy Among Clients Over Rounds
    plt.figure(figsize=(8, 5))
    plt.plot(r_axis, acc_hist.mean(axis=1), color="black", linewidth=2, marker="s")
    plt.title("History of Average Evaluation Accuracy Among Clients Over Rounds")
    plt.xlabel("Round"); plt.ylabel("Average Accuracy"); plt.grid(True)
    plt.savefig(os.path.join(output_dir, "history_avg_acc.png"))
    plt.close()

def main():
    print(f"Using device: {device}")

    # EXPERIMENT 1: NO ATTACK
    print("\nRunning Clean Simulation (No Attack)")
    loss_clean, acc_clean = run_fl_experiment(poisoned_cids=[], num_rounds=ROUNDS)
    save_results_and_plots(loss_clean, acc_clean, NO_ATTACK_DIR)

    # EXPERIMENT 2: GROUP 5's ATTACK SIMULATION (10/20 POISONED CLIENTS = 50%)
    poisoned_cids = list(range(10)) # Clients 0-9 poisoned, Clients 10-19 clean
    print(f"\nRunning Group 5 Attack Simulation ({len(poisoned_cids)}/20 Poisoned Clients)")
    loss_attack, acc_attack = run_fl_experiment(poisoned_cids=poisoned_cids, num_rounds=ROUNDS)
    save_results_and_plots(loss_attack, acc_attack, ATTACK_DIR, poisoned_ids=poisoned_cids)

    # EXPERIMENT 3: ATTACK SEVERITY SWEEP
    # This one tries different numbers of affected clients to show how problematic additional poisoned clients can be
    print("\n--- Running Attack Severity Sweep ---")
    num_poisoned_list = [4, 10, 16] # 20%, 50%, and 80% compromised client sweeps
    sweep_results = []

    for n_p in num_poisoned_list:
        p_ids = list(range(n_p))
        l_h, a_h = run_fl_experiment(poisoned_cids=p_ids, num_rounds=5)
        sweep_results.append({
            "num_poisoned": n_p,
            "final_loss": l_h[-1].mean(),
            "final_acc": a_h[-1].mean(),
        })

    # Save severity metrics CSV and generate Plot 5
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(os.path.join(ATTACK_DIR, "attack_severity.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df_sweep["num_poisoned"], df_sweep["final_acc"], marker="o", color="red", linewidth=2)
    plt.title("Final Accuracy Dependence on Attack Severity")
    plt.xlabel("Number of Poisoned Clients"); plt.ylabel("Final Average Accuracy"); plt.grid(True)
    plt.savefig(os.path.join(ATTACK_DIR, "attack_severity_dependence.png"))
    plt.close()

    print("\nAll simulations completed! All CSV metrics and plot images generated in results.")


if __name__ == "__main__":
    main()