"""
Copyright 2021 Novartis Institutes for BioMedical Research Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import importlib
import sys, os
sys.path.append('/mnt/disk1/xueying/jtvae_att/jtvae-trans/Jaejer/icml18-jtnn')
sys.path.append('/mnt/disk1/xueying/jtvae_att/jtvae-trans/Jaejer/icml18-jtnn/jtnn')
sys.path.append('/mnt/disk1/xueying/jtvae_att/jtvae-trans/Jaejer/JAEGER/src')

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_curve, auc
import scipy.stats
from sklearn.model_selection import train_test_split
import itertools
import json
from datetime import datetime
from torch.utils import data
from tqdm import tqdm

# --- JAEGER
import jaeger as jgr

# --- JT-VAE
from jtnn.jtprop_vae_cross_att_gs import JTPropVAE
from jtnn.datautils import ToxPropDataset

from jaeger.utils.jtvae_utils import load_data, get_vocab, extract_and_save_propnn_features

# --- utils
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def aupr(predicted_scores, true_labels):
    """
    Calculate AUPR[0], AUPR[1] and AUPR[hmean] for single experiment.
    :param predicted_scores: list of floats, list of predicted scores, i.e., 1 or 0.
    :param true_labels: list of ints, list of true labels, i.e., 1, 0.
    :return: list of floats, AUPR[0], AUPR[1], AUPR[hmean]
    """
    auprs = []

    # aupr for each class
    predicted_scores = np.array(predicted_scores)
    predicted_scores = np.vstack((1 - predicted_scores, predicted_scores))
    for i in range(2):
        p, r, th = precision_recall_curve(true_labels, predicted_scores[i,:], pos_label=i)
        aupr_val = auc(r, p)
        auprs.append(aupr_val)
    # hmean aupr
    if all(x > 0.0 for x in auprs):
        aupr_hmean = scipy.stats.hmean(auprs)
    else:
        aupr_hmean = 0.0

    return auprs[0], auprs[1], aupr_hmean


def train_propnn(model, train_loader, valid_loader, train_losses, valid_losses, prop_train_losses, prop_valid_losses, train_smiles, train_props, valid_smiles, valid_props, task_type, train_metrics, valid_metrics, epochs=100, lr=1e-3, device='cuda', weight_decay=0, beta=0, patience=5, propnn_frozen = False):
    """Train propNN and return train/valid losses per epoch, with early stopping for propNN"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_valid_prop_loss = float('inf')
    patience_counter = 0
    # propnn_frozen = False

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_train_loss, epoch_train_prop_loss = 0, 0
        for batch in train_loader:
            for mol_tree, _ in batch:
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)
                        node.cand_mols.append(node.label_mol)
            # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss, prop_loss = model(batch, beta)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_train_prop_loss += prop_loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))
        prop_train_losses.append(epoch_train_prop_loss / len(train_loader))

        # Validation
        model.eval()
        epoch_valid_loss, epoch_valid_prop_loss = 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                for mol_tree, _ in batch:
                    for node in mol_tree.nodes:
                        if node.label not in node.cands:
                            node.cands.append(node.label)
                            node.cand_mols.append(node.label_mol)
                loss, prop_loss = model(batch, 0.005)
                epoch_valid_loss += loss.item()
                epoch_valid_prop_loss += prop_loss.item()

        valid_losses.append(epoch_valid_loss / len(valid_loader))
        prop_valid_losses.append(epoch_valid_prop_loss / len(valid_loader))

        # Evaluate metrics on full train and valid sets
        train_epoch_metrics = evaluate_propnn(model, train_smiles, train_props, device, task_type)
        valid_epoch_metrics = evaluate_propnn(model, valid_smiles, valid_props, device, task_type)
        train_metrics.append(train_epoch_metrics)
        valid_metrics.append(valid_epoch_metrics)

        # Early stopping for propNN
        if not propnn_frozen:
            if epoch_valid_prop_loss < best_valid_prop_loss:
                best_valid_prop_loss = epoch_valid_prop_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered for propNN at epoch {epoch+1}. Freezing propNN weights.")
                    for param in model.propNN.parameters():
                        param.requires_grad = False
                    propnn_frozen = True
                    # Optionally, you can adjust optimizer to exclude frozen params
                    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        # else: 
        #     # 打印冻结后的参数requires_grad
        #     print("After freezing the propNN layers:")
        #     for name, param in model.propNN.named_parameters():
        #         print(f"Layer: {name}, requires_grad: {param.requires_grad}")

    return propnn_frozen


def evaluate_propnn(model, smiles, props, device='cuda', task_type='reg'):
    """Evaluate propNN on test set"""
    model.eval()
    preds, probs = [], []
    targets = []
    for k, idx in enumerate(smiles.index):
        sml = smiles.loc[idx]
        targets.append(props.loc[idx])

        out, vec = model.predict(sml)  # 模型输出：回归→tensor；分类→logits/prob
        if task_type == "reg":
            preds.append(float(out.item()))

        else:  # 分类
            out_np = out.detach().cpu().numpy().ravel()
            # 若输出是 logits → 做 softmax/sigmoid
            if len(out_np) == 1:
                prob = 1 / (1 + np.exp(-out_np[0]))  # Convert to probability
                preds = int(prob >= 0.5)  # Classify as 0 or 1
                probs.append(prob)  # Store probability of class 1
            else:
                exp = np.exp(out_np - out_np.max())  # Avoid numerical issues with softmax
                prob_vec = exp / exp.sum()  # Normalize to get class probabilities
                prob = float(prob_vec[1])  # Example: Get probability of class 1
                preds = int(prob_vec.argmax())  # Predicted class
                probs.append(prob_vec)  # Assign the probabilities of all classes

    if task_type == 'reg':
        mse = mean_squared_error(targets, preds)
        return {'mse': mse}
    else:
        auprs = aupr(probs, targets)
        return {'auprs[0]': auprs[0], 'auprs[1]': auprs[1], 'aupr_hmean': auprs[2]}


def plot_losses(train_losses, valid_losses, prop_train_losses, prop_valid_losses, config, save_path):
    """Plot train and valid losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.plot(prop_train_losses, label='Prop Train Loss')
    plt.plot(prop_valid_losses, label='Prop Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'PropNN Training - Config: {config}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_losses2(train_losses, valid_losses, config, save_path):
    """Plot train and valid losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'PropNN Training - Config: {config}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_metrics(train_metrics, valid_metrics, task_type, config, save_path):
    """Plot train and valid metrics over epochs"""
    plt.figure(figsize=(10, 6))
    if task_type == 'reg':
        train_scores = [m['mse'] for m in train_metrics]
        valid_scores = [m['mse'] for m in valid_metrics]
        ylabel = 'MSE'
    else:
        train_scores = [m['aupr_hmean'] for m in train_metrics]
        valid_scores = [m['aupr_hmean'] for m in valid_metrics]
        ylabel = 'AUPR HMean'
    plt.plot(train_scores, label='Train Score')
    plt.plot(valid_scores, label='Valid Score')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'PropNN Metrics - Config: {config}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PropNN Hyperparameter Search")
    parser.add_argument("--csv_train", type=str, required=True, help="CSV file path")
    parser.add_argument("--csv_valid", type=str, required=True, help="CSV file path")
    parser.add_argument("--assay_id", type=str, required=True, help="Assay ID")
    parser.add_argument("--use_qualified", type=str2bool, default=True)
    parser.add_argument("--filter_mols", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="propnn_hpo")
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_threads", type=int, default=12, help="Number of workers")
    parser.add_argument(
        '--use_vocab', type=bool, default=True
    )
    parser.add_argument(
        '--is_ac50', action='store_true'
    )
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping on propNN")

    args = parser.parse_args()

    # Initialize wandb
    run = wandb.init(project=args.wandb_project, name=f"propnn_hpo_{args.assay_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Load data
    drop_qualified = not args.use_qualified
    _, _, toxdata_train = load_data(args.csv_train, drop_qualified=drop_qualified, filter_mols=args.filter_mols, pac50=args.is_ac50)
    _, _, toxdata_valid = load_data(args.csv_valid, drop_qualified=drop_qualified, filter_mols=args.filter_mols, pac50=args.is_ac50)

    # Determine task type
    uniq = np.unique(toxdata_train.val)
    if len(uniq) <= 10 and np.allclose(uniq, uniq.astype(int)):
        task_type = "clf"
    else:
        task_type = "reg"

    print(f"Detected task type: {task_type}")

    # Load pre-trained JT-VAE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_params = dict(hidden_size=420, latent_size=args.latent_size, depth=7, task_type=task_type)
    model_name = (
        "jtvae-h-"
        + str(model_params["hidden_size"])
        + "-l-"
        + str(model_params["latent_size"])
        + "-d-"
        + str(model_params["depth"])
    )
    assay_dir = jgr.BASE_DIR + "/" + str(args.assay_id)
    model_dir = assay_dir + "/jtvae/" + model_name
    print(f"Model directory {model_dir}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)  # 如果父目录不存在，会创建父目录

    # --- derive model for inference / molecule optimization
    infer_dir = model_dir + "/infer/"
    if not os.path.exists(infer_dir):
        os.mkdir(infer_dir)

    train_dataset = ToxPropDataset(toxdata_train.smiles, toxdata_train.val)
    valid_dataset = ToxPropDataset(toxdata_valid.smiles, toxdata_valid.val)
    batch_size = 8
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_threads,
        collate_fn=lambda x: x,
        drop_last=True,
    )
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_threads,
        collate_fn=lambda x: x,
        drop_last=False,
    )

    # Hyperparameter search space
    if task_type == 'reg':
        hpo_configs = {
            'res_block': [2, 4, 6],
            'hidden_multiplier': [1, 2, 3],  # hidden_size = latent_size * multiplier
            'dropout': [0.1, 0.25, 0.4],
            'lr': [1e-4, 1e-3, 1e-2],
            'weight_decay': [0.0, 1e-4, 1e-3]
        }
        # hpo_configs = {
        #     'res_block': [2],
        #     'hidden_multiplier': [1, 2],  # hidden_size = latent_size * multiplier
        #     'dropout': [0.1]
        # }
    else:  # clf
        # hpo_configs = {
        #     'res_block': [1, 2, 3, 4],
        #     'hidden_multiplier': [1],
        #     'dropout': [0.1, 0.2, 0.3, 0.4],
        #     'lr': [1e-4, 1e-3, 1e-2],
        #     'weight_decay': [0.0, 1e-4, 1e-3]
        # }
        hpo_configs = {
            'res_block': [1],
            'hidden_multiplier': [1],
            'dropout': [0.2],
            'lr': [1e-4, 1e-3],
            'weight_decay': [0.0, 1e-4]
        }

    # Generate all combinations
    keys = hpo_configs.keys()
    values = hpo_configs.values()
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    best_score = float('inf') if task_type == 'reg' else 0
    best_config = None

    vocab = get_vocab(assay_dir, args.assay_id, toxdata_train, args.use_vocab)
    for i, config in tqdm(enumerate(configs), total=len(configs), desc="Testing configs"):
        print(f"Testing config {i+1}/{len(configs)}: {config}")
        train_losses, valid_losses, prop_train_losses, prop_valid_losses= [], [], [], []
        train_metrics, valid_metrics = [], []
        lr, weight_decay = config['lr'], config['weight_decay']
        del config['lr'], config['weight_decay']
        merged_param = {**model_params, **config}
        model = JTPropVAE(vocab, **merged_param).to(device)
        config['lr'], config['weight_decay'] = lr, weight_decay

        # Train
        print("Starting pretraining...")
        propnn_frozen = train_propnn(model, train_loader, valid_loader, train_losses, valid_losses, prop_train_losses, prop_valid_losses, \
                toxdata_train.smiles, toxdata_train.val, toxdata_valid.smiles, toxdata_valid.val, task_type, \
                train_metrics, valid_metrics, epochs=args.epochs, lr=lr, weight_decay=weight_decay, device=device, beta=0, patience=args.patience)
        print("Starting finetuning...")
        train_propnn(model, train_loader, valid_loader, train_losses, valid_losses, prop_train_losses, prop_valid_losses, \
                toxdata_train.smiles, toxdata_train.val, toxdata_valid.smiles, toxdata_valid.val, task_type, \
                train_metrics, valid_metrics, epochs=args.epochs, lr=lr, weight_decay=weight_decay, device=device, beta=0.005, patience=args.patience, propnn_frozen=propnn_frozen)

        # Evaluate
        train_metrics_final = evaluate_propnn(model, toxdata_train.smiles, toxdata_train.val, device, task_type)
        valid_metrics_final = evaluate_propnn(model, toxdata_valid.smiles, toxdata_valid.val, device, task_type)
        torch.save(model.cpu().state_dict(), model_dir + "/model-config-" + str(i+1))

        train_score = train_metrics_final['mse'] if task_type == 'reg' else train_metrics_final['aupr_hmean']
        valid_score = valid_metrics_final['mse'] if task_type == 'reg' else valid_metrics_final['aupr_hmean']

        # Plot losses
        plot_path = os.path.join(model_dir, f"all_losses_config_{i}.png")
        prop_plot_path = os.path.join(model_dir, f"predictor_losses_config_{i}.png")
        metrics_plot_path = os.path.join(model_dir, f"prop_aupr_config_{i}.png")
        # plot_losses(train_losses, valid_losses, prop_train_losses, prop_valid_losses, config, plot_path)
        plot_losses2(train_losses, valid_losses, config, plot_path)
        plot_losses2(prop_train_losses, prop_valid_losses, config, prop_plot_path)
        plot_metrics(train_metrics, valid_metrics, task_type, config, metrics_plot_path)



        # Log to wandb
        wandb.log({
            f"config_{i}_train_loss": train_losses[-1],
            f"config_{i}_valid_loss": valid_losses[-1],
            f"config_{i}_prop_train_loss": prop_train_losses[-1],
            f"config_{i}_prop_valid_loss": prop_valid_losses[-1],
            f"config_{i}_train_score": train_score,
            f"config_{i}_valid_score": valid_score,
            f"config_{i}_plot": wandb.Image(plot_path),
            f"config_{i}_predictor_plot": wandb.Image(prop_plot_path),
            f"config_{i}_metrics_plot": wandb.Image(metrics_plot_path),
        })

        results.append({
            'config': config,
            'final_train_loss': train_losses[-1],
            'final_valid_loss': valid_losses[-1],
            'metrics': valid_metrics_final,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'prop_train_losses': prop_train_losses,
            'prop_valid_losses': prop_valid_losses,
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics
        })

        # Update best
        if (task_type == 'reg' and valid_score < best_score) or (task_type == 'clf' and valid_score > best_score):
            best_score = valid_score
            best_config = config

    # Save results
    with open(os.path.join(model_dir, f'hpo_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Best config: {best_config}")
    print(f"Best score: {best_score}")

    wandb.log({
        "best_config": str(best_config),
        "best_score": best_score
    })

    run.finish()


if __name__ == "__main__":
    main()
