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
# ---
import os
import random
import sys
import wandb

import numpy as np
import pandas as pd
# import plotly.graph_objects as go
import rdkit.Chem as Chem
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# --- JT-VAE
# from jtnn import *  # not cool, but this is how they do it ...
sys.path.append('/mnt/disk1/xueying/jtvae-trans/Jaejer/icml18-jtnn')
sys.path.append('/mnt/disk1/xueying/jtvae-trans/Jaejer/icml18-jtnn/jtnn')
sys.path.append('/mnt/disk1/xueying/jtvae-trans/Jaejer/JAEGER/src')

from jtnn.datautils import ToxPropDataset
# --- disable rdkit warnings
from datetime import datetime
from rdkit import RDLogger
from tqdm import tqdm
from torch.utils import data
# from toxsquad.data import *
from toxsquad.losses import *
from toxsquad.modelling import *
from toxsquad.visualizations import Visualizations

# --- toxsquad
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


import os
import pickle

# ------------ PRE-PROCESSING ROUTINES ------------
from mol_tree import *
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

def create_var(tensor, requires_grad=None):
    from torch.autograd import Variable
    if requires_grad is None:
        return Variable(tensor).cuda()
    else:
        return Variable(tensor, requires_grad=requires_grad).cuda()

def extract_and_save_propnn_features(model, dataloader, save_path="propnn_valid_features.npz", batch_size=32):
    """
    Extracts latent embeddings (inputs to propNN) and labels from the VAE.
    
    Args:
        model: The trained JTPropVAE model.
        toxdata: The dataframe containing smiles and values (dataset).
        save_path: Where to save the .npz file.
    """
    print(f"Extracting features to {save_path}...")
    model.eval()
    
    # Prepare Data Loader
    # Note: Reusing the specific collation logic from your existing code
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            for mol_tree, _ in batch:
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)
                        node.cand_mols.append(node.label_mol)
            # Unpack batch (list of tuples -> tuple of lists)
            mol_batch, prop_batch = list(zip(*batch))
            
            # Get the latent vector (Concatenated Tree Mean + Mol Mean)
            # Shape: [Batch_Size, Latent_Size]
            latent_vectors = model.encode_for_prop(mol_batch)
            
            # Store features
            all_features.append(latent_vectors.cpu().numpy())
            
            # Store labels
            # Handle tensor scalar wrapping
            labels = [p.item() if isinstance(p, torch.Tensor) else p for p in prop_batch]
            all_labels.append(np.array(labels))

    # Concatenate all batches
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"y shape: {y.shape}, y[:10]: {y[:10]}")
    
    print(f"Saving Features shape: {X.shape}, Labels shape: {y.shape}")
    np.savez(save_path, features=X, labels=y)
    print("Save complete.")

def save_object(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def open_object(filename):
    with open(filename, "rb") as input:
        reopened = pickle.load(input)
    return reopened


def get_vocab(assay_dir, assay_id, toxdata, use_vocab = True):
    filename = assay_dir + "/jtvae/" + str(assay_id) + "-vocab.pkl"
    # use_vocab = False
    if os.path.isfile(filename) and use_vocab:
        print("Re-opening vocabulary file")
        vocab = open_object(filename)
    else:
        print("Deriving vocabulary")
        vocab = set()
        for (
            smiles
        ) in toxdata.smiles:  # I guess here we should only use the training data??
            try:
                mol = MolTree(smiles)
                for c in mol.nodes:
                    vocab.add(c.smiles)
            except Exception as e:
                print('exception: ', e, 'error smiles', smiles) 
        vocab = Vocab(list(vocab))
        save_object(vocab, filename)
    return vocab


# ------------ MODEL OPTIMIZATION ROUTINES ------------
def derive_inference_model(
    toxdata,
    vocab,
    infer_dir,
    model_params,
    vis,
    device,
    model_name,
    base_lr=0.003,
    beta=0.005,
    num_threads = 24,
    weight_decay = 0.000,
    epoch = 36,
    wandb_name = "wandb_name",
):
    # from jtnn.jtprop_vae import JTPropVAE
    # from jtnn.jtprop_vae_cross_att import JTPropVAE
    from jtnn.jtprop_vae_cross_att_gs import JTPropVAE

    shuffle = False
    run = None
    # run = wandb.init(
    #     entity="zhengxueyingbupt-global-health-drug-discovery-institute",
    #     # project="dev",
    #     project='vis_weight_0610',
    #     name=wandb_name,
    #     config={
    #         "model_params": model_params,
    #         "epochs": epoch,
    #         "beta": beta,
    #         "shuffle": shuffle,
    #         "framework": "chemberta transformer",
    #     },
    # )
    smiles = toxdata.smiles
    props = toxdata.val

    uniq = np.unique(props)
    if len(uniq) <= 10 and np.allclose(uniq, uniq.astype(int)):
        task = "clf"
    else:
        task = "reg"

    print(f"Detected task: {task}")
    model_params['task_type'] = task

    print(f'props[:10]: {props[:10]}')
    dataset = ToxPropDataset(smiles, props)
    batch_size = 32
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=True,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=lambda x: x,
        drop_last=True,
    )

    pri_config = {
            'res_block': 2,
            'hidden_multiplier': [1],
            'dropout': 0.3
    }
    lr, weight_decay = 1e-3, 1e-4

    merged_param = {**model_params, **pri_config}
    model = JTPropVAE(vocab, **merged_param).to(device)            
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    scheduler.step()
    
    # extract_and_save_propnn_features(model, dataloader, "propnn_valid_features.npz")
    # exit(0)
    # --- pre-train AE
    total_step_count = 0
    total_step_count = pre_train_jtvae(
        model,
        optimizer,
        scheduler,
        dataloader,
        device,
        infer_dir,
        vis,
        total_step_count,
        model_name,
        MAX_EPOCH=epoch,
        PRINT_ITER=36,
        wandb_run=run,
    )
    # train (set a smaller initial LR, beta to  0.005)
    optimizer = optim.Adam(model.parameters(), lr=0.0003,weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    scheduler.step()
    print("[DEBUG] TRAINING")
    total_step_count = train_jtvae(
        model,
        optimizer,
        scheduler,
        dataloader,
        device,
        infer_dir,
        vis,
        total_step_count,
        beta=0.005,
        model_name=model_name,
        MAX_EPOCH=epoch,
        PRINT_ITER=20,
        wandb_run=run,
    )

    # --- fine tune AE
    # optimizer = optim.Adam(model.parameters(), lr=0.0003)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    # scheduler.step()
    # total_step_count = train_jtvae(model, optimizer, scheduler, dataloader, device, infer_dir, vis, total_step_count, 0.005, model_name, MAX_EPOCH=36, PRINT_ITER=5)


def cross_validate_jtvae(
        toxdata,
        partitions,
        xval_dir,
        vocab,
        model_params,
        device,
        model_name,
        base_lr=0.003,
        vis_host=None,
        vis_port=8912,
        assay_name="",
        num_threads = 24,
        weight_decay = 0.0000
):
    """
    :todo ensure same training parameters are used for inference and cross-val models
    """
    MAX_EPOCH=36
    PRINT_ITER = 5
    run = 0
    scores = []
    for partition in partitions:
        # I/O
        save_dir = xval_dir + "/run-" + str(run)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # vis
        if vis_host is not None:
            vis = Visualizations(
                env_name="jtvae-xval-" + str(assay_name) + "-run-" + str(run), server=vis_host, port=vis_port
            )
        else:
            vis = None
        # data

        smiles = toxdata.smiles.loc[partition["train"]]
        props = toxdata.val.loc[partition["train"]]
        dataset = ToxPropDataset(smiles, props)
        batch_size = 32
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_threads,
            collate_fn=lambda x: x,
            drop_last=True,
        )
        # model
        from jtnn.jtprop_vae import JTPropVAE

        pri_config = {
            'res_block': 2,
            'hidden_multiplier': [1],
            'dropout': 0.3,
            'lr': 1e-3,
            'weight_decay': 1e-4
        }
        merged_param = {**model_params, **pri_config}

        model = JTPropVAE(vocab, **merged_param).to(device)
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
        scheduler.step()


        # pretrain
        print("[DEBUG] PRETRAINING")
        total_step_count = pre_train_jtvae(
            model,
            optimizer,
            scheduler,
            dataloader,
            device,
            save_dir,
            vis,
            0,
            model_name,
            MAX_EPOCH=36,
            PRINT_ITER=20,
        )
        # train (set a smaller initial LR, beta to  0.005)
        optimizer = optim.Adam(model.parameters(), lr=0.0003,weight_decay=weight_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
        scheduler.step()
        print("[DEBUG] TRAINING")
        total_step_count = train_jtvae(
            model,
            optimizer,
            scheduler,
            dataloader,
            device,
            save_dir,
            vis,
            total_step_count,
            beta=0.005,
            model_name=model_name,
            MAX_EPOCH=36,
            PRINT_ITER=20,
        )
        # evaluate (only property prediction accuracy for now)
        scores.append(
            evaluate_predictions_model(
                model,
                toxdata.smiles.loc[partition["test"]],
                toxdata.val.loc[partition["test"]],
                vis,
            )
        )
        # memory management
        del model        
        del optimizer
        torch.cuda.empty_cache()
        run = run + 1
    return scores


def pre_train_jtvae(
    model,
    optimizer,
    scheduler,
    dataloader,
    device,
    model_dir,
    vis,
    total_step_count,
    model_name,
    MAX_EPOCH=36,
    PRINT_ITER=20,
    wandb_run=None,
):
    my_log = open(model_dir + "/loss-pre.txt", "w")
    for epoch in tqdm(range(MAX_EPOCH)):
        
        # print("现在的时间是:" + datetime.now().strftime("%H:%M:%S") + " pre train epoch: " + str(epoch))
        word_acc, topo_acc, assm_acc, steo_acc, prop_acc = 0, 0, 0, 0, 0
        for it, batch in enumerate(dataloader):
            for mol_tree, _ in batch:
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)
                        node.cand_mols.append(node.label_mol)
            model.zero_grad()
            torch.cuda.empty_cache()
            loss, prop_loss = model(batch, beta=0)
            loss.backward()
            optimizer.step()
            # word_acc += wacc
            # topo_acc += tacc
            # assm_acc += sacc
            # steo_acc += dacc
            # prop_acc += pacc
            # if (it + 1) % PRINT_ITER == 0:
            #     word_acc = word_acc / PRINT_ITER * 100
            #     topo_acc = topo_acc / PRINT_ITER * 100
            #     assm_acc = assm_acc / PRINT_ITER * 100
            #     steo_acc = steo_acc / PRINT_ITER * 100
            #     prop_acc = prop_acc / PRINT_ITER
            #     if vis is not None:
            #         vis.plot_loss(word_acc, total_step_count, 1, f"{model_name}_word_acc", "word-acc")
            #         vis.plot_loss(prop_acc, total_step_count, 1, f"{model_name}_prop_acc", "mse")
            #     print(
            #         "Epoch: %d, Step: %d, KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Prop: %.4f"
            #         % (
            #             epoch,
            #             it + 1,
            #             kl_div,
            #             word_acc,
            #             topo_acc,
            #             assm_acc,
            #             steo_acc,
            #             prop_acc,
            #         ),
            #         file=my_log,
            #         flush=True,
            #     )
            #     word_acc, topo_acc, assm_acc, steo_acc, prop_acc = 0, 0, 0, 0, 0
            del loss
            # del kl_div
            # total_step_count = total_step_count + 1
            torch.cuda.empty_cache()
        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        torch.save(
            model.cpu().state_dict(), model_dir + "/model-pre.iter-" + str(epoch)
        )
        torch.cuda.empty_cache()
        model = model.to(device)
    my_log.close()
    return total_step_count


def train_jtvae(
    model,
    optimizer,
    scheduler,
    dataloader,
    device,
    model_dir,
    vis,
    total_step_count,
    beta,
    model_name,
    MAX_EPOCH=36,
    PRINT_ITER=20,
    wandb_run=None,
):
    my_log = open(model_dir + "/loss-ref.txt", "w")
    for epoch in tqdm(range(MAX_EPOCH)):
        # print("现在的时间是:" + datetime.now().strftime("%H:%M:%S") + " train epoch: " + str(epoch))
        word_acc, topo_acc, assm_acc, steo_acc, prop_acc = 0, 0, 0, 0, 0
        for it, batch in enumerate(dataloader):
            for mol_tree, _ in batch:
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)
                        node.cand_mols.append(node.label_mol)
            model.zero_grad()
            torch.cuda.empty_cache()
            loss, prop_loss = model(batch, beta)
            loss.backward()
            optimizer.step()
            # word_acc += wacc
            # topo_acc += tacc
            # assm_acc += sacc
            # steo_acc += dacc
            # prop_acc += pacc
            # if (it + 1) % PRINT_ITER == 0:
            #     word_acc = word_acc / PRINT_ITER * 100
            #     topo_acc = topo_acc / PRINT_ITER * 100
            #     assm_acc = assm_acc / PRINT_ITER * 100
            #     steo_acc = steo_acc / PRINT_ITER * 100
            #     prop_acc /= PRINT_ITER
            #     if vis is not None:
            #         vis.plot_loss(word_acc, total_step_count, 1, model_name, "word-acc")
            #         vis.plot_loss(prop_acc, total_step_count, 1, model_name, "mse")
            #     print(
            #         "Epoch: %d, Step: %d, KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Prop: %.4f"
            #         % (
            #             epoch,
            #             it + 1,
            #             kl_div,
            #             word_acc,
            #             topo_acc,
            #             assm_acc,
            #             steo_acc,
            #             prop_acc,
            #         ),
            #         file=my_log,
            #         flush=True,
            #     )
            #     word_acc, topo_acc, assm_acc, steo_acc, prop_acc = 0, 0, 0, 0, 0
            # if (it + 1) % 1500 == 0:  # Fast annealing
            #    # does this make sense? With the smaller datasets
            #    # we don't get to 1500? Why is this happening?
            #    # I don't quite trust it
            #    # But here, since we call model.cpu()
            #    # we need to move the model to the device again
            #    # else we ran onto that weird issue!
            #    scheduler.step()
            #    print("learning rate: %.6f" % scheduler.get_lr()[0])
            #    #torch.save(
            #    #    model.cpu().state_dict(),
            #    #    model_dir + "/model-ref.iter-%d-%d" % (epoch, it + 1),
            #    #)
            #    model.to(device)
            # del loss
            # del kl_div
            total_step_count = total_step_count + 1
        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        torch.save(
            model.cpu().state_dict(), model_dir + "/model-ref.iter-" + str(epoch)
        )  # is this the expensive part?
        model = model.to(device)
    my_log.close()
    return total_step_count

def evaluate_predictions_model(
    model, 
    smiles, 
    props, 
    vis,  
    wandb_name,
    task=None  # 可为 None（自动判断）、"reg"、"cls"
):
    """
    Extended evaluator: supports both regression & classification.

    Parameters
    ----------
    model : JT-VAE or other model with model.predict(smiles)
    smiles : pandas.Series
    props : pandas.Series (regression: float; classification: int/0-1)
    vis : visualization object
    wandb_name : str
    task : None / "reg" / "cls"
        - None: automatically infer task type based on props
        - "reg": regression
        - "cls": classification

    Returns
    -------
    scores : dict
        regression: {"mse":..., "corr":...}
        classification: {"acc":..., "auroc":..., "auprc":...}
    coords : numpy array or dict
        regression: Nx2 array (actual, predicted)
        classification: dict with keys: y_true, y_pred, y_prob
    """

    # ----------- 1. 推断任务类型 -----------
    if task is None:
        # 离散 label 就视为分类
        uniq = np.unique(props)
        if len(uniq) <= 10 and np.allclose(uniq, uniq.astype(int)):
            task = "clf"
        else:
            task = "reg"

    print(f"Detected task: {task}")

    # ----------- 2. 初始化 WandB -----------
    # run = wandb.init(
    #     entity="zhengxueyingbupt-global-health-drug-discovery-institute",
    #     project='vis_weight_0610',
    #     name=f'{wandb_name}_valid_test_{task}',
    # )
    run = None

    model = model.eval()
    n = len(smiles)

    # 回归时 Nx2；分类时保存列表
    if task == "reg":
        coords = np.zeros((n, 2))
    else:
        y_true, y_pred, y_prob = [], [], []

    # ----------- 3. 预测循环 -----------
    for k, idx in enumerate(smiles.index):
        print_status(k, n)
        sml = smiles.loc[idx]
        y = props.loc[idx]

        try:
            out, vec = model.predict(sml)  # 模型输出：回归→tensor；分类→logits/prob

            if task == "reg":
                pred = float(out.item())
                coords[k, 0] = float(y)
                coords[k, 1] = pred

            else:  # 分类
                out_np = out.detach().cpu().numpy().ravel()
                # 若输出是 logits → 做 softmax/sigmoid
                if len(out_np) == 1:
                    prob = 1 / (1 + np.exp(-out_np[0]))
                    pred = int(prob >= 0.5)
                else:
                    exp = np.exp(out_np - out_np.max())
                    prob_vec = exp / exp.sum()
                    prob = float(prob_vec[1])
                    pred = int(prob_vec.argmax())

                y_true.append(int(y))
                y_pred.append(pred)
                y_prob.append(prob)

        except Exception as e:
            print(f"Error processing SMILES {sml}: {e}")

    model = model.train()

    # ----------- 4. 统计指标 -----------
    scores = {}
    if task == "reg":
        print(coords[:10, :])
        actual = coords[:, 0]
        pred = coords[:, 1]
        mse = np.mean((pred - actual)**2)
        corr = np.corrcoef(pred, actual)[0, 1]

        print(f"MSE: {mse}")
        print(f"Corr: {corr:.4f}")

        scores["mse"] = mse
        scores["corr"] = corr

        # WandB
        table = wandb.Table(columns=["Model", "MSE", "Corr"])
        table.add_data(wandb_name, mse, corr)

        # vis scatter
        if vis is not None:
            vis.plot_scatter_gt_predictions(
                coords, f"MSE={mse:.2f}, r={corr:.2f}", ""
            )

        return scores, coords

    else:
        # --------- 分类任务 metrics ----------
        labels, predictions = np.array(y_true), np.array(y_pred)
        cm = confusion_matrix(labels, predictions, labels=[1, 0])
        print(f'Confusion Matrix:\n{cm}')
        
        tp = cm[0, 0]  # True Positive
        fn = cm[0, 1]  # False Negative
        fp = cm[1, 0]  # False Positive
        tn = cm[1, 1]  # True Negative
        total_samples = tp + tn + fp + fn

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        accuracy = (tp + tn) / total_samples if total_samples != 0 else 0.0
        print(f'Precision: {precision * 100:.2f}% Recall: {recall * 100:.2f}% Accuracy: {accuracy * 100:.2f}%')

        metric_value_dict = aupr(predictions, labels)
        print(f'AUPR[1], AUPR[0], AUPR[hmean]: {metric_value_dict[0] * 100:.2f}%, {metric_value_dict[1] * 100:.2f}%, {metric_value_dict[2]* 100:.2f}%')
        metric = calculate_PR(predictions, labels)
        print(f'Precision[1], Recall[1], Precision[0], Recall[0]: {metric[0] * 100:.2f}%, {metric[1] * 100:.2f}%, {metric[2] * 100:.2f}%, {metric[3] * 100:.2f}%')

        # ------ 返回 ------
        scores = {
            "AUPR[1]": metric_value_dict[0],
            "AUPR[0]": metric_value_dict[1],
            "AUPR[hmean]": metric_value_dict[2],
            "Precision[1]": metric[0],
            "Recall[1]": metric[1],
            "Precision[0]": metric[2],
            "Recall[0]": metric[3],
            "confusion_matrix": cm
        }

        coords = {
            "y_true": np.array(y_true),
            "y_pred": np.array(y_pred),
            "y_prob": np.array(y_prob)
        }

        return scores, coords
    
def calculate_PR(predicted_labels, true_labels):
    """
    Calculate precision and recall for single experiment.
    :param predicted_labels: list of ints, list of predicted labels, i.e., 1 or 0.
    :param true_labels: list of ints, list of true labels, i.e., 1, 0, Nan.
    :return: list of floats, precision_1, recall_1, precision_0, recall_0.
    """
    assert len(predicted_labels) == len(true_labels), 'Error: Number of predicted labels should be the same as that of true labels'
    TP, FP, TN, FN = 0, 0, 0, 0

    for i, pred_label in enumerate(predicted_labels):
        true_label = true_labels[i]
        try:
            pred_label, true_label = int(pred_label), int(true_label)
            if pred_label == 1 and true_label == 1:
                TP += 1
            elif pred_label == 1 and true_label == 0:
                FP += 1
            elif pred_label == 0 and true_label == 0:
                TN += 1
            elif pred_label == 0 and true_label == 1:
                FN += 1
        except:
            continue

    # precision for label 1
    precision_1 = np.nan if TP + FP == 0 else TP * 1.0 / (TP + FP)
    # recall for label 1
    recall_1 = np.nan if TP + FN == 0 else TP * 1.0 / (TP + FN)
    # precision for label 0
    precision_0 = np.nan if TN + FN == 0 else TN * 1.0 / (TN + FN)
    # recall for label 0
    recall_0 = np.nan if TN + FP == 0 else TN * 1.0 / (TN + FP)

    return precision_1, recall_1, precision_0, recall_0


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
        aupr = auc(r, p)
        auprs.append(aupr)
    # hmean aupr
    if all(x > 0.0 for x in auprs):
        aupr_hmean = scipy.stats.hmean(auprs)
    else:
        aupr_hmean = 0.0

    return auprs[1], auprs[0], aupr_hmean

# ------------ MODEL EVALUATION ROUTINES ------------
def evaluate_predictions_model_bk(model, smiles, props, vis, wandb_name):
    """
    Return evaluation objects for JT-VAE model.

    This function will return a list of [mse, r2] for the smiles passed in,
    and also return a 2-col matrix for plotting predicted vs. actual.

    vis object allows us to use Visdom to directly update
    a live performance plot view.

    :param model: JT-VAE model
    :param smiles: Pandas series with SMILES as entries
        We usually pass toxdata.smiles
    :param props: Pandas series with molecular activity or property to predict
    :param vis: Visualization object from toxsquad.visualizations

    :returns: Scores, coords
        - Scores is a list of mean squared error and correlation coefficient
          (for entire smiles batch). This is of length 2.
        - coords are x, y coordinates for the "performance plot"
          (where x=actual and y=predicted).
    """
    run = wandb.init(
        entity="zhengxueyingbupt-global-health-drug-discovery-institute",
        # project="dev",
        project='vis_weight_0610',
        name=f'{wandb_name}_valid_test',
    )
    predictions, feature = dict(), dict()
    n_molecules = len(smiles)
    coords = np.zeros((n_molecules, 2))
    # k = 0;
    model = model.eval()
    for k, idx in enumerate(smiles.index):
        print_status(k, n_molecules)
        sml = smiles.loc[idx]
        prop = props.loc[idx]
        # model.predict(sml) returns a torch tensor
        # on which we need to call .item()
        # to get the actual floating point value out.
        pre, vec = model.predict(sml)
        predictions[idx] = pre.item()
        feature[sml] = vec
        coords[k, 0] = prop.item()
        coords[k, 1] = predictions[idx]
        # k = k + 1;
    model = model.train()
    mse = np.mean((coords[:, 1] - coords[:, 0]) ** 2)
    corr = np.corrcoef(coords[:, 1], coords[:, 0])[0, 1]
    print("MSE: " + str(mse))
    print("Corr: " + str(corr))
    if run is not None:
        table = wandb.Table(columns=["Model", "MSE", "Corr"])

        # 添加行数据
        table.add_data(wandb_name, mse, corr)
        wandb.log({"eval_results": table})
    scores = []
    scores.append(mse)
    scores.append(corr)
    # 提取并处理新张量
    # df_feature = []
    # for k in feature.keys():
    #     df_feature.append(feature[k].cpu().detach().numpy().squeeze())
    # df_feature = pd.DataFrame(df_feature, index=feature.keys(), columns=[f"Feature_{i}" for i in range(840)])

    # df_feature.to_csv("/mnt/disk1/xueying/jtvae/data/trainset/all_data_trans_7341_nov_feature.csv")
    # TODO do reconstruction test

    if vis is not None:
        vis.plot_scatter_gt_predictions(
            coords, f"{mse:.2f}" + "-r: " + f"{corr:.2f}", ""
        )

    return scores, coords




# ------------ LATENT SPACE ROUTINES ------------
from numpy.random import choice
from rdkit import DataStructs
from rdkit.Chem import AllChem


def get_neighbor_along_direction_tree(sample_latent, direction, step_size):
    """
    Direction should be normalized
    Direction is in tree space
    """
    tree_vec, mol_vec = torch.chunk(sample_latent, 2, dim=1)
    new_tree_vec = tree_vec + (direction * step_size)
    new_sample = torch.cat([new_tree_vec, mol_vec], dim=1)
    return new_sample


def get_neighbor_along_direction_graph(sample_latent, direction, step_size):
    """
    Direction should be normalized
    """

    tree_vec, mol_vec = torch.chunk(sample_latent, 2, dim=1)
    # update graph
    new_mol_vec = mol_vec + (
        direction * step_size
    )  # maybe the step size will have to be different?
    new_sample = torch.cat([tree_vec, new_mol_vec], dim=1)
    return new_sample


def get_neighbors_along_directions_tree_then_graph(
    model,
    smiles,
    directions,
    scale_factors,
    direction_graph,
    scale_factor_graph,
    n_neighbors=10,
    val_to_beat=-2,
    max_cosine_distance=1.6,
    direction_graph_plus=None,
    convert_to_pac50=False,
):
    sample_latent = model.embed(smiles)

    n_directions = len(directions)
    new_samples = []

    int_step_sizes = np.arange(-n_neighbors, n_neighbors + 1, 1)
    idx = int_step_sizes == 0
    int_step_sizes = np.delete(int_step_sizes, np.where(idx)[0][0])
    actual_n_neighbors = len(int_step_sizes)

    # dynamic range (this adds a loot of additional samples ... just takes longer)
    step_sizes_graph = np.arange(-n_neighbors, n_neighbors + 1, 1)
    step_sizes_graph = step_sizes_graph * scale_factor_graph
    # fixed range (original implementation)
    step_sizes_graph_original = np.arange(-1, 2, 1)
    step_sizes_graph_original = (
        step_sizes_graph_original * 0.5
    )  # so here the step size is also fixed!

    step_sizes_graph = np.concatenate(
        (step_sizes_graph, step_sizes_graph_original), axis=None
    )

    actual_n_neighbors_graph = len(step_sizes_graph)

    # this is pretty quick, as it's just arimethic operations in latent space
    # todo: since cosine similarity in latent space correlates to an extent with
    # chemical similarity, we could further reduce the number of evaluations based on that
    cos = nn.CosineSimilarity(dim=1)
    for k in range(n_directions):  # iterate over axes
        step_sizes = int_step_sizes * scale_factors[k]
        for i in range(actual_n_neighbors):  # iterate over steps along axis
            sample = get_neighbor_along_direction_tree(
                sample_latent, directions[k], step_sizes[i]
            )  # tree sample
            for j in range(actual_n_neighbors_graph):  # iterate along graph axis
                graph_sample = get_neighbor_along_direction_graph(
                    sample, direction_graph, step_sizes_graph[j]
                )
                # check cosine
                cdistance = 1 - cos(sample_latent, graph_sample)
                if cdistance.item() < max_cosine_distance:
                    new_samples.append(graph_sample)
                # additional direction
                if direction_graph_plus is not None:
                    graph_sample = get_neighbor_along_direction_graph(
                        sample, direction_graph_plus, step_sizes_graph[j]
                    )
                    # check cosine
                    cdistance = 1 - cos(sample_latent, graph_sample)
                    if cdistance.item() < max_cosine_distance:
                        new_samples.append(graph_sample)

    # predict activity and decode samples (probably should be another function, also because this happens ALL the time)
    new_smiles, new_activities, new_samples = predict_and_decode_strict(
        model, new_samples, val_to_beat, convert_to_pac50
    )

    return (
        new_samples,
        new_smiles,
        new_activities,
        sample_latent.squeeze().cpu().detach().numpy(),
    )


# I guess the min val should be informed also relative to the MSE of the model
#
def predict_and_decode_strict(model, new_samples, min_val, convert_to_pac50=False):
    n_samples = len(new_samples)
    new_smiles = []
    new_activities = []
    my_bar = None
    filtered_samples = []
    try:
        import streamlit as st

        st.write("Decoding progress")
        my_bar = st.progress(0)
    except ImportError:
        pass

    for i in range(n_samples):
        if my_bar is not None:
            my_bar.progress((i + 1) / n_samples)

        print_status(i, n_samples)
        prediction = (
            
            model.propNN(new_samples[i]).squeeze().cpu().detach().numpy()
        )  # compute the activity predictions

        if convert_to_pac50: 
            prediction = (prediction - 6) * -1

        # HIGHER IS BETTER    
        prediction_condition = prediction > min_val
            
        if prediction_condition:
            new_activities.append(prediction)
            tree_vec, mol_vec = torch.chunk(new_samples[i], 2, dim=1)
            more_smiles = model.decode(tree_vec, mol_vec, prob_decode=False)
            new_smiles.append(more_smiles)
            new_samples[i] = new_samples[i].squeeze().cpu().detach().numpy()
            filtered_samples.append(new_samples[i])
    return new_smiles, new_activities, filtered_samples


def predict_and_decode(model, new_samples, show_st=False):
    n_samples = len(new_samples)
    new_smiles = []
    new_activities = []
    my_bar = None
    if show_st:
        try:
            import streamlit as st

            st.write("Decoding progress")
            my_bar = st.progress(0)
        except ImportError:
            pass

    for i in range(n_samples):
        if my_bar is not None:
            my_bar.progress((i + 1) / n_samples)

        print_status(i, n_samples)
        prediction = (
            model.propNN(new_samples[i]).squeeze().cpu().detach().numpy()
        )  # compute the activity predictions
        new_activities.append(prediction)
        tree_vec, mol_vec = torch.chunk(new_samples[i], 2, dim=1)
        more_smiles = model.decode(tree_vec, mol_vec, prob_decode=False)
        new_smiles.append(more_smiles)
        new_samples[i] = new_samples[i].squeeze().cpu().detach().numpy()
    return new_smiles, new_activities




def sample_gaussian(mean, sigma, n_samples):
    center = mean
    covariance = sigma
    m = torch.distributions.MultivariateNormal(center, covariance)
    samples = []
    for i in range(n_samples):
        samples.append(m.sample())
    samples = torch.stack(samples)
    return samples

def sample_gaussian_and_predict(model, n_samples, mean, sigma):
    dim = int(model.latent_size)
    center = mean
    covariance = sigma
    m = torch.distributions.MultivariateNormal(center, covariance)
    samples = []
    for i in range(n_samples):
        samples.append(m.sample())
    samples = torch.stack(samples)
    cur_vec = create_var(samples.data, False)
    predictions = model.propNN(cur_vec).squeeze()
    vectors = cur_vec.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()

    return vectors, predictions


def get_embeddings(model, toxdata):
    k = 0
    n_molecules = len(toxdata)
    vectors = {}
    for idx in toxdata.smiles.index:
        print_status(k, n_molecules)
        sml = toxdata.smiles.loc[idx]
        vectors[idx] = model.embed(sml).cpu().detach().numpy().ravel()
        k = k + 1
    return vectors


from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import cosine_similarity


def sample_latent_space(model, latent, n_samples=2000, decode=False):
    mu = torch.from_numpy(np.mean(latent).values).float()
    sigma = torch.from_numpy(np.cov(latent.values.transpose())).float()
    return sample_latent_space_pass_normal(model, mu, sigma, n_samples, decode)


def sample_latent_space_pass_normal(model, mu, sigma, n_samples=2000, decode=False):
    samples, samples_predictions = model.sample_gaussian_and_predict(
        n_samples, mu, sigma
    )  # this is fast
    samples = samples.astype("float64")
    samples_predictions = samples_predictions.astype("float64")
    # dim = int(model_params["latent_size"] / 2)
    dim = int(model.latent_size / 2)
    tree_vec = create_var(torch.from_numpy(samples[:, 0:dim]).float())
    mol_vec = create_var(torch.from_numpy(samples[:, dim : dim * 2]).float())
    samples_decoded = []
    if decode:
        for i in range(n_samples):
            print_status(i, n_samples)
            samples_decoded.append(
                model.decode(
                    tree_vec[i, :].reshape(1, -1),
                    mol_vec[i, :].reshape(1, -1),
                    prob_decode=False,
                )
            )  # this is slow

        samples_decoded_df = pd.DataFrame(data=samples_decoded)
        samples_decoded_df.columns = ["smiles"]
    else:
        samples_decoded_df = None

    return samples, samples_predictions, samples_decoded_df


# ------------ MISC ROUTINES ------------
def print_status(i, maxSteps):
    percent = "0.00"
    percentage = (float(i) / float(maxSteps)) * 100
    divisor = 5
    if i % divisor == 0:
        sys.stdout.write("Progress: %d%%   \r" % (percentage))
        sys.stdout.flush()


# ------------ DISTANCES ROUTINES ------------
def normalize_morgans(morgans):
    morgans_normalized = {}
    for key in morgans.keys():
        fp = morgans[key]
        fp_array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        morgans_normalized[key] = normalize_to_unity(fp_array)
    return morgans_normalized


def normalize_to_unity(fp):
    if np.sum(fp) == 0:
        print("invalid fp")
        return fp
    else:
        return fp / np.sum(fp)




import cadd.sascorer as sascorer
import networkx as nx
# ------------ CHEMISTRY ROUTINES ------------
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem.Descriptors import ExactMolWt


def get_cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    current_cycle_score = cycle_length
    return current_cycle_score


import cadd.sascorer as sascorer
# toxdata should include a mols value
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt

NumHDonors = lambda x: rdMolDescriptors.CalcNumHBD(x)
NumHAcceptors = lambda x: rdMolDescriptors.CalcNumHBA(x)
from rdkit.Chem import Descriptors

TPSA = lambda x: Descriptors.TPSA(x)




def compute_properties(toxdata):
    n_molecules = len(toxdata)
    k = 0
    mw = {}
    na = {}
    log_p = {}
    sas = {}
    cycle_scores = {}
    # more properties
    nhdon= {}
    nhacc = {}
    tpsa = {}
    
    
    for idx in toxdata.index:
        print_status(k, n_molecules)
        mol = toxdata.loc[idx].mols
        try:
            mw[idx] = ExactMolWt(mol)
            log_p[idx] = Descriptors.MolLogP(mol)
            sas[idx] = sascorer.calculateScore(mol)
            cycle_scores[idx] = get_cycle_score(mol)
            na[idx] = mol.GetNumAtoms()
            nhdon[idx] = NumHDonors(mol)
            nhacc[idx] = NumHAcceptors(mol)
            tpsa[idx] = TPSA(mol)
            
        except:
            print("[DEBUG] Error computing properties")
            mw[idx] = np.nan
            log_p[idx] = np.nan
            sas[idx] = np.nan
            cycle_scores[idx] = np.nan
            na[idx] = np.nan
            nhdon[idx] = np.nan
            nhacc[idx] = np.nan
            tpsa[idx]  = np.nan                                    
            continue
        k = k + 1

    props = [
        pd.DataFrame.from_dict(mw, orient="index"),
        pd.DataFrame.from_dict(log_p, orient="index"),
        pd.DataFrame.from_dict(sas, orient="index"),
        pd.DataFrame.from_dict(cycle_scores, orient="index"),
        pd.DataFrame.from_dict(na, orient="index"),
        pd.DataFrame.from_dict(nhdon, orient="index"),
        pd.DataFrame.from_dict(nhacc, orient="index"),
        pd.DataFrame.from_dict(tpsa, orient="index"),                
        
    ]

    props_df = pd.concat(props, axis=1)
    props_df.columns = ["mw", "log_p", "sas", "cycle_scores", "n_atoms","HBD",
                        "HBA",
                        "TPSA"]



    toxdata_props = pd.merge(toxdata, props_df, left_index=True, right_index=True)
    return toxdata_props


def check_for_similarity(ref_smiles, test_smiles, do_tanimoto=True):
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    if do_tanimoto:
        ref_fp = AllChem.GetMorganFingerprint(ref_mol, 2)

    n_test = len(test_smiles)
    test_mols = []
    test_fps = []
    for i in range(n_test):
        try:
            test_mol = Chem.MolFromSmiles(test_smiles[i])
            test_mols.append(test_mol)
            test_fps.append(AllChem.GetMorganFingerprint(test_mol, 2))
        except:
            test_fps.append(
                ref_fp
            )  # this smiles will be dropped because we drop sim == 1
            test_mols.append(ref_mol)
    s = DataStructs.BulkTanimotoSimilarity(ref_fp, test_fps)
    return s, test_mols, test_fps


def check_for_similarity_to_collection_fp(test_fps, ref_fps, do_tanimoto=True):
    similarities = []
    n_test = len(test_fps)
    for i in range(n_test):
        dists = 1 - np.array(DataStructs.BulkTanimotoSimilarity(test_fps[i], ref_fps))
        similar = False
        if any(dists < 0.0001):
            similar = True
        similarities.append(similar)

    return similarities



from toxsquad.data import modelling_data_from_csv


def load_data(csv_file, filter_mols=True,
              drop_qualified=False,
              pac50=True,
              binary_fp = False):
    # ok, from now on (2020 April 23)
    # we'll be using pAC50s
    morgans_df, targets, toxdata = modelling_data_from_csv(csv_file,
                                                                filter_mols=filter_mols,
                                                                drop_qualified =drop_qualified,
                                                                convert_to_pac50 = pac50,
                                                                binary_fp = binary_fp)
    return morgans_df, targets, toxdata


def reconstruct(csv_file, assay_id, filter_mols=True,
              drop_qualified=False,
              pac50=True,
              binary_fp = False):
    import jaeger as jgr
    _, _, toxdata = modelling_data_from_csv(csv_file,
                                            filter_mols=filter_mols,
                                            drop_qualified =drop_qualified,
                                            convert_to_pac50 = pac50,
                                            binary_fp = binary_fp)
    dataset = ToxPropDataset(toxdata.smiles, toxdata.val)
    batch_size = 32
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=True,
        shuffle=False,
        num_workers=24,
        collate_fn=lambda x: x,
        drop_last=True,
    )
    
    assay_dir = jgr.BASE_DIR + "/" + str(assay_id)
    vocab = get_vocab(assay_dir, assay_id, toxdata)
    # --- hardware settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # --- define model
    model_params = dict(hidden_size=420, latent_size=56, depth=7)
    model_name = (
        "jtvae-h-"
        + str(model_params["hidden_size"])
        + "-l-"
        + str(model_params["latent_size"])
        + "-d-"
        + str(model_params["depth"])
    )


        
    from jtnn.jtprop_vae import JTPropVAE
    model = JTPropVAE(vocab, **model_params).to(device)
    
    model_dir = assay_dir + "/jtvae/" + model_name
    infer_dir = model_dir + "/infer/"
    param = torch.load(infer_dir + "/model-ref.iter-0")
    model.load_state_dict(param) # TODO CHANGE
    model = model.eval()
    # for it, batch in enumerate(dataloader):
    #         for mol_tree, _ in batch:
    #             for node in mol_tree.nodes:
    #                 if node.label not in node.cands:
    #                     node.cands.append(node.label)
    #                     node.cand_mols.append(node.label_mol)
    #         tree_vec, mol_vec = model.encode_latent_mean(batch)
    for i in range(len(toxdata)):
        sml = toxdata.smiles.iloc[i]
        tree_feature_vec = model.encode_latent_mean([sml])
        split_tensors = torch.split(tree_feature_vec, 28, dim=1)
        reconstructed_smiles = model.decode(split_tensors[0], split_tensors[1], prob_decode=False)
        print(f"Original: {sml}, Reconstructed: {reconstructed_smiles}")
    

def process_smiles(smiles):
    """处理单个 SMILES 字符串，返回节点集合和错误信息"""
    try:
        mol = MolTree(smiles)
        nodes = {c.smiles for c in mol.nodes}
        return nodes, None
    except Exception as e:
        return set(), (smiles, str(e))
    
def get_vocab_v2(assay_dir, assay_id, toxdata, smiles_col, n_jobs=None):
    from tqdm import tqdm
    from multiprocessing import Pool
    import os, time
    from functools import partial
    filename = os.path.join(assay_dir, "jtvae", f"{assay_id}-vocab.pkl")

    print("Deriving vocabulary")
    start = time.time()
    smiles_list = list(toxdata[smiles_col])
    
    # 多进程并行处理
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(process_smiles, smiles_list),
            total=len(smiles_list),
            desc="Processing SMILES"
        ))
    
    # 合并结果和错误信息
    vocab_set = set()
    errors = []
    for nodes, error in results:
        vocab_set.update(nodes)
        if error:
            errors.append(error)
    
    # 打印错误信息（与原逻辑一致）
    for smiles, e in errors:
        print(f"exception: {e}, error smiles: {smiles}")
    
    vocab = Vocab(list(vocab_set))
    save_object(vocab, filename)
    print(f"Time cost: {time.time() - start} seconds") 
    return vocab


if __name__ == "__main__":
    df = pd.read_csv('/mnt/disk1/xueying/deepmirror/data/genenrated_molecules.csv')
    df['mols'] = df['SMILES'].apply(Chem.MolFromSmiles)
    train = compute_properties(df)
    train.to_csv('/mnt/disk1/xueying/deepmirror/data/genenrated_molecules_with_prop.csv', index=False)
    # reconstruct('/mnt/disk1/xueying/mol-gen/JAEGER/models/training_data/Novartis_GNF_rm_error_5251_v3.csv', 'all_data_trans_7341')
    # _, _, tox_data = load_data('/mnt/disk1/xueying/jtvae/Jaeger/JAEGER/models/training_data/Novartis_GNF_cleaned_with_prop.csv', filter_mols=True, drop_qualified=False, pac50=True)
    # get_vocab_v2('/mnt/disk1/xueying/jtvae/Jaeger/JAEGER/models/assays/topscience', 'topscience', df, 'Cleaned_SMILES', 100)


