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
import sys
sys.path.append('/mnt/disk1/xueying/jtvae_att/jtvae-trans/Jaejer/icml18-jtnn')
sys.path.append('/mnt/disk1/xueying/jtvae_att/jtvae-trans/Jaejer/icml18-jtnn/jtnn')
sys.path.append('/mnt/disk1/xueying/jtvae_att/jtvae-trans/Jaejer/JAEGER/src')
import torch

import numpy as np
import pandas as pd

from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
from tqdm import tqdm
# --- disable rdkit warnings
from rdkit import RDLogger
import matplotlib.pyplot as plt
import re

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# --- JAEGER
import jaeger as jgr

# --- parse params

import argparse

# --- JT-VAE
from jtnn import *
from jtnn.chemutils import *
# from jtnn.jtprop_vae import JTPropVAE
# from jtnn.jtprop_vae_cross_att import JTPropVAE
from jtnn.jtprop_vae_cross_att_gs import JTPropVAE

from jaeger.utils.jtvae_utils import *

import os
import glob

# USER ARGS

from jaeger.utils.jtvae_utils import load_data

def validate_multi(csv_folder, model_folder, assay_id, output_csv, use_qualified=True, filter_mols=True, pac50=False):
    # List CSV files
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_folder}")
        return

    # List model files
    model_files = glob.glob(os.path.join(model_folder, "*.iter-*"))
    if not model_files:
        print(f"No .pt model files found in {model_folder}")
        return

    # Assay directories
    assay_dir = jgr.BASE_DIR + "/" + str(assay_id)
    jtvae_dir = assay_dir + "/jtvae"
    model_name = 'jtvae-h-420-l-56-d-7'  # Assume same model params
    model_dir = jtvae_dir + "/" + model_name

    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_params = dict(hidden_size=420, latent_size=56, depth=7)

    # Load vocab from first CSV
    first_csv = csv_files[0]
    drop_qualified = not use_qualified
    _, _, toxdata = load_data(first_csv, pac50=pac50, drop_qualified=drop_qualified, filter_mols=filter_mols)
    vocab = get_vocab(assay_dir, assay_id, toxdata)

    # Results list
    results = []

    for csv_file in csv_files:
        print(f"Processing CSV: {csv_file}")
        test_set = os.path.basename(csv_file).replace(".csv", "")

        # Load data for this CSV
        _, _, toxdata = load_data(csv_file, pac50=pac50, drop_qualified=drop_qualified, filter_mols=filter_mols)

        for model_file in tqdm(model_files):
            print(f"Evaluating model: {model_file}")
            model = JTPropVAE(vocab, **model_params).to(device)
            param = torch.load(model_file, map_location=device)
            model.load_state_dict(param)
            model = model.eval()

            # Evaluate
            scores, coords = evaluate_predictions_model(model, toxdata.smiles, toxdata.val, None, assay_id)

            # Collect row
            row = {
                "test_set": test_set,
                "model": os.path.basename(model_file),
            }
            row.update(scores)
            results.append(row)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")


    df = results_df

    def map_test_set(name):
        name = name.lower()
        if "train" in name:
            return "train"
        elif "valid" in name:
            return "valid"
        else:
            return name   # 兜底，防止异常

    df["test_set_label"] = df["test_set"].apply(map_test_set)

    # 提取 iter
    df["iter"] = df["model"].apply(
        lambda x: int(re.search(r"iter-(\d+)", x).group(1))
    )

    # 提取 stage
    df["stage"] = df["model"].apply(
        lambda x: "pre" if "model-pre" in x else "ref"
    )

    # stage 顺序
    df["stage_order"] = df["stage"].map({"pre": 0, "ref": 1})

    # 排序：pre.iter.0→35 再 ref.iter.0→35
    df = df.sort_values(by=["stage_order", "iter"])

    # 构造连续 x 轴
    df["step"] = df["stage_order"] * (df["iter"].max() + 1) + df["iter"]

    metrics = ["AUPR[hmean]", "Precision[1]", "Recall[1]"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))

        for label in df["test_set_label"].unique():
            df_ts = df[df["test_set_label"] == label]

            plt.plot(
                df_ts["step"],
                df_ts[metric],
                marker="o",
                label=label
            )

        # pre / ref 分割线
        split = df["iter"].max() + 0.5
        plt.axvline(split, color="gray", linestyle="--", alpha=0.6)

        plt.xlabel("Iteration (pre → ref)")
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()

        fname = metric.replace("[", "_").replace("]", "").replace(" ", "")
        plt.savefig(f"TB_{fname}.png", dpi=300)
        print(f"Saved figure {fname}.png")
        plt.close()



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



def main():
    parser = argparse.ArgumentParser(
        description=jgr.NAME + " validate multi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv_folder", type=str, default="", help="Folder containing CSV files",
    )
    parser.add_argument(
        "--model_folder", type=str, default="", help="Folder containing .pt model files",
    )
    parser.add_argument(
        "--assay_id", type=str, default="", help="Assay ID",
    )
    parser.add_argument(
        "--output_csv", type=str, default="tb_validation_results.csv", help="Output CSV file for results",
    )
    parser.add_argument(
        '--is_ac50', action='store_true', help="Use AC50 instead of PAC50"
    )
    parser.add_argument(
        "--use_qualified",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use molecules with qualified values",
    )
    parser.add_argument(
        "--drop_larger_mols",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Drop molecules with 50 or more atoms",
    )


    args = parser.parse_args()
    validate_multi(args.csv_folder, args.model_folder, args.assay_id, args.output_csv, args.use_qualified, filter_mols=args.drop_larger_mols, pac50=args.is_ac50)




if __name__ == "__main__":
    main()
