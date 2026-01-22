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

# --- disable rdkit warnings
from rdkit import RDLogger

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


# USER ARGS

from jaeger.utils.jtvae_utils import load_data

def validate(csv_file, assay_id, use_qualified, filter_mols = True, wandb_name = 'wandb_name', pac50 = False):
    model_name = 'jtvae-h-420-l-56-d-7' #TODO change this
    
    # # --- LOAD DATA
    # if assay_id == '152930':
    #     pac50=False

    # if assay_id == 'malaria_blood_stage':        
    #     pac50=False


    drop_qualified = not use_qualified
    morgans_df, targets, toxdata =load_data(csv_file, pac50=pac50, drop_qualified=drop_qualified, filter_mols = filter_mols)
    print(f'toxdata[:10]: {toxdata[:10]}')
    # --- I/O
    # these dirs are created at training time
    assay_dir = jgr.BASE_DIR + "/" + str(assay_id)
    jtvae_dir = assay_dir + "/jtvae"
    model_dir = assay_dir + "/jtvae/" + model_name
    infer_dir = model_dir + "/infer/"

    # --- LOAD MODEL
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_params = dict(hidden_size=420, latent_size=56, depth=7) 
    vocab = get_vocab(assay_dir, assay_id, toxdata)
    model = JTPropVAE(vocab, **model_params).to(device)
    # param = torch.load(infer_dir + "/model-ref.iter-35")
    files = [
        (f, os.path.getmtime(os.path.join(infer_dir, f)))
        for f in os.listdir(infer_dir)
        if os.path.isfile(os.path.join(infer_dir, f)) and f.startswith("model")
    ]
    # 按修改时间排序（最新文件在最后）
    files_sorted = sorted(files, key=lambda x: x[1])
    # 取最后一个（最新修改的）
    model_file = files_sorted[-1][0] if files_sorted else None
    print(f"Using model: {model_file}")
    param = torch.load(infer_dir + "/" + model_file)
    model.load_state_dict(param) # TODO CHANGE    
    model = model.eval()

    # Print the model architecture
    print("Model Architecture:")
    print(model)
    # exit()

    # Calculate and print the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Number of Parameters: {total_params}")

    # --- CHECK PREDICTIONS
    scores, coords = evaluate_predictions_model(model, toxdata.smiles, toxdata.val, None, wandb_name)
    # coords_df = pd.DataFrame(data=coords, columns=["gt", "pred"])
    coords_df = pd.DataFrame(data=coords, columns=list(coords.keys()))
    input_df = pd.read_csv(csv_file)
    coords_df = pd.concat([input_df.reset_index(drop=True), coords_df.reset_index(drop=True)], axis=1)
    input_file = os.path.basename(csv_file).replace(".csv", "")
    coords_df.to_csv(model_dir + f"/predictions_{input_file}.csv")
    print(f"Saved predictions to: {model_dir}/predictions_{input_file}.csv")

    # --- GET EMBEDDINGS
    # These two could be merged, I think
    from jaeger.utils.jtvae_utils import get_embeddings
    vectors = get_embeddings(model, toxdata)
    latent = pd.DataFrame.from_dict(vectors, orient="index")    
    latent.to_csv(model_dir + f"/embeddings_{input_file}.csv")


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
        description=jgr.NAME + " train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv_file", type=str, default="", help="CSV File",
    )
    parser.add_argument(
        "--assay_id", type=str, default="", help="Assay ID",
    )
    parser.add_argument(
        '--is_ac50', action='store_true'
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
    validate(args.csv_file, args.assay_id, args.use_qualified, filter_mols = args.drop_larger_mols, wandb_name=args.assay_id, pac50 = args.is_ac50)
    



if __name__ == "__main__":
    main()
