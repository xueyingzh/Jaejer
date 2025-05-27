import torch, pickle
import torch.nn as nn
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

import rdkit
import rdkit.Chem as Chem
import pandas as pd
from tqdm.auto import tqdm

sys.path.append('/mnt/disk1/xueying/jtvae-trans/Jaejer/icml18-jtnn')
sys.path.append('/mnt/disk1/xueying/jtvae-trans/Jaejer/icml18-jtnn/jtnn')

from jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=420)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=7)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts,args = parser.parse_args()


def open_object(filename):
    with open(filename, "rb") as input:
        reopened = pickle.load(input)
    return reopened
   
vocab = open_object(opts.vocab_path)
# vocab = Vocab(vocab)

# stereo = True if int(opts.stereo) == 1 else False

model_params = dict(hidden_size=int(opts.hidden_size), latent_size=int(opts.latent_size), depth=int(opts.depth)) 
model = JTPropVAE(vocab, **model_params)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()
model = model.eval()

data = pd.read_csv(opts.test_path)

results = []  # 存储所有结果的列表
acc = 0.0
tot = 0
for smiles in tqdm(data['SMILES'], desc="Processing SMILES"):
    record = {
        'smiles': smiles,
        'recon_acc': 0.0,
    }
    # mol = Chem.MolFromSmiles(smiles)
    # smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
    # try:
    # print('SMILES:', smiles)
    encode_pbar = tqdm(range(10), desc="Encoding", leave=False)
    for i in encode_pbar:
        tree_vec, tran_vec = model.reconstruct_encode(smiles)
        decode_pbar = tqdm(range(100), desc="Decoding", leave=False)
        for j in decode_pbar:
            dec_smiles = model.reconstruct_decode(tree_vec, tran_vec)
            if dec_smiles == smiles:
                acc += 1
        decode_pbar.close()  # 关闭内层进度条
    encode_pbar.close()  # 关闭外层进度条
    record['recon_acc'] = acc / 1000.0  # 更新记录
    # except:
    #     print('Invalid SMILES:', smiles)
    #     continue
    print('SMILES:', smiles, 'Accuracy:', acc / 1000.0)
    results.append(record)
    pd.DataFrame(results).to_csv("reconstruction_accuracy.csv", index=False)
    """
    dec_smiles = model.recon_eval(smiles3D)
    tot += len(dec_smiles)
    for s in dec_smiles:
        if s == smiles3D:
            acc += 1
    print acc / tot
    """

