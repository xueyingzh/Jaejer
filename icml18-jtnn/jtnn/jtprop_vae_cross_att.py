import torch
import torch.nn as nn
from mol_tree import Vocab, MolTree
from nnutils import create_var
from jtnn_enc import JTNNEncoder
from jtnn_dec import JTNNDecoder
from trans_enc import TransformerEncoder
# from cross_attention import CrossAttention
from batch_cross import BatchCrossAttention
from mpn import MPN, mol2graph
from jtmpn import JTMPN
# WG
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity;

from chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols, atom_equal, decode_stereo
import rdkit
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import copy, math

import matplotlib.pyplot as plt
import seaborn as sns
import wandb

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            try:
                wid = vocab.get_index(node.smiles)
            except:
                wid = None
            if wid is not None:
                node.wid = wid
            tot += 1

#yixin debug
class JTPropVAE(nn.Module):
    def __init__(self, vocab,  hidden_size, latent_size, depth, model_pooling='average', out_size=1, prop_loss=nn.MSELoss):
        super(JTPropVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.model_pooing = model_pooling

        self.embedding = nn.Embedding(vocab.size(), hidden_size)
        self.jtnn = JTNNEncoder(vocab, hidden_size, self.embedding)
        self.jtmpn = JTMPN(hidden_size, depth)
        self.transmpn = TransformerEncoder()
        # self.mpn = MPN(hidden_size, depth)
        self.decoder = JTNNDecoder(vocab, hidden_size, int(latent_size / 2), self.embedding)

        self.T_mean = nn.Linear(hidden_size, int(latent_size / 2))
        self.T_var = nn.Linear(hidden_size,  int(latent_size / 2))
        self.G_mean = nn.Linear(hidden_size, int(latent_size / 2))
        self.G_var = nn.Linear(hidden_size,  int(latent_size / 2))
        
        self.T_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.G_fc = nn.Linear(hidden_size * 2, hidden_size)
        # original
        #self.propNN = nn.Sequential(
        #        nn.Linear(self.latent_size, self.hidden_size),
        #        nn.Tanh(),
        #        nn.Linear(self.hidden_size, 1)
        #)
        # WG 11/04
        self.propNN = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.LeakyReLU(0.01),
            ResidualBlock(self.hidden_size),
            ResidualBlock(self.hidden_size),
            ResidualBlock(self.hidden_size),
            ResidualBlock(self.hidden_size),
            ResidualBlock(self.hidden_size),
            ResidualBlock(self.hidden_size),
            ResidualBlock(self.hidden_size),
            nn.Linear(self.hidden_size, out_size)
       )    
        self.cross_attn = BatchCrossAttention(self.hidden_size) 
        # 新增预归一化层
        self.pre_layernorm_jt = nn.LayerNorm(hidden_size)
        self.pre_layernorm_seq = nn.LayerNorm(hidden_size) 
        # # 层归一化, norm1&norm2
        self.layernormjt = nn.LayerNorm(self.hidden_size)
        self.layernormseq = nn.LayerNorm(self.hidden_size) 
        # # 残差norm3
        self.res_layernormjt = nn.LayerNorm(self.hidden_size)
        self.res_layernormseq = nn.LayerNorm(self.hidden_size)
        #self.prop_loss = nn.MSELoss() # WG
        self.prop_loss = prop_loss()
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)
        self.stereo_loss = nn.CrossEntropyLoss(size_average=False)
        self.init()

    def init(self): # WG
        for name, param in self.named_parameters():
            if 'transmpn.model' in name or 'layernorm' in name:
                # print(f"Skipping initialization for {name}")
                continue
        # for param in self.parameters():
            if param.dim() == 1:
                # print(f"Initializing {name} with zeros")
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)        
    
    def encode(self, mol_batch):
        set_batch_nodeID(mol_batch, self.vocab) # 给molTree中的每一个node增加wid(查询在vocab中的index)
        root_batch = [mol_tree.nodes[0] for mol_tree in mol_batch]
        tree_mess,tree_vec,tnode_vecs = self.jtnn(root_batch) # tree_vec:8*420, tree_mess字典，存储图中每个节点对的隐藏状态，这些隐藏状态在模型的前向传播过程中不断更新

        smiles_batch = [mol_tree.smiles for mol_tree in mol_batch]
        # tran_vec = self.mpn(mol2graph(smiles_batch))
        tran_vec, logits = self.transmpn(smiles_batch)
        return tree_mess, tree_vec, tnode_vecs, logits, tran_vec

    def encode_latent_mean(self, smiles_list):
        mol_batch = [MolTree(s) for s in smiles_list]
        for mol_tree in mol_batch:
            mol_tree.recover()

        _, tree_vec, tran_vec = self.encode(mol_batch)
        tree_mean = self.T_mean(tree_vec)
        mol_mean = self.G_mean(tran_vec)
        return torch.cat([tree_mean,mol_mean], dim=1)

    def forward(self, mol_batch, beta, wandb_run=None, total_step_count=0):
        batch_size = len(mol_batch) # list (MolTree, tensor(prop_value)), len = batch_size
        mol_batch, prop_batch = list(zip(*mol_batch)) # mol_batch：list of MolTree, prop_batch:list of tensor values
        tree_mess, tree_vec, tnode_vecs, trans_logits, tran_vec = self.encode(mol_batch) # tree_vec：torch.Size([8, 420])，tnode_vecs：torch.Size([8, 20, 420])，trans_logits：torch.Size([8, 65, 420])，tran_vec：torch.Size([8, 420])
        # print(tree_vec.mean().item(), tree_vec.std().item())
        # print(tran_vec.mean().item(), tran_vec.std().item())
        
        # tnode_mean, tnode_std = tnode_vecs.mean().item(), tnode_vecs.std().item()
        # trans_logits_mean, trans_logits_std = trans_logits.mean().item(), trans_logits.std().item()
        
        tnode_vecs = self.pre_layernorm_jt(tnode_vecs)
        trans_logits = self.pre_layernorm_seq(trans_logits)
        
        # tnode_mean_a, tnode_std_a = tnode_vecs.mean().item(), tnode_vecs.std().item()
        # trans_logits_mean_a, trans_logits_std_a = trans_logits.mean().item(), trans_logits.std().item()
        tree_vec, tran_vec = self.cross_att_norm(tnode_vecs, trans_logits) # tree_vec: torch.Size([8, 420]), tran_vec: torch.Size([8, 420])

        tree_vet_mean, tree_vet_std = tree_vec.mean().item(), tree_vec.std().item()
        tran_vet_mean, tran_vet_std = tran_vec.mean().item(), tran_vec.std().item()
        
        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec)) #Following Mueller et al.
        mol_mean = self.G_mean(tran_vec)
        mol_log_var = -torch.abs(self.G_var(tran_vec)) #Following Mueller et al.

        z_mean = torch.cat([tree_mean,mol_mean], dim=1)
        z_log_var = torch.cat([tree_log_var,mol_log_var], dim=1)
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size

        # create_var 用于创建 PyTorch 变量
        epsilon = create_var(torch.randn(batch_size, int(self.latent_size / 2)), False)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = create_var(torch.randn(batch_size, int(self.latent_size / 2)), False)
        tran_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
        
        
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(mol_batch, tree_vec)
        torch.cuda.empty_cache()
        assm_loss, assm_acc = self.assm(mol_batch, tran_vec, tree_mess)
        stereo_loss, stereo_acc = self.stereo(mol_batch, tran_vec)

        all_vec = torch.cat([tree_vec, tran_vec], dim=1)
        #prop_label = create_var(torch.Tensor(prop_batch))
        prop_label = create_var(torch.stack(prop_batch).squeeze()) # WG
        prop_loss = self.prop_loss(self.propNN(all_vec).squeeze(), prop_label)
        
        loss = word_loss + topo_loss + assm_loss + 2 * stereo_loss + prop_loss + beta * kl_loss
        if wandb_run is not None:
            wandb_run.log({"word_loss": word_loss, "topo_loss": topo_loss, "assm_loss": assm_loss, "stereo_loss": stereo_loss, "prop_loss": prop_loss,\
                           "kl div": kl_loss, "total loss": loss, "tree_vet_mean": tree_vet_mean, "tree_vet_std": tree_vet_std, "tran_vet_mean": tran_vet_mean,\
                            "tran_vet_std": tran_vet_std}, step=total_step_count)
            # wandb_run.log({"word_loss": word_loss, "topo_loss": topo_loss, "assm_loss": assm_loss, "stereo_loss": stereo_loss, "prop_loss": prop_loss,\
            #                "kl div": kl_loss, "total loss": loss, "tree_vet_mean": tree_vet_mean, "tree_vet_std": tree_vet_std, "tran_vet_mean": tran_vet_mean,\
            #                 "tran_vet_std": tran_vet_std, "tnode_mean_before": tnode_mean,  "tnode_std_before": tnode_std,  "trans_logits_mean_before": trans_logits_mean,  "trans_logits_std_before": trans_logits_std, \
            #                      "tnode_mean_after": tnode_mean_a,  "tnode_std_after": tnode_std_a,  "trans_logits_mean_after": trans_logits_mean_a,  "trans_logits_std_after": trans_logits_std_a}, step=total_step_count)
        return loss, kl_loss.item(), word_acc, topo_acc, assm_acc, stereo_acc, prop_loss.item()

    def cross_att_norm(self, tnode_vecs, trans_logits):
        tnode_vecs = self.pre_layernorm_jt(tnode_vecs)
        trans_logits = self.pre_layernorm_seq(trans_logits)
        
        tree_value, tree_attn_weights = self.cross_attn(tnode_vecs, trans_logits) # tnode_vecs: torch.Size([8, 20, 420]),trans_logits： torch.Size([8, 65, 420])
        # q:trans_logits, kv:tnode_vecs
        tran_value, tran_attn_weights = self.cross_attn(trans_logits, tnode_vecs) # tnode_vecs: torch.Size([8, 66, 840]), trans_logits: torch.Size([8, 420])
        
        # # norm1
        # tree_value = self.layernormjt(tree_value)
        # tran_value = self.layernormseq(tran_value)
        # # norm add, norm2
        # tree_value = self.layernormjt(tree_value) + tnode_vecs
        # tran_value = self.layernormseq(tran_value) + trans_logits
        # # norm add norm, norm3
        # tree_value = self.layernormjt(tree_value) + self.res_layernormjt(tnode_vecs)
        # tran_value = self.layernormseq(tran_value) + self.res_layernormseq(trans_logits)
        # norm 5 remove duplicate norm, pre_layernorm
        tree_value = self.layernormjt(tree_value) + tnode_vecs
        tran_value = self.layernormseq(tran_value) + trans_logits
        
        if self.model_pooing == "average":
            # average pooling
            tree_vec = tree_value.mean(dim=1)
            tran_vec = tran_value.mean(dim=1)
        elif self.model_pooing == "max":
            # max pooling, tree_value:torch.Size([8, 23, 420]), tran_value: torch.Size([8, 91, 420])
            tree_vec = torch.max(tree_value, dim=1)[0]  # [batch_size, hidden_size]
            tran_vec = torch.max(tran_value, dim=1)[0]
        elif self.model_pooing == "sum":
            tree_vec = torch.sum(tree_value, dim=1)
            tran_vec = torch.sum(tran_value, dim=1)
            
        return tree_vec, tran_vec
        
    def assm(self, mol_batch, tran_vec, tree_mess):
        # 实现了一个组装模型（assembly model），该模型用于预测分子树的节点如何连接
        '''
        通过计算候选分子与现有分子的相似度来预测分子树的节点如何连接。它首先收集候选分子及其相关信息，然后计算候选分子的向量表示，接着计算相似度分数，并最终计算损失和准确率。
        '''
        # 候选分子
        cands = []
        batch_idx = []
        for i,mol_tree in enumerate(mol_batch):
            for node in mol_tree.nodes:
                #Leaf node's attachment is determined by neighboring node's attachment
                if node.is_leaf or len(node.cands) == 1: continue
                # 候选分子及其相关信息添加到cands列表中，并将批次索引添加到batch_idx列表中
                cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cand_mols] )
                batch_idx.extend([i] * len(node.cands))

        torch.cuda.empty_cache()        
        cand_vec = self.jtmpn(cands, tree_mess) 
        cand_vec = self.G_mean(cand_vec)

        batch_idx = create_var(torch.LongTensor(batch_idx))
        tran_vec = tran_vec.index_select(0, batch_idx)

        tran_vec = tran_vec.view(-1, 1, int(self.latent_size / 2))
        cand_vec = cand_vec.view(-1, int(self.latent_size / 2), 1)
        scores = torch.bmm(tran_vec, cand_vec).squeeze()
        
        cnt,tot,acc = 0,0,0
        all_loss = []
        for i,mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append( self.assm_loss(cur_score.view(1,-1), label) )
        
        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def stereo(self, mol_batch, tran_vec):
        '''
        通过计算候选立体异构体与已知立体异构体的相似度来预测分子的立体化学。它首先收集候选立体异构体及其相关信息，然后计算候选立体异构体的向量表示，接着计算相似度分数，并最终计算损失和准确率。
        '''
        stereo_cands,batch_idx = [],[]
        labels = []
        for i,mol_tree in enumerate(mol_batch):
            cands = mol_tree.stereo_cands
            if len(cands) == 1: continue
            if mol_tree.smiles3D not in cands:
                cands.append(mol_tree.smiles3D)
            stereo_cands.extend(cands)
            batch_idx.extend([i] * len(cands))
            labels.append( (cands.index(mol_tree.smiles3D), len(cands)) )

        if len(labels) == 0: 
            return create_var(torch.zeros(1)), 1.0

        batch_idx = create_var(torch.LongTensor(batch_idx))
        # stereo_cands = self.mpn(mol2graph(stereo_cands))
        stereo_cands, _ = self.transmpn(stereo_cands)
        stereo_cands = self.G_mean(stereo_cands)
        stereo_labels = tran_vec.index_select(0, batch_idx)
        scores = torch.nn.CosineSimilarity()(stereo_cands, stereo_labels)

        st,acc = 0,0
        all_loss = []
        for label,le in labels:
            cur_scores = scores.narrow(0, st, le)
            if cur_scores.data[label] >= cur_scores.max().item(): # if the item with max cosine similarity is below (more similar) that of the index of the known 3D configuration ... shouldn't it be equal? Actually that's the loss below, is saying that. Is there some particular order to the stereo_cands?
                acc += 1
            label = create_var(torch.LongTensor([label]))
            all_loss.append( self.stereo_loss(cur_scores.view(1,-1), label) )
            st += le
        all_loss = sum(all_loss) / len(labels)
        return all_loss, acc * 1.0 / len(labels)

    def reconstruct_encode(self, smiles, prob_decode=False):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _,tree_vec,tran_vec = self.encode([mol_tree])
        # print(tree_vec.shape, tran_vec.shape)
        
        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec)) #Following Mueller et al.
        mol_mean = self.G_mean(tran_vec)
        mol_log_var = -torch.abs(self.G_var(tran_vec)) #Following Mueller et al.

        epsilon = create_var(torch.randn(1, int(self.latent_size / 2)), False)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = create_var(torch.randn(1, int(self.latent_size / 2)), False)
        tran_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
        # print(tree_vec.shape, tran_vec.shape)
        # return self.decode(tree_vec, tran_vec, prob_decode)
        return tree_vec, tran_vec
    
    def reconstruct_decode(self, tree_vec, tran_vec, prob_decode=False, calcu_log=False):
        return self.decode(tree_vec, tran_vec, prob_decode, calcu_log)

    def sample_prior(self, prob_decode=False):
        tree_vec = create_var(torch.randn(1, int(self.latent_size / 2)), False)
        tran_vec = create_var(torch.randn(1, int(self.latent_size / 2)), False)
        return self.decode(tree_vec, tran_vec, prob_decode)

    def optimize(self, smiles, sim_cutoff, lr=2.0, num_iter=20):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _,tree_vec,tran_vec = self.encode([mol_tree])
        
        mol = Chem.MolFromSmiles(smiles)
        fp1 = AllChem.GetMorganFingerprint(mol, 2)

        # VAE calculations happen here.
        # (mean, var) calculation for gaussian draws.
        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec)) #Following Mueller et al.
        mol_mean = self.G_mean(tran_vec)
        mol_log_var = -torch.abs(self.G_var(tran_vec)) #Following Mueller et al.
        mean = torch.cat([tree_mean, mol_mean], dim=1)
        log_var = torch.cat([tree_log_var, mol_log_var], dim=1)
        cur_vec = create_var(mean.data, True)

        visited = [] 
        for step in range(num_iter):
            prop_val = self.propNN(cur_vec).squeeze()
            # here, torch.autograd.grad takes derivative of one tensor w.r.t. another tensor,
            # because all tensor ops are tracked for potential use with backprop anyways.
            # this semantic is different from jax/autograd.grad, inwhich
            # grad will return a function that will then return the gradient.
            grad = torch.autograd.grad(prop_val, cur_vec)[0]
            cur_vec = cur_vec.data - lr * grad.data
            cur_vec = create_var(cur_vec, True)
            # visited records the trajectory of sampled points.
            # this is how the trajectory visualization is calculated.
            visited.append(cur_vec)
        
        l,r = 0, num_iter - 1 
        while l < r - 1:
            mid = int((l + r) / 2)
            new_vec = visited[mid]
            tree_vec,tran_vec = torch.chunk(new_vec, 2, dim=1)
            new_smiles = self.decode(tree_vec, tran_vec, prob_decode=False)
            if new_smiles is None:
                r = mid - 1
                continue

            new_mol = Chem.MolFromSmiles(new_smiles)
            fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
            if sim < sim_cutoff:
                r = mid - 1
            else:
                l = mid
        """
        best_vec = visited[0]
        for new_vec in visited:
            tree_vec,tran_vec = torch.chunk(new_vec, 2, dim=1)
            new_smiles = self.decode(tree_vec, tran_vec, prob_decode=False)
            if new_smiles is None: continue
            new_mol = Chem.MolFromSmiles(new_smiles)
            fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
            if sim >= sim_cutoff:
                best_vec = new_vec
        """
        # this portion grabs out the best vector that was found through optimization search.
        tree_vec,tran_vec = torch.chunk(visited[l], 2, dim=1)
        #tree_vec,tran_vec = torch.chunk(best_vec, 2, dim=1)
        new_smiles = self.decode(tree_vec, tran_vec, prob_decode=False)
        if new_smiles is None:
            return smiles, 1.0, visited, l, tree_vec, tran_vec
        new_mol = Chem.MolFromSmiles(new_smiles)
        fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
        if sim >= sim_cutoff:
            return new_smiles, sim, visited, l, tree_vec, tran_vec
        else:
            return smiles, 1.0, visited, l, tree_vec, tran_vec      
    
    def decode(self, tree_vec, tran_vec, prob_decode, calcu_log = False):
        pred_root,pred_nodes = self.decoder.decode(tree_vec, prob_decode) # pred_root：MolTreeNode, pred_nodes: list of MolTreeNodes

        #Mark nid & is_leaf & atommap
        for i,node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        tree_mess = self.jtnn([pred_root])[0] # root encoder

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol = self.dfs_assemble(tree_mess, tran_vec, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, calcu_log)
        if cur_mol is None:  # cur_mol: rdkit.Chem.rdchem.RWMol可编辑mol object
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        if cur_mol is None: return None

        smiles2D = Chem.MolToSmiles(cur_mol)
        # 处理手性问题
        stereo_cands = decode_stereo(smiles2D) # list 手性化合物
        if len(stereo_cands) == 1: 
            return stereo_cands[0]
        stereo_vecs, _ = self.transmpn(stereo_cands)
        # stereo_vecs = self.mpn(mol2graph(stereo_cands))
        stereo_vecs = self.G_mean(stereo_vecs)
        scores = nn.CosineSimilarity()(stereo_vecs, tran_vec)
        _,max_id = scores.max(dim=0)
        return stereo_cands[max_id.item()]

    def dfs_assemble(self, tree_mess, tran_vec, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode, calcu_log = False):
        # tree_mess: tree root encoder, junction tree decoder pred_root, self.jtnn([pred_root])[0], tran_vec: graph encoder vector
        # all_nodes: all tree pred_nodes, cur_mol: tree root mol, rdkit.Chem.rdchem.RWMol允许修改的mol
        # global_amap: atom map, fa_amap: 父节点atom map, cur_node:开始tree root encoder
        # fa_node: 父节点, prob_decode：decode过程中是否需要计算概率
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0:
            return None
        cand_smiles,cand_mols,cand_amap = list(zip(*cands))

        cands = [(candmol, all_nodes, cur_node) for candmol in cand_mols]

        cand_vecs = self.jtmpn(cands, tree_mess) # cans: rdkit Mol, tree_mess: hidden state of jt root node, dict{(2,1):torch.Size([420])}, 420：hidden_size. output cand_vecs:torch.Size([1, 420])
        cand_vecs = self.G_mean(cand_vecs) # input:torch.Size([1, 420]), output: torch.Size([1, 28])
        tran_vec = tran_vec.squeeze() # tran_vec input: , output: torch.Size([28])
        
        # raise ValueError('stop')
        if calcu_log:
            scores = torch.log(torch.mv(cand_vecs, tran_vec)) # cand_vecs：torch.Size([71, 28])  tran_vec：torch.Size([28]), output:tensor([440.3280], device='cuda:0', grad_fn=<MulBackward0>)
        else: 
            scores = torch.mv(cand_vecs, tran_vec) * 20 # cand_vecs：torch.Size([71, 28])  tran_vec：torch.Size([28]), output:tensor([440.3280], device='cuda:0', grad_fn=<MulBackward0>)
        # 

        if prob_decode:
            # probs = nn.Softmax()(scores.view(1,-1)).squeeze() + 1e-5 #prevent prob = 0
            # cand_idx = torch.multinomial(probs, probs.numel())
            if scores.shape == torch.Size([1]):
                cand_idx = create_var(torch.tensor([0]))
            else:
                probs = nn.Softmax()(scores.view(1,-1)).squeeze() + 1e-5 #prevent prob = 0
                cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _,cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id,ctr_atom,nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap) #father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue
            
            result = True
            for nei_node in children:
                if nei_node.is_leaf: continue
                cur_mol = self.dfs_assemble(tree_mess, tran_vec, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node, prob_decode, calcu_log)
                if cur_mol is None: 
                    result = False
                    break
            if result: return cur_mol

        return None

    # --- WG 
    def encode_single_smile(self, smiles):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _, tree_vec, tnode_vecs, trans_logits, tran_vec = self.encode([mol_tree])

        tree_vec, tran_vec = self.cross_att_norm(tnode_vecs, trans_logits)
        
        tree_mean = self.T_mean(tree_vec)
        mol_mean = self.G_mean(tran_vec)
        return tree_mean, mol_mean
   

    def embed(self, smiles):
        tree_mean, mol_mean = self.encode_single_smile(smiles);
        mean = torch.cat([tree_mean, mol_mean], dim=1)
        cur_vec = create_var(mean.data, False)            
        return cur_vec;

    def predict(self, smiles):   
        cur_vec = self.embed(smiles)
        prop_val = self.propNN(cur_vec).squeeze()
        return prop_val, cur_vec

    def predict_from_embedding(self, embedding):
        cur_vec = create_var(embedding, False)
        prop_val = self.propNN(cur_vec).squeeze()
        return prop_val.cpu().detach().numpy();
    
    def sample_and_predict(self, n_samples):
        tree_vec = create_var(torch.randn(n_samples, int(self.latent_size / 2)), False)
        tran_vec = create_var(torch.randn(n_samples, int(self.latent_size / 2)), False)
        # print(torch.randn(n_samples, int(self.latent_size / 2)).shape)
        # print(tree_vec.shape)
        
        mean = torch.cat([tree_vec, tran_vec], dim=1)
        cur_vec = create_var(mean.data, True)
        predictions = self.propNN(cur_vec).squeeze()

        vectors = cur_vec.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy();

        return vectors, predictions;

    def sample_zero_mean_gaussian_and_predict(self, n_samples, sigma):
        dim =int(self.latent_size)
        center = torch.zeros(dim);
        covariance = torch.eye(dim) * (sigma ** 2);
        m = torch.distributions.MultivariateNormal(center, covariance)
        samples = []
        for i in range(n_samples):
            samples.append(m.sample())
        samples = torch.stack(samples);        
        cur_vec = create_var(samples.data, False)
        predictions = self.propNN(cur_vec).squeeze()

        vectors = cur_vec.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy();

        return vectors, predictions;

    def sample_gaussian_and_predict(self, n_samples, mean, sigma):
        dim =int(self.latent_size)
        center = mean;
        covariance = sigma;
        m = torch.distributions.MultivariateNormal(center, covariance)
        samples = []
        for i in range(n_samples):
            samples.append(m.sample())
        samples = torch.stack(samples);        
        cur_vec = create_var(samples.data, False)
        predictions = self.propNN(cur_vec).squeeze()
        vectors = cur_vec.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy();

        return vectors, predictions;
    

    def plot_attention(self, attn_weights, row_labels, col_labels, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights, annot=False, cmap='viridis', 
                    xticklabels=col_labels, yticklabels=row_labels)
        plt.title(title)
        plt.xlabel("SMILES Tokens")
        plt.ylabel("Tree Nodes")
        plt.tight_layout()
        return plt

    # 在训练或推断后调用
    def log_attention_visualizations(self, tree_attn, tran_attn, tree_labels, smiles_tokens, wandb_run = None):
        # 获取注意力权重
        # tree_attn = model.tree_attn_weights[batch_idx].mean(dim=0)  # 平均多头
        # tran_attn = model.tran_attn_weights[batch_idx].mean(dim=0)
        
        # # 获取标签
        # tree_labels = model.tree_node_smiles[batch_idx]
        # smiles_tokens = model.smiles_tokens_list[batch_idx]
        
        # 生成热力图
        tree_attn_fig = self.plot_attention(tree_attn.cpu().numpy(), 
                                    tree_labels, smiles_tokens, 
                                    "Tree to SMILES Attention")
        tran_attn_fig = self.plot_attention(tran_attn.cpu().numpy(),
                                    smiles_tokens, tree_labels,
                                    "SMILES to Tree Attention")
        
        # 记录到WandB
        wandb_run.log({
            "Tree_to_SMILES_Attention": wandb.Image(tree_attn_fig),
            "SMILES_to_Tree_Attention": wandb.Image(tran_attn_fig)
        })
        plt.close()

  

# WG
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.hidden_size = hidden_size
        self.input_norm = nn.BatchNorm1d(hidden_size) # relu this
        self.ip0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.transform_norm = nn.BatchNorm1d(hidden_size) # relu this
        self.drop_pre_ip1 = nn.Dropout(p=0.25)
        #self.drop_pre_ip1 = nn.Dropout(p=0)
        self.ip1 = nn.Linear(hidden_size, hidden_size, bias=False)

    
    def forward(self, x):
        residual = F.relu ( self.input_norm( x ) )
        residual = self.ip0( residual )
        residual = self.drop_pre_ip1 (
            F.relu (
                self.transform_norm( residual) )
        )
        residual = self.ip1(residual)
        x = x + residual
        return x

        
