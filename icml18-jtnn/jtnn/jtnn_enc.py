import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from mol_tree import Vocab, MolTree
from nnutils import create_var, GRU

#MAX_NB = 8
#MAX_NB = 20 # UDPDATE 20200417 worked well with n_atoms < 50
MAX_NB = 30 # UPDATE 20200629 adme model

class JTNNEncoder(nn.Module):

    def __init__(self, vocab, hidden_size, embedding=None):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)
        self.W = nn.Linear(2 * hidden_size, hidden_size)
    def forward(self, root_batch):
        orders = [] # 存储每个分子树的节点顺序
        for root in root_batch: # root_batch list of root of mol trees, len = batch_size
            order = get_prop_order(root)
            orders.append(order)
        
        h = {}
        max_depth = max([len(x) for x in orders])
        #print("DEBUG max_depth " + str(max_depth))
        padding = create_var(torch.zeros(self.hidden_size), False) # torch.Size([hidden_size])

        for t in range(max_depth):
            prop_list = []
            for order in orders:
                if t < len(order):
                    prop_list.extend(order[t])

            cur_x = [] # 存储当前层节点的特征（如原子类型编码 wid）
            cur_h_nei = [] # 收集每个节点的邻居隐藏状态，排除当前节点对的目标节点 y。
            for node_x,node_y in prop_list:
                x,y = node_x.idx,node_y.idx
                cur_x.append(node_x.wid)

                h_nei = []
                for node_z in node_x.neighbors:
                    z = node_z.idx
                    if z == y: continue
                    h_nei.append(h[(z,x)])

                pad_len = MAX_NB - len(h_nei)
                #print("len(h_nei): " + str(len(h_nei)))
                #print("pad_len: " + str(pad_len))
                h_nei.extend([padding] * pad_len)
                #print("paddedlen(h_nei): " + str(len(h_nei)))
                cur_h_nei.extend(h_nei)

            cur_x = create_var(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x)
            cur_h_nei = torch.cat(cur_h_nei, dim=0).view(-1,MAX_NB,self.hidden_size)

            #print("DEBUG cur_x shape " + str(cur_x.shape))
            #print("DEBUG cur_h_nei shape " + str(cur_h_nei.shape))

            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
            for i,m in enumerate(prop_list):
                x,y = m[0].idx,m[1].idx
                h[(x,y)] = new_h[i]


        # root_vecs, jt_vecs = node_aggregate(root_batch, h, self.embedding, self.W) # root_vecs: torch.Size([8, 420]), jt_vecs: torch.Size([8, 30, 840])
        jt_vecs = node_aggregate_bfs(root_batch, h, self.embedding, self.W) # root_vecs: torch.Size([8, 420]), jt_vecs: torch.Size([8, 30, 840])
        root_vecs = node_aggregate_ori(root_batch, h, self.embedding, self.W)

        return h, root_vecs, jt_vecs

"""
Helper functions
"""

def get_prop_order(root):
    '''
    这个函数的主要目的是通过 BFS 遍历图，生成两个顺序列表 order1 和 order2，并将它们合并为一个顺序列表 order。这个顺序列表可以用于后续的图神经网络处理中。

    order1 和 order2 的区别在于它们的方向：

    order1 是从父节点到子节点的顺序。
    order2 是从子节点到父节点的顺序。
    通过合并这两个顺序列表，可以得到一个完整的节点顺序，用于处理图结构数据。    
    '''
    queue = deque([root])
    visited = set([root.idx])
    root.depth = 0
    order1,order2 = [],[]
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.neighbors:
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth-1].append( (x,y) )
                order2[y.depth-1].append( (y,x) )
    order = order2[::-1] + order1
    return order

def node_aggregate(nodes, h, embedding, W):
    '''
    聚合节点的隐藏状态，并计算节点向量。通过将节点的嵌入向量与邻居节点的隐藏状态相加，并应用线性变换和激活函数，可以得到节点的最终表示
    '''
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_dim # 420
    batch_size = len(nodes)  # 假设nodes按batch组织
    padding = create_var(torch.zeros(hidden_size), False) # torch.Size([420])

    for node_x in nodes:
        x_idx.append(node_x.wid)
        nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
        pad_len = MAX_NB - len(nei)
        nei.extend([padding] * pad_len)
        h_nei.extend(nei)
    # h_nei: list of torch.Size([420]), len = 240 = 8 * 30 = batch_size * MAX_NB
    h_nei = torch.cat(h_nei, dim=0).view(-1,MAX_NB,hidden_size) # h_nei torch.Size([8, 30, 420])
    sum_h_nei = h_nei.sum(dim=1) # sum_h_nei torch.Size([8, 420])
    x_vec = create_var(torch.LongTensor(x_idx)) # x_vec torch.Size([8, 420])
    x_vec = embedding(x_vec) # x_vec torch.Size([8, 420])
    node_vec = torch.cat([x_vec, sum_h_nei], dim=1) # node_vec torch.Size([8, 840])
    
    # 拼接节点自身嵌入与邻居特征矩阵（可选）
    expanded_x = x_vec.unsqueeze(1).expand(-1, MAX_NB, -1)  # [batch_size, MAX_NB, hidden_size]
    combined = torch.cat([expanded_x, h_nei], dim=2)        # [batch_size, MAX_NB, 2*hidden_size]
    
    return nn.ReLU()(W(node_vec)), combined


def node_aggregate_bfs(nodes, h, embedding, W):
    '''
    基于BFS的树节点特征聚合，支持多树并行处理
    - nodes: 多个树的根节点列表（支持批量处理）
    - h: 字典{(子节点idx,父节点idx): 隐藏状态}
    - 返回：各节点特征矩阵，及其邻居组合特征
    '''
    all_node_vecs = []
    hidden_size = embedding.embedding_dim
    padding = create_var(torch.zeros(hidden_size), False)
    max_num_node = 0
    
    # 对每个树进行独立BFS遍历
    for root in nodes:
        queue = deque([root])
        visited = set([root.idx])
        batch_nodes = []
        
        # 树节点处理队列
        while len(queue) > 0:
            # 按层处理提升效率
            current = queue.popleft()
            batch_nodes.append(current)
            # 子节点入队（树结构特有的前向传播方向）
            for child in current.neighbors:  # 假设节点有.children属性
                if child.idx not in visited:
                    visited.add(child.idx)
                    queue.append(child)
        
        max_num_node = max(max_num_node, len(batch_nodes))
        # 批量处理当前层级节点
        x_idx = []
        h_nei = []
        
        for node_x in batch_nodes:
            x_idx.append(node_x.wid)

            # 获取所有父节点的隐藏状态（树结构的上游信息）
            nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]  # 假设有.parents属性
            
            # 动态填充至MAX_NB
            pad_len = MAX_NB - len(nei)
            nei.extend([padding] * pad_len)
            h_nei.extend(nei)

        # 特征计算（保持原处理逻辑）
        h_nei = torch.cat(h_nei, dim=0).view(-1, MAX_NB, hidden_size) # torch.Size([num_node, 30, 420]) num_node, MAX_NB, hidden_size
        sum_h_nei = h_nei.sum(dim=1) # torch.Size([num_node, 420]) num_node, hidden_size
        x_vec = embedding(create_var(torch.LongTensor(x_idx))) # torch.Size([num_node, 420]) num_node, hidden_size
        
        # 特征拼接与变换
        node_vec = torch.cat([x_vec, sum_h_nei], dim=1) # torch.Size([12, 840]) num_node, 2*hidden_size        
        all_node_vecs.append(nn.ReLU()(W(node_vec))) # 0: torch.Size([num_node0, 420]) 1: torch.Size([num_node1, 420])
    
    padded_node_vecs = []
    for vec in all_node_vecs:
        current_nodes = vec.size(0)  # 当前节点数（如12,7等）
        pad_len = max_num_node - current_nodes
        
        # 在节点维度（dim=0）填充零向量
        padded_vec = F.pad(vec, (0, 0, 0, pad_len))  # 参数格式：(左填充, 右填充, 上填充, 下填充)
        padded_node_vecs.append(padded_vec)
    
    stacked_node_vecs = torch.stack(padded_node_vecs, dim=0)  # 形状 [8, 20, 420] batch_size, max_num_node, hidden_size
    # 合并所有树的特征矩阵
    return stacked_node_vecs

def node_aggregate_ori(nodes, h, embedding, W):
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_dim
    padding = create_var(torch.zeros(hidden_size), False)

    for node_x in nodes:
        x_idx.append(node_x.wid)
        nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
        pad_len = MAX_NB - len(nei)
        nei.extend([padding] * pad_len)
        h_nei.extend(nei)
    
    h_nei = torch.cat(h_nei, dim=0).view(-1,MAX_NB,hidden_size)
    sum_h_nei = h_nei.sum(dim=1)
    x_vec = create_var(torch.LongTensor(x_idx))
    x_vec = embedding(x_vec)
    node_vec = torch.cat([x_vec, sum_h_nei], dim=1)
    return nn.ReLU()(W(node_vec))