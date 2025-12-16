import rdkit
import rdkit.Chem as Chem
import copy
from chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab] # 遍历每个原子，并提取其符号、形式电荷和氢原子总数
        
    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)

class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)
        # 本MolTreeNode对应clique的atom ID
        self.clique = [x for x in clique] #copy
        # 在junction tree层面，通过edge相连的其他MolTreeNode
        self.neighbors = []
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        '''
        分子树中恢复原始分子的结构，并为每个节点生成一个标签（SMILES字符串）。这个标签可以用于后续的分析或可视化
        '''
        # print('MolTreeNode, smiles: ', self.smiles)
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                # 为原始分子中对应索引的原子设置原子映射编号（AtomMapNum）为当前节点的nid
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf: #Leaf node, no need to mark 
                continue
            for cidx in nei_node.clique:
                #allow singleton node override the atom mapping
                # 将clique中的原子的index，设置为node的ID（会重复）
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        self.label_mol = get_mol(self.label)

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label
    
    def assemble(self):
        '''
        从当前节点的邻居节点中筛选出可能的组装候选，并将它们存储在当前节点的cands和cand_mols成员中。这些候选可以用于后续的组装过程
        '''
        # 以下4行，按照neighbor的GetNumAtoms对整个neighbor进行了排序，为什么要如下操作而不是直接sort，猜想是增加sort的速度
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands = enum_assemble(self, neighbors)
        if len(cands) > 0:
            self.cands, self.cand_mols, _ = list(zip(*cands))
            self.cands = list(self.cands)
            self.cand_mols = list(self.cand_mols)
        else:
            self.cands = []
            self.cand_mols = []

class MolTree(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        #print("n_atoms: " + str(self.mol.GetNumAtoms()))

        #Stereo Generation
        # print(f'MolTree: {smiles}')
        mol = Chem.MolFromSmiles(smiles)
        # xueying debug
        Chem.Kekulize(mol, clearAromaticFlags=True)
        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles2D = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles2D)

        cliques, edges = tree_decomp(self.mol)
        self.nodes = []
        root = 0
        for i,c in enumerate(cliques):
            # 一个分子对象mol和一个原子索引列表atoms。函数的目的是从给定的分子中提取包含指定原子的子结构，并返回一个新的分子对象
            cmol = get_clique_mol(self.mol, c)
            # c: atom index
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0:
                root = i
            # except Exception as e:
            #     print(f"Exception occurred: {type(e).__name__}")
            #     print(f"Details: {e.args}")

        for x,y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])
        
        if root > 0:
            self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]

        for i,node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1: #Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()

if __name__ == "__main__":
    # import sys
    # lg = rdkit.RDLogger.logger() 
    # lg.setLevel(rdkit.RDLogger.CRITICAL)

    # cset = set()
    # for i,line in enumerate(sys.stdin):
    #     smiles = line.split()[0]
    #     mol = MolTree(smiles)
    #     for c in mol.nodes:
    #         cset.add(c.smiles)
    # for x in cset:
    #     print(x)
    smiles = 'CC1(C)N=C(N)N=C(N)N1C2=CC=C(Br)C=C2'
    mol_tree = MolTree(smiles)
