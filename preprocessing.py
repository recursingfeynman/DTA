import os.path as osp
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

from sklearn.model_selection import train_test_split


def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target] 

VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

def min_max_scale(x, xmin, xmax):
    x = (x - xmin) / (xmax + xmin)
    return x

def data_split_train_val_test(path, split, sizes, scale):

    data_path = osp.join(path, 'raw', 'data.csv')
    data_df = pd.read_csv(data_path)

    # Split data in train:val:test = 8:1:1 with the same random seed as previous study.
    # Please see https://github.com/masashitsubaki/CPI_prediction
    
    if split == 'stratified':
        df_train, df_val = train_test_split(data_df, stratify = data_df['activity'], test_size = sizes[1] + sizes[2])
        df_val, df_test = train_test_split(df_val, stratify = df_val['activity'], test_size = sizes[1] / (sizes[1] + sizes[2]))
    else:
        df_train, df_val = train_test_split(data_df, shuffle = True, test_size = sizes[1] + sizes[2])
        df_val, df_test = train_test_split(df_val, shuffle = True, test_size = sizes[1] / (sizes[1] + sizes[2]))
    
    xmin = df_train['affinity'].min()
    xmax = df_train['affinity'].max()

    df_train.describe().to_csv(osp.join(path, 'raw', 'metadata.csv'))
    
    if scale:
        df_train['affinity'] = df_train['affinity'].apply(lambda x: min_max_scale(x, xmin, xmax))
        df_val['affinity'] = df_val['affinity'].apply(lambda x: min_max_scale(x, xmin, xmax))    
        df_test['affinity'] = df_test['affinity'].apply(lambda x: min_max_scale(x, xmin, xmax))
    
    df_train.to_csv(osp.join(path, 'raw', 'data_train.csv'), index=False)
    df_val.to_csv(osp.join(path, 'raw', 'data_val.csv'), index=False)
    df_test.to_csv(osp.join(path, 'raw', 'data_test.csv'), index=False)

    

    print(f"{path} split done!")
    print("Number of data: ", len(data_df))
    print("Number of train: ", len(df_train))
    print("Number of val: ", len(df_val))
    print("Number of test: ", len(df_test))

def atom_features(atom):
    encoding = one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    encoding += one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10]) 
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]) 
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other']) 
    encoding += [atom.GetIsAromatic()]

    try:
        encoding += one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]
    
    return np.array(encoding)
    
def mol_to_graph(mol):
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature/np.sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    if len(edges) == 0:
        return features, [[0, 0]]

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return features, edge_index


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

class RegressionPreprocessor(InMemoryDataset):

    def __init__(self, root, types = 'train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if types == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif types == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif types == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_val.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_val.pt', 'processed_data_test.pt']


    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass


    def process_data(self, data_path, graph_dict):
        df = pd.read_csv(data_path)

        data_list = []
        for i, row in df.iterrows():
            smi = row['smiles']
            sequence = row['sequence']
            affinity = row['affinity']
            activity = row['activity']

            x, edge_index, edge_attr = graph_dict[smi]

            # caution
            x = (x - x.min()) / (x.max() - x.min())

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len- len(target)))
            else:
                target = target[:target_len]

            # Get Labels
            try:
                data = DATA.Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    activity=torch.LongTensor([activity]),
                    affinity=torch.FloatTensor([affinity]),
                    target=torch.LongTensor([target])
                )
            except:
                    print("unable to process: ", smi)

            data_list.append(data)

        return data_list

    def process(self):

        df_train = pd.read_csv(self.raw_paths[0])
        df_val = pd.read_csv(self.raw_paths[1])
        df_test = pd.read_csv(self.raw_paths[2])
        df = pd.concat([df_train, df_val, df_test])

        # df_train = pd.read_csv(self.raw_paths[0])
        # df_test = pd.read_csv(self.raw_paths[1])
        # df = pd.concat([df_train, df_test])

        smiles = df['smiles'].unique()
        graph_dict = dict()
        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            g = self.mol2graph(mol)
            graph_dict[smile] = g

        train_list = self.process_data(self.raw_paths[0], graph_dict)
        val_list = self.process_data(self.raw_paths[1], graph_dict)
        test_list = self.process_data(self.raw_paths[2], graph_dict)

        if self.pre_filter is not None:
            train_list = [train for train in train_list if self.pre_filter(train)]
            val_list = [val for val in val_list if self.pre_filter(val)]
            test_list = [test for test in test_list if self.pre_filter(test)]

        if self.pre_transform is not None:
            train_list = [self.pre_transform(train) for train in train_list]
            val_list = [self.pre_transform(val) for val in val_list]
            test_list = [self.pre_transform(test) for test in test_list]

        print('Graph construction done. Saving to file.')

        data, slices = self.collate(train_list)
        # save preprocessed train data:
        torch.save((data, slices), self.processed_paths[0])

        data, slices = self.collate(test_list)
        # save preprocessed val data:
        torch.save((data, slices), self.processed_paths[1])
        
        data, slices = self.collate(test_list)
        # save preprocessed test data:
        torch.save((data, slices), self.processed_paths[2])

    def get_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            # 5 more
            h_t.append(d['ExplicitValence'])
            h_t.append(d['FormalCharge'])
            h_t.append(d['ImplicitValence'])
            h_t.append(d['NumExplicitHs'])
            h_t.append(d['NumRadicalElectrons'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])

        return node_attr

    def get_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]

            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t

        if len(e) == 0:
            return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def mol2graph(self, mol):
        if mol is None:
            return None
        feats = chem_feature_factory.GetFeaturesForMol(mol)
        g = nx.DiGraph()

        # Create nodes
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),

                       # 5 more node features
                       ExplicitValence=atom_i.GetExplicitValence(),
                       FormalCharge=atom_i.GetFormalCharge(),
                       ImplicitValence=atom_i.GetImplicitValence(),
                       NumExplicitHs=atom_i.GetNumExplicitHs(),
                       NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['acceptor'] = 1

        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               # 1 more edge features 2 dim
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )

        node_attr = self.get_nodes(g)
        edge_index, edge_attr = self.get_edges(g)

        return node_attr, edge_index, edge_attr

class ClassificationPreprocessor(InMemoryDataset):
    def __init__(self, root, types = 'train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if types == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif types == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif types == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_val.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_val.pt', 'processed_data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process_data(self, data_path, graph_dict):
        df = pd.read_csv(data_path)

        data_list = []
        delete_list = []
        for i, row in df.iterrows():
            smi = row['smiles']
            sequence = row['sequence']
            affinity = row['affinity']
            activity = row['activity']

            if graph_dict.get(smi) == None:
                print("Unable to process: ", smi)
                delete_list.append(i)
                continue

            x, edge_index = graph_dict[smi]

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len- len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                    x=torch.FloatTensor(x),
                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                    target=torch.LongTensor([target]),
                    activity=torch.LongTensor([activity]),
                    affinity=torch.FloatTensor([affinity])
                )

            data_list.append(data)

        if len(delete_list) > 0:
            df = df.drop(delete_list, axis=0, inplace=False)
            df.to_csv(data_path, index=False)

        return data_list

    def process(self):
        df_train = pd.read_csv(self.raw_paths[0])
        df_val = pd.read_csv(self.raw_paths[1])
        df_test = pd.read_csv(self.raw_paths[2])
        df = pd.concat([df_train, df_val, df_test])
        smiles = df['smiles']

        graph_dict = dict()
        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            if mol == None:
                print("Unable to process: ", smile)
                continue
            graph_dict[smile] = mol_to_graph(mol)

        train_list = self.process_data(self.raw_paths[0], graph_dict)
        val_list = self.process_data(self.raw_paths[1], graph_dict)
        test_list = self.process_data(self.raw_paths[2], graph_dict)

        if self.pre_filter is not None:
            train_list = [train for train in train_list if self.pre_filter(train)]
            val_list = [val for val in val_list if self.pre_filter(val)]
            test_list = [test for test in test_list if self.pre_filter(test)]

        if self.pre_transform is not None:
            train_list = [self.pre_transform(train) for train in train_list]
            val_list = [self.pre_transform(val) for val in val_list]
            test_list = [self.pre_transform(test) for test in test_list]

        print('Graph construction done. Saving to file.')

        # save preprocessed train data:
        data, slices = self.collate(train_list)
        torch.save((data, slices), self.processed_paths[0])

        # save preprocessed val data:
        data, slices = self.collate(val_list)
        torch.save((data, slices), self.processed_paths[1])

        # save preprocessed test data:
        data, slices = self.collate(test_list)
        torch.save((data, slices), self.processed_paths[2])


def preprocess(path, task, split = 'stratified', sizes = [0.6, 0.2, 0.2], scale = True):
    data_split_train_val_test(path, split, sizes, scale)
    if task == 'regression':
        RegressionPreprocessor(path)
    elif task == 'classification':
        ClassificationPreprocessor(path)