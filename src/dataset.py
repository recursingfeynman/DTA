import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdmolops

def onehot(x, allowable_set, unique):
    if x not in allowable_set:
        if unique:
            x = allowable_set[-1]
        else:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    
    return list(map(lambda s: x == s, allowable_set))

HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP, 
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, 
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2, 
    'other']

AMINOACIDS = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
	"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
	"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
	"U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, 
	"Z": 25 }

# ATOMS = {
#     'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Si': 6, 
#     'P': 7, 'Cl': 8, 'Br': 9, 'Mg' : 10, 'Na': 11, 'Ca': 12, 
#     'Fe': 13, 'As': 14, 'Al': 15, 'I': 16, 'B': 17, 'V': 18, 
#     'K': 19, 'Tl': 20, 'Yb': 21,'Sb': 22, 'Sn': 23, 'Ag': 24, 
#     'Pd': 25, 'Co': 26, 'Se': 27, 'Ti': 28, 'Zn': 29, 'H': 30,
#     'Li': 31, 'Ge': 32, 'Cu': 33, 'Au': 34, 'Ni': 35, 'Cd': 36, 
#     'In': 37, 'Mn': 38, 'Zr': 39,'Cr': 40, 'Pt': 41, 'Hg': 42, 
#     'Pb': 43, 'Te': 44}

ATOMS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 
    'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 
    'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 
    'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 
    'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Te', 'Unknown']




class MoleculeDataset(Dataset):
    def __init__(self, root, transform = None, pre_transform = None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return 'raw_data.csv'

    @property
    def processed_file_names(self):
        return 'processed_data.csv'

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        
        for index, sample in tqdm(self.data.iterrows(), total = len(self.data)):
            smiles = sample['smiles']
            sequence = sample['sequence']
            affinity = sample['affinity']
            activity = sample['activity']

            molecule = Chem.MolFromSmiles(smiles)
            
            # Get Node feature
            node_features = self._get_node_feature(molecule)
            # Get edge feature
            edge_attr = self._get_edge_feature(molecule)
            # Get adjacency matrix
            edge_index = self._get_adjacency_matrix(molecule)
            # Get encoded sequence
            encoded_sequence = self._encode_sequence(sequence)

            data = Data(
                node_feature = node_features,
                edge_index = edge_index,
                edge_attr = edge_attr,
                sequence = encoded_sequence,
                affinity = affinity,
                activity = activity,
                smiles = smiles,
            )

            torch.save(data, os.path.join(self.processed_dir, f"data_{index}.pt"))

    def _get_node_feature(self, mol):

        nodes_feature = []

        for atom in mol.GetAtoms():
            node_feature = []
            # Compute atom features
            node_feature.append(onehot(atom.GetSymbol(), ATOMS, True))
            node_feature.append(atom.GetAtomicNum())
            node_feature.append(atom.GetTotalNumHs())
            node_feature.append(atom.GetExplicitValence())
            node_feature.append(atom.GetImplicitValence())
            node_feature.append(atom.GetNumRadicalElectrons())
            node_feature.append(atom.GetNumExplicitHs())
            node_feature.append(atom.GetDegree())
            node_feature.append(atom.GetFormalCharge())
            node_feature.append(onehot(atom.GetHybridization(), HYBRIDIZATION, True))
            node_feature.append(atom.GetIsAromatic())
            node_feature.append(atom.IsInRing())

            nodes_feature.append(np.hstack(node_feature))

        nodes_feature = np.asarray(nodes_feature)

        return torch.tensor(nodes_feature, dtype = torch.float32)

    def _get_edge_feature(self, mol):

        edges_feature = []

        for bond in mol.GetBonds():
            edge_feature = []

            edge_feature.append(bond.GetBondTypeAsDouble())
            edge_feature.append(bond.IsInRing())

            edges_feature.append(edge_feature)

        edges_feature = np.asarray(edges_feature)

        return torch.tensor(edges_feature, dtype = torch.float32)

    def _get_adjacency_matrix(self, mol):
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        row, col = np.where(adj_matrix)
        coo = np.array(list(zip(row, col)))
        coo = np.transpose(coo)
        return torch.tensor(coo, dtype = torch.long)

    def _encode_sequence(self, seq):
        encoded_sequence = [AMINOACIDS[x] for x in seq]
        max_len = 1200
        if len(encoded_sequence) < max_len:
            encoded_sequence = np.pad(encoded_sequence, (0, max_len - len(encoded_sequence)))
        else:
            encoded_sequence = encoded_sequence[:1200]

        return torch.tensor(encoded_sequence, dtype = torch.long).unsqueeze(0)

    def len(self):
        return len(self.data)

    def get(self, index):
        data = torch.load(os.path.join(self.processed_dir, f"data_{index}.pt"))
        return data
