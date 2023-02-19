import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
import numpy as np
import torch
fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
          

PROTEIN = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
	"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
	"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
	"U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, 
	"Z": 25 }

LIGAND = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 
    'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 
    'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 
    'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 
    'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP, 
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, 
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2, 
    'other']

DEGREE = [0,1,2,3,4,5,6,7,8,9,10]
VALENCE = [0,1,2,3,4,5,6,7,8,9,10]
NHYDROGENS = [0,1,2,3,4,5,6,7,8,9,10]

class classificationGraph(object):

    def __call__(self, mol):
        features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom)
            features.append(feature/np.sum(feature))

        # Compute bond(edge) order
        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

        # If there is no bonds -> return [[0, 0]]
        if len(edges) == 0:
            return features, [[0, 0]]

        # Construct graph from bonds(edges)
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])
        
        return [features, edge_index]

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def atom_features(self, atom):
        encoding = []
        encoding += self.one_of_k_encoding_unk(atom.GetSymbol(), LIGAND)
        encoding += self.one_of_k_encoding(atom.GetDegree(), DEGREE) 
        encoding += self.one_of_k_encoding_unk(atom.GetTotalNumHs(), NHYDROGENS) 
        encoding += self.one_of_k_encoding_unk(atom.GetImplicitValence(), VALENCE) 
        encoding += self.one_of_k_encoding_unk(atom.GetHybridization(), HYBRIDIZATION) 
        encoding += [atom.GetIsAromatic()]

        try:
            encoding += self.one_of_k_encoding_unk(
                        atom.GetProp('_CIPCode'),
                        ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]
        
        return np.array(encoding)

class regressionGraph(object):

    def __call__(self, mol):
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

def seqs2int(sequence):
    return [PROTEIN[s] for s in sequence]

class ClassificationPreprocessor(InMemoryDataset):
    def __init__(self, root, pre_filter = None, pre_transform = None, transform = None):
        super().__init__(root, pre_filter, pre_transform, transform)

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_val.csv', 'data_test.csv']
    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_val.pt', 'processed_data_test.pt']

    def _process_graphs(self, path, graph_dict):
        raw_data = pd.read_csv(path)

        processed_graphs = []
        delete_list = []

        for i, row in raw_data.iterrows():
            # Get smiles, sequence and target values corresponding one complex
            smi = row['smiles']
            sequence = row['sequence']
            affinity = row['affinity']
            activity = row['activity']

            # If unable to process current smile -> append to `delete_list` and drop later
            if graph_dict.get(smi) is None:
                print("Unable to process: ", smi)
                delete_list.append(i)
                continue

            node_feature, edge_index = graph_dict[smi]

            # Compute protein features
            sequence = seqs2int(sequence)
            seq_len = 1200 # Fix target length
            
            # If length or protein sequence less than fixed value -> pad with zeros
            if len(sequence) < seq_len:
                sequence = np.pad(sequence, (0, seq_len - len(sequence)))
            else:
                sequence = sequence[:seq_len]

            # Construct graph
            graph = Data(
                    node_feature=torch.FloatTensor(np.vstack(node_feature)),
                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                    sequence=torch.LongTensor(np.vstack([sequence])),
                    activity=torch.LongTensor([activity]),
                    affinity=torch.FloatTensor([affinity])
                )

            # Append single graph to data_list
            processed_graphs.append(graph)

        # Drop incorrect smiles
        if len(delete_list) > 0:
            raw_data = raw_data.drop(delete_list, axis=0, inplace=False)
            raw_data.to_csv(path, index=False)

        # Return processed complex as graph
        return processed_graphs


    def process(self):
        train = pd.read_csv(self.raw_paths[0])
        valid = pd.read_csv(self.raw_paths[1])
        test = pd.read_csv(self.raw_paths[2])

        data = pd.concat([train, valid, test], axis = 0)
        del train, valid, test

        smiles = data['smiles']

        graph_dict = dict()
        for index, smi in enumerate(tqdm(smiles, total = len(smiles))):
            mol = Chem.MolFromSmiles(smi)
            
            if mol is not None:
                graph_dict[smi] = classificationGraph()(mol)
            else:
                print(f"Unable to process {smi} [{index}]")

        train_list = self._process_graphs(self.raw_paths[0], graph_dict)
        val_list = self._process_graphs(self.raw_paths[1], graph_dict)
        test_list = self._process_graphs(self.raw_paths[2], graph_dict)

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

class RegressionPreprocessor(InMemoryDataset):
    def __init__(self, root, pre_filter = None, pre_transform = None, transform = None):
        super().__init__(root, pre_filter, pre_transform, transform)

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_val.csv', 'data_test.csv']
    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_val.pt', 'processed_data_test.pt']

    def _process_graphs(self, path, graph_dict):
        raw_data = pd.read_csv(path)

        processed_graphs = []
        delete_list = []

        for i, row in raw_data.iterrows():
            # Get smiles, sequence and target values corresponding one complex
            smi = row['smiles']
            sequence = row['sequence']
            affinity = row['affinity']
            activity = row['activity']

            # If unable to process current smile -> append to `delete_list` and drop later
            if graph_dict.get(smi) is None:
                print("Unable to process: ", smi)
                delete_list.append(i)
                continue

            node_feature, edge_index, edge_attr = graph_dict[smi]

            # Compute protein features
            sequence = seqs2int(sequence)
            seq_len = 1200 # Fix target length
            
            # If length or protein sequence less than fixed value -> pad with zeros
            if len(sequence) < seq_len:
                sequence = np.pad(sequence, (0, seq_len - len(sequence)))
            else:
                sequence = sequence[:seq_len]

            # Construct graph
            graph = Data(
                    node_feature=torch.FloatTensor(np.vstack(node_feature)),
                    edge_index=torch.LongTensor(edge_index),
                    edge_attr=torch.FloatTensor(edge_attr),
                    sequence=torch.LongTensor(np.vstack([sequence)),
                    activity=torch.LongTensor([activity]),
                    affinity=torch.FloatTensor([affinity])
                )

            # Append single graph to data_list
            processed_graphs.append(graph)

        # Drop incorrect smiles
        if len(delete_list) > 0:
            raw_data = raw_data.drop(delete_list, axis=0, inplace=False)
            raw_data.to_csv(path, index=False)

        # Return processed complex as graph
        return processed_graphs


    def process(self):
        train = pd.read_csv(self.raw_paths[0])
        valid = pd.read_csv(self.raw_paths[1])
        test = pd.read_csv(self.raw_paths[2])

        data = pd.concat([train, valid, test], axis = 0)
        del train, valid, test

        smiles = data['smiles']

        graph_dict = dict()
        for index, smi in enumerate(tqdm(smiles, total = len(smiles))):
            mol = Chem.MolFromSmiles(smi)
            
            if mol is not None:
                graph_dict[smi] = regressionGraph()(mol)
            else:
                print(f"Unable to process {smi} [{index}]")

        train_list = self._process_graphs(self.raw_paths[0], graph_dict)
        val_list = self._process_graphs(self.raw_paths[1], graph_dict)
        test_list = self._process_graphs(self.raw_paths[2], graph_dict)

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

def prepare_data(path, task):
    if task == 'classification':
        ClassificationPreprocessor(path)
    elif task == 'regression':
        RegressionPreprocessor(path)
    else:
        raise ValueError("Task not supported. Available tasks: `classification`, `regression`.")
