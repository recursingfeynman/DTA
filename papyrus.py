from papyrus_scripts.download import download_papyrus
from papyrus_scripts.preprocess import keep_quality, keep_organism, keep_accession, keep_type, keep_protein_class,consume_chunks
from papyrus_scripts.reader import read_papyrus,read_protein_set

import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
def get_data(vocab, activity_threshold = 6.25):
    download_papyrus(version='latest', structures=False, descriptors = None)

    sample_data = read_papyrus(is3d=False, chunksize = 100_000, source_path=None)
    protein_data = read_protein_set(source_path=None)
    
    uniprot_ids = list(vocab['proteins'].keys())
    f = keep_accession(sample_data, uniprot_ids)
    # f = keep_type(data = f, activity_types=['Ki', 'KD', 'IC50'])
    data = consume_chunks(f, total = 13, progress = True)

    def get_names(activity, pid):
        activity = int(activity > activity_threshold)
        return vocab['activity'][activity] + "_" + vocab['proteins'][pid]

    pdata = data[['SMILES', 'accession', 'pchembl_value_Mean']]

    for uni in uniprot_ids:
        for idx, pid in enumerate(protein_data.target_id.values):
            if uni in pid:
                pdata.loc[pdata['accession'] == uni, 'sequence'] = protein_data.iloc[idx]['Sequence']

    pdata = pdata.drop_duplicates(subset = ['SMILES'])
    
    counts_df = pd.DataFrame({
        "Proteins" : list(pdata['accession'].value_counts().index),
        "Counts" : list(pdata['accession'].value_counts().values)
        })

    counts_df['Proteins'] = counts_df['Proteins'].apply(lambda x: vocab['proteins'][x])
    counts_df = counts_df.groupby("Proteins", as_index = False).sum()
    
    print("Counts: ")
    [print((x, y)) for x, y in zip(counts_df.Proteins.values, counts_df.Counts.values)]

    pdata['activity'] = pdata['pchembl_value_Mean'].apply(lambda x: 1 if x >= activity_threshold else 0)
    pdata = pdata.rename(columns = {"SMILES" : 'smiles', "pchembl_value_Mean" : "affinity", "accession" : "protein"})[['smiles', 'sequence', 'activity', 'affinity', 'protein']].reset_index(drop = True)
    pdata['protein'] = pdata['protein'].apply(lambda x: vocab['proteins'][x])
    
    print("Molecules: {} \tFeatures ({}): {}".format(*pdata.shape, list(pdata.columns)))
    
    os.makedirs("./data/papyrus/raw/", exist_ok = True)

    pdata.to_csv("data/papurys/raw/data.csv", index = False)
    print("Saved as data/raw/data.csv")

    return pdata, protein_data

# class_names = {'l3': 'SLC superfamily of solute carriers'}
def get_specific_class(class_names, act_threshold, inact_threshold, version = "latest", drop_duplicates = False, multiclass = False):

    download_papyrus(version=version, structures=False, descriptors = None)

    sample_data = read_papyrus(is3d=False, chunksize = 100_000, source_path=None)
    protein_data = read_protein_set(source_path=None)
    
    f = keep_protein_class(data=sample_data, protein_data=protein_data, classes=class_names)
    data = consume_chunks(f, total = 13, progress = True)
    pdata = data[['SMILES', 'accession', 'pchembl_value_Mean']]

    if drop_duplicates:
        pdata = pdata.drop_duplicates(subset = ['SMILES'])

    for accession in np.unique(data['accession'].values):
        for idx, pid in enumerate(protein_data.target_id.values):
            if accession in pid:
                pdata.loc[pdata['accession'] == accession, 'sequence'] = protein_data.iloc[idx]['Sequence']

    def encode(x, act_threshold, inact_threshold):
        if x >= act_threshold:
            return 1
        elif x <= inact_threshold:
            return 0
        else:
            return np.nan
    pdata['activity'] = pdata['pchembl_value_Mean'].apply(lambda x: encode(x, act_threshold, inact_threshold))
    pdata = pdata.dropna(subset = ['activity'])
   
    pdata['count'] = pdata.groupby('activity', as_index = false)['accession'].transform("count")
    pdata = pdata.loc[pdata['count'] > 1]
    pdata = pdata.drop("count", axis = 1)

    if multiclass:
        le = LabelEncoder()
        pdata['activity'] = pdata['accession'] + "-" + pdata['activity'].astype(str)
        pdata['activity'] = le.fit_transform(pdata['activity'].values)
    
    pdata['activity'] = pdata['activity'].astype(int)

    pdata = pdata.rename(columns = {"SMILES" : 'smiles', "pchembl_value_Mean" : "affinity", "accession" : "protein"})[['smiles', 'sequence', 'activity', 'affinity', 'protein']].reset_index(drop = True)
    print("Molecules: {} \tFeatures ({}): {}".format(*pdata.shape, list(pdata.columns)))
    os.makedirs("./data/papyrus/raw/", exist_ok = True)
    pdata.to_csv("data/papyrus/raw/data.csv", index = False)
    print("Saved as data/papyrus/raw/data.csv")

    return pdata, protein_data

    
    
    
    
