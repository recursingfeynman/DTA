from papyrus_scripts.download import download_papyrus
from papyrus_scripts.preprocess import keep_quality, keep_organism, keep_accession, keep_type, consume_chunks
from papyrus_scripts.reader import read_papyrus,read_protein_set

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
def get_data(vocab, activity_threshold = 6.25):
    download_papyrus(version='latest', structures=False, descriptors = None)

    sample_data = read_papyrus(is3d=False, chunksize = 100_000, source_path=None)
    protein_data = read_protein_set(source_path=None)
    
    uniprot_ids = list(vocab['proteins'].keys())
    
    f = keep_accession(sample_data, uniprot_ids)
    f = keep_quality(data = f, min_quality='medium')
    # f = keep_organism(data= f, protein_data=protein_data, organism=['Human', 'Rat', 'Mouse'], generic_regex=True)
    # f = keep_type(data = f, activity_types=['Ki', 'KD', 'IC50'])
    data = consume_chunks(f, total = 13, progress = True)
    data.head()

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

    le = LabelEncoder()
    pdata['activity'] = pdata[['pchembl_value_Mean', 'accession']].apply(lambda x: get_names(x[0], x[1]), axis = 1)
    pdata['activity'] = le.fit_transform(pdata['activity'])
    pdata['activity'].value_counts()
    pdata = pdata.rename(columns = {"SMILES" : 'smiles', "pchembl_value_Mean" : "affinity"})[['smiles', 'sequence', 'activity', 'affinity']].reset_index(drop = True)

    meta = [(x, str(le.inverse_transform([x])[0])) for x in sorted(pdata['activity'].unique())]


    print("Molecules: {} \tFeatures ({}): {}".format(*pdata.shape, list(pdata.columns)))
    print(f"Encodings (threshold = {activity_threshold}):")
    [print(x) for x in meta];

    pdata.to_csv("data/raw/data.csv", index = False)
    print("Saved to data/raw/data.csv")