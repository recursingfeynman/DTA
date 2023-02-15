from papyrus_scripts.download import download_papyrus
from papyrus_scripts.preprocess import keep_quality, keep_organism, keep_accession, keep_type, consume_chunks
from papyrus_scripts.reader import read_papyrus,read_protein_set

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
def get_data(uniprot_ids, vocab, activity_threshold = 6.25):
    download_papyrus(version='latest', structures=False, descriptors = None)

    sample_data = read_papyrus(is3d=False, chunksize = 100_000, source_path=None)
    protein_data = protein_data = read_protein_set(source_path=None)
    
    # uniprot_ids = ["P08183","P43245", "P06795","P21447", "Q9UNQ0", "Q80W57", "Q7TMS5"]
    
    f = keep_accession(sample_data, uniprot_ids)
    f = keep_quality(data = f, min_quality='medium')
    # f = keep_organism(data= f, protein_data=protein_data, organism=['Human', 'Rat', 'Mouse'], generic_regex=True)
    # f = keep_type(data = f, activity_types=['Ki', 'KD', 'IC50'])
    data = consume_chunks(f, total = 13, progress = True)
    data.head()

    # activity_threshold = 6.25

    # vocab = {
    #     "activity" : ["Inactive", "Active"],
    #     "proteins" : {
    #         "P08183" : "PGP",
    #         "P43245" : "PGP",
    #         "P06795" : "PGP",
    #         "P21447" : "PGP",
    #         "Q9UNQ0" : "BCRP",
    #         "Q80W57" : "BCRP",
    #         "Q7TMS5" : "BCRP"
    #     }
    # }

    def get_names(activity, pid):
        activity = int(activity > activity_threshold)
        return vocab['activity'][activity] + "_" + vocab['proteins'][pid]

    pdata = data[['SMILES', 'accession', 'pchembl_value_Mean']]

    for uni in uniprot_ids:
        for idx, pid in enumerate(protein_data.target_id.values):
            if uni in pid:
                pdata.loc[pdata['accession'] == uni, 'sequence'] = protein_data.iloc[idx]['Sequence']

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