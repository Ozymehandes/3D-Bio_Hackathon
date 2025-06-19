import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist, pdist


def load_peptide_data(data_csv="DB/NesDB_all_CRM1_with_peptides_train.csv", include_nesdoubt=True, include_nodoubt=True,
                      max_peptide_len=22):
    """
    Loads all negative and positive peptide data
    :param data_csv: The path to the NesDB csv file
    :param include_nesdoubt: Whether to include doubt data (default=True)
    :param include_nodoubt: Whether to include no doubt data (default=True)
    :param max_peptide_len: Maximal allowed length for a peptide (default=22)
    :return: Three lists: [positive peptides], [negative peptides], [is doubt labels]
    """
    df = pd.read_csv(data_csv).dropna(subset=['Peptide_sequence', 'Negative_sequence', 'Sequence'])
    if not include_nesdoubt:
        df = df[df['is_NesDB_doubt'] != True].reset_index(drop=True)
    if not include_nodoubt:
        df = df[df['is_NesDB_doubt'] != False].reset_index(drop=True)

    pos_pep = []
    neg_pep = []
    data_doubt = []
    counter = 0

    for index, row in df.iterrows():
        pep = row['Peptide_sequence']
        neg = row['Negative_sequence']
        if len(pep) <= max_peptide_len and pep != '' and len(neg) <= max_peptide_len and neg != '':
            pos_pep.append((f"{counter}", pep))
            neg_pep.append((f"{counter}", neg))
            data_doubt.append(row['is_NesDB_doubt'])
            counter += 1

    return pos_pep, neg_pep, data_doubt


def pep_train_test_split(pos_pep, neg_pep, doubt_list, test_size=0.1, seed=42):
    """
    Splits the peptide data into training and testing sets
    :param pos_pep: positive peptides ESM-2 sequence embeddings
    :param neg_pep: negative peptides ESM-2 sequence embeddings
    :param doubt_list: doubt labels list
    :param test_size: Desired test size
    :param seed: Random seed
    :return: pos_train, neg_train, doubt_train, pos_test, neg_test, doubt_test
    """
    assert len(pos_pep) == len(neg_pep) == len(doubt_list)
    pos_pep, neg_pep, doubt_list = np.array(pos_pep), np.array(neg_pep), np.array(doubt_list)

    # Split to train and test
    train_idx, test_idx = train_test_split(range(len(pos_pep)), test_size=test_size, random_state=seed)
    pos_train, pos_test = pos_pep[train_idx], pos_pep[test_idx]
    neg_train, neg_test = neg_pep[train_idx], neg_pep[test_idx]
    doubt_train, doubt_test = doubt_list[train_idx], doubt_list[test_idx]

    return pos_train, neg_train, doubt_train, pos_test, neg_test, doubt_test


def get_peptide_distances(pos_neg_test_peptides, pos_train_peptides, reduce_func=np.mean):
    """
    Returns for each peptide in 'pos_neg_test_peptides' the mean Euclidean distance from all of the peptides in 'pos_train_peptides'
    :param pos_neg_test_peptides: ESM-2 sequence embeddings of test peptides (negative or positives)
    :param pos_train_peptides: ESM-2 sequence embeddings of train peptides (positives)
    :param reduce_func: How to reduce the pair distances (mean/median...)
    :return: numpy array with distance for eact peptide in 'pos_neg_test_peptides'
    """
    # Get all of the Euclidean distances of the ESM embeddings for each test-train pair
    distances = cdist(np.array(pos_neg_test_peptides), np.array(pos_train_peptides), metric="euclidean")

    # For each test peptide get the mean/median/max... etc. from all the positive training peptides
    reduced_distances = reduce_func(distances, axis=-1)

    return reduced_distances





