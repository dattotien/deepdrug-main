import os
import warnings
import pickle as pkl
from copy import deepcopy
from itertools import product
from functools import reduce
from collections import defaultdict, Counter, OrderedDict

import numpy as np
import pandas as pd
import random, time, datetime, json
import matplotlib.pyplot as plt
import argparse
from time import sleep

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse, to_undirected, remove_self_loops, degree
from torch_geometric.utils import add_self_loops as pyg_add_self_loops
from torch_sparse import coalesce

from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import OneHotEncoder

from molGraphConvFeaturizer import MolGraphConvFeaturizer as user_MolGraphConvFeaturizer

warnings.filterwarnings('ignore')


# --------------------------
# Helpers
# --------------------------
def read_df_or_parquet(file, save_parquet=False, **args):
    parquet_file = file.replace('.csv', '.parquet')
    if os.path.exists(parquet_file):
        print('Parquet format file was used:', parquet_file)
        return pd.read_parquet(parquet_file)
    elif os.path.exists(file):
        tmp_df = pd.read_csv(file, **args)
        print('Saving parquet file to load quickly next time:', parquet_file)
        if save_parquet:
            tmp_df.to_parquet(parquet_file)
        return tmp_df
    else:
        return None


def graph_to_undirected(data):
    edge_index, edge_attr, num_nodes = data.edge_index, data.edge_attr, data.num_nodes
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    if edge_attr is not None:
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes, op="mean")
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data


def graph_add_degree(data, degree_type='both'):
    assert degree_type in ['indegree', 'outdegree', 'both']
    deg1 = degree(data.edge_index[0], data.num_nodes).reshape(-1, 1)
    deg2 = degree(data.edge_index[1], data.num_nodes).reshape(-1, 1)
    data.x = torch.cat([data.x, deg1, deg2], dim=1)
    return data


def add_self_loops(edge_index, edge_weight=None, fill_value=0., num_nodes=None):
    edge_index, edge_weight = pyg_add_self_loops(
        edge_index, edge_attr=edge_weight, fill_value=fill_value, num_nodes=num_nodes
    )
    return edge_index, edge_weight


# --------------------------
# Entry Dataset
# --------------------------
class EntryDataset(InMemoryDataset):
    def __init__(self, root_folder, transform=None, pre_transform=None,
                 pre_filter=None, filename='data', inmemory=False):
        if not inmemory:
            os.makedirs(os.path.join(root_folder, 'processed'), exist_ok=True)
        super().__init__(root_folder, transform, pre_transform, pre_filter)
        self.inmemory = inmemory
        self.filename = filename

        if os.path.exists(self.processed_paths[0]):
            print('Loading processed data...')
            tmp = torch.load(self.processed_paths[0])
            if len(tmp) == 3:
                self.data, self.slices, self.entryIDs = tmp
            elif len(tmp) == 4:
                self.data, self.slices, self.entryIDs, self.unpIDs = tmp
        else:
            print('Processed file not found.')

    @property
    def processed_file_names(self):
        return f'{self.filename}.pt'

    def add_self_loops_all(self):
        print('Adding self-loops for each graph...')
        data_list = [
            Data(
                x=data.x,
                edge_index=add_self_loops(data.edge_index, edge_weight=data.edge_attr,
                                           fill_value=0, num_nodes=data.num_nodes)[0],
                edge_attr=data.edge_attr
            ) for data in self
        ]
        self.data, self.slices = self.collate(data_list)
        self._data_list = data_list

    def to_undirected_all(self):
        print('Converting each graph to undirected...')
        data_list = [graph_to_undirected(data) for data in self]
        self.data, self.slices = self.collate(data_list)
        self._data_list = data_list

    def add_node_degree_all(self):
        print('Adding degree features for each graph...')
        data_list = [graph_add_degree(data) for data in self]
        self.data, self.slices = self.collate(data_list)
        self._data_list = data_list

    # drug_process & protein_process giữ nguyên, chỉ cần convert numpy -> tensor đúng dtype
    # ...


# --------------------------
# Sequence Dataset
# --------------------------
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ?"
seq_dict = {v: i for i, v in enumerate(seq_voc)}

smile_dict = { ... }  # giữ nguyên từ code của bạn

def trans_seqs(x, seq_dict, max_seq_len=200, upper=True):
    if upper:
        x = x.upper()
    temp = [i if i in seq_dict else '?' for i in x]
    temp = (temp + ['?'] * max_seq_len)[:max_seq_len]
    temp = [seq_dict[i] for i in temp]
    return temp


class SeqDataset(data.Dataset):
    def __init__(self, data_file, max_len=200, data_type='protein', onehot=False):
        word_dict = seq_dict if data_type == 'protein' else smile_dict
        upper = data_type == 'protein'
        self.onehot = onehot

        df = read_df_or_parquet(data_file, save_parquet=False)
        df.iloc[:, 0] = df.iloc[:, 0].astype(str)
        df['entry'] = df.iloc[:, 0].apply(trans_seqs, max_seq_len=max_len, seq_dict=word_dict, upper=upper)
        self.df = df
        self.entry_dict = df['entry'].to_dict()
        self.entryIDs = df.index.values
        self.num_samples = df.shape[0]

        if onehot:
            self.encoder = OneHotEncoder(sparse=False).fit(np.arange(len(word_dict)).reshape(-1, 1))
            self.org_entry_dict = deepcopy(self.entry_dict)
            self.entry_dict = {}

    def __getitem__(self, idx):
        entryID = self.entryIDs[idx]
        if self.onehot and entryID not in self.entry_dict:
            self.entry_dict[entryID] = self.encoder.transform(np.reshape(self.org_entry_dict[entryID], (-1, 1))).transpose()
        data = self.entry_dict[entryID] if self.onehot else self.entry_dict[entryID]
        return torch.Tensor(data).float() if self.onehot else torch.Tensor(data).long()

    def __len__(self):
        return self.num_samples


# --------------------------
# MultiEmbedDataset_v1 & PairedDataset_v1
# --------------------------
class MultiEmbedDataset_v1(data.Dataset):
    def __init__(self, *args):
        self.datasets = args
        self.num_samples = len(self.datasets[0])
        self.entryIDs = self.datasets[0].entryIDs
        dataset2_entryIDs = self.datasets[1].entryIDs.tolist()
        if not np.all(self.entryIDs == dataset2_entryIDs):
            self.datasets2_idx = {idx: dataset2_entryIDs.index(entry) for idx, entry in enumerate(self.entryIDs)}
        else:
            self.datasets2_idx = {idx: idx for idx in range(self.num_samples)}

    def __getitem__(self, idx):
        return [self.datasets[0][idx], self.datasets[1][self.datasets2_idx[idx]]]

    def __len__(self):
        return self.num_samples


class PairedDataset_v1(data.Dataset):
    def __init__(self, entry1, entry2, entry_pairs, pair_labels):
        self.entry1 = entry1
        self.entry2 = entry2
        self.entry_pairs = entry_pairs
        self.pair_labels = pair_labels
        self.entry1_ids = self.entry1.entryIDs.tolist()
        self.entry2_ids = self.entry2.entryIDs.tolist()
        self.num_samples = pair_labels.shape[0]

    def __getitem__(self, idx):
        tmp1, tmp2 = self.entry_pairs[idx]
        return ((self.entry1[self.entry1_ids.index(tmp1)],
                 self.entry2[self.entry2_ids.index(tmp2)]),
                self.pair_labels[idx])

    def __len__(self):
        return self.num_samples


# --------------------------
# Lightning DataModule
# --------------------------
class DeepDrug_Dataset(LightningDataModule):
    def __init__(self, entry1_data_folder, entry2_data_folder, entry_pairs_file,
                 pair_labels_file, cv_file=None, cv_fold=0, batch_size=128,
                 task_type='binary', y_transfrom_func=None, category=None,
                 entry1_seq_file=None, entry2_seq_file=None):
        super().__init__()

        self.task_type = task_type
        self.entry1_data_folder = entry1_data_folder
        self.entry2_data_folder = entry2_data_folder
        self.entry_pairs_file = entry_pairs_file
        self.pair_labels_file = pair_labels_file
        self.cv_file = cv_file
        self.cv_fold = cv_fold
        self.batch_size = batch_size
        self.y_transfrom_func = y_transfrom_func
        self.category = category

        self.entry1_seq_len = 200
        self.entry2_seq_len = 200 if category == 'DDI' else 1000
        self.entry2_type = 'drug' if category == 'DDI' else 'protein'

        self.entry1_multi_embed = True
        self.entry2_multi_embed = True
        self.entry1_seq_file = entry1_seq_file
        self.entry2_seq_file = entry2_seq_file

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.my_prepare_data()

    def train_dataloader(self):
        train_pairs = self.entry_pairs[self.train_indexs]
        train_labels = self.pair_labels[self.train_indexs]
        train_split = PairedDataset_v1(self.entry1_dataset, self.entry2_dataset, train_pairs, train_labels)
        return DataLoader(train_split, num_workers=8, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        val_split = PairedDataset_v1(self.entry1_dataset, self.entry2_dataset,
                                     self.entry_pairs[self.valid_indexs], self.pair_labels[self.valid_indexs])
        return DataLoader(val_split, num_workers=8, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        test_split = PairedDataset_v1(self.entry1_dataset, self.entry2_dataset,
                                      self.entry_pairs[self.test_indexs], self.pair_labels[self.test_indexs])
        return DataLoader(test_split, num_workers=8, shuffle=False, batch_size=self.batch_size)
