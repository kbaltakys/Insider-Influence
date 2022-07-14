import os
from typing import Dict, Set, Tuple, List
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
from private import DATASET_PATH, DATASPLIT_PATH, NODE_SECURITY_DISTANCE, FAMNODE_TO_ISIN
#

def load_w2v_feature(file: str, max_idx: int = 0) -> np.ndarray:
    with open(file, "rb") as f:
        nu = 0
        for line in tqdm(f, desc='Loading embedding', leave=False):
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                # Change to numpy array
                feature = [[0.] * d for i in range(max(n, max_idx + 1))]
                continue
            index = int(content[0])
            # Done when node id > max_idx or node # indicated in embedding
            while len(feature) <= index:
                raise Exception(
                    f'Embedding contains too high index({index}) for a node, when max should be n={n} and max_idx={max_idx}')
                feature.append([0.] * d)
            for i, x in enumerate(content[1:]):
                feature[index][i] = float(x)
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)


def anonymize_data(dataset_path: str,
                   datasplit_path: str,
                   node_security_distance: Dict[Tuple[int, int], int],
                   family_node_security_list_map: Dict[int, Set[str]]):

    # `vertices` contains N vectors of length 50. Each vector contains node
    # labels the last node is always the ego investor.
    vertices = np.load(os.path.join(dataset_path, "vertex_id.npy")
                       ).astype(np.int64)
    # `vertex_adjacency` contains subgraph adjacencies for each of the
    # 1705 ego investors
    vertex_adjacency = np.load(
        os.path.join(dataset_path, "vertex_adjacency.npy"),
        allow_pickle=True).item()
    # `graphs` contain N 50x50 graph adjacencies for each observation
    graphs = np.array(
        [vertex_adjacency[vertex].toarray() for vertex in
            vertices[:, -1]], dtype=float).astype(np.float32)
    graphs[graphs != 0] = 1.0
    # `influence_features` contains N 50x2 influence features, where the first
    # column in each 50x2 array indicates neighbor activity and the second
    # column always indicates the last node as the ego investor.
    influence_features = np.load(
        os.path.join(dataset_path, "influence_feature.npy"),
        allow_pickle=True).astype(np.float32)
    # `labels` is a vector containing N elements indicating whether the ego
    # investor was active in the prediction period.
    labels = np.load(os.path.join(dataset_path, "label.npy")
                     ).astype(np.int64)

    embedding_dim = 64
    embedding_path = os.path.join(dataset_path,
                                  "deepwalk.emb_%d" % embedding_dim)
    max_vertex_idx = np.max(vertices)
    # `embedding` contains a 64-dimension embedding for each of 1705 nodes
    embedding = load_w2v_feature(embedding_path, max_vertex_idx)

    normalizer = nn.InstanceNorm1d(64, momentum=0.0, affine=True)
    emb = torch.from_numpy(np.array([embedding[vertices_obs] for vertices_obs in vertices]))
    norm_emb = normalizer(emb.transpose(1, 2)).transpose(1, 2)
    normalized_embeddings = norm_emb #.detach().numpy()
    # `security_ids` contains information about which security was traded
    security_ids = np.load(os.path.join(dataset_path, "security_ids.npy"))

    # Figure out distance to securities, own-security and family flags
    egos = vertices[:, -1]
    egos_security_ids = [(ego, isin) for ego, isin in zip(egos, security_ids)]

    distances = np.array([node_security_distance.get(ego_security_id)
                         for ego_security_id in egos_security_ids])
    own_company = (distances == 0).astype(int)
    family_flag = np.array([
        ego in family_node_security_list_map for ego in egos], dtype=int)
    data_splits = np.load(datasplit_path).astype(np.int64)

    save_data(dataset_path=dataset_path,
                         graphs=graphs,
                         influence_features=influence_features,
                         labels=labels,
                         distances=distances,
                         embeddings=normalized_embeddings,
                         own_company=own_company,
                         data_splits=data_splits,
                         family_flag=family_flag)


def save_data(dataset_path: str,
                         graphs: np.ndarray,
                         influence_features: np.ndarray,
                         labels: np.ndarray,
                         distances: np.ndarray,
                         embeddings: np.ndarray,
                         own_company: np.ndarray,
                         data_splits: np.ndarray,
                         family_flag: np.ndarray) -> None:
    """Save all anonymized data to `dataset_path` with prefix `Public_`. The
    data set contains a total of N observations

    Args:
        dataset_path (str): Location where all datasets are saved.
        graphs (np.ndarray): contain N 50x50 graph adjacencies.
        influence_features (np.ndarray): contains N 50x2 influence features,
        where the first column in each 50x2 array indicates neighbor activity
        and the second column always indicates the last node as the ego investor.
        labels (np.ndarray): vector containing N elements indicating whether
        the ego investor was active in the prediction period.
        embeddings (np.ndarray): contains a 64-dimension embeddings
        own_company (np.ndarray): a flag indicating whether the observation is
        about the ego investor trading her own company security (1) or
        securities of other companies (0).
        data_splits (np.ndarray): an indicator assigning the observations to
        training (0), validation (1), or test (2) data sets.
        family_flag (np.ndarray): a flag indicating whether the ego investor is
        a family member (1) or an insider (0).
    """
    df_graphs = pd.concat(
        [pd.DataFrame(graph, dtype=int)
            for graph in graphs],
        keys=range(graphs.shape[0]),
        names=['observation id', 'neighbor id'])
    df_graphs.columns.name = 'neighbor id'

    df_normalized_embedding = pd.concat(
        [pd.DataFrame(normalized_embedding, dtype=float)
            for normalized_embedding in embeddings],
        keys=range(graphs.shape[0]),
        names=['observation id', 'neighbor id'])
    df_normalized_embedding.columns.name = 'embedding dimension'

    df_influence = pd.concat(
        [pd.DataFrame(influence_feature,
                      columns=['influence', 'ego id'], dtype=int)
            for influence_feature in influence_features],
        keys=range(influence_features.shape[0]),
        names=['observation id', 'neighbor id'])

    df_labels = pd.Series(labels, name='label')
    df_labels.index.name = 'observation id'

    df_data_splits = pd.Series(data_splits, name='data_split')
    df_data_splits.index.name = 'observation id'

    df_family_flag = pd.Series(family_flag, name='family_flag')
    df_family_flag.index.name = 'observation id'

    df_distances = pd.Series(distances, name='distance')
    df_distances.index.name = 'observation id'

    df_own_company = pd.Series(own_company, name='own_company')
    df_own_company.index.name = 'observation id'

    dataset = os.path.basename(dataset_path[:-1])
    save_path = f'./public/data/{dataset}'
    os.makedirs(save_path, exist_ok=True)
    df_normalized_embedding.to_csv(save_path + '/Public_Normalized_Embedding.csv')
    # np.save(save_path + '/Public_Normalized_Embedding', embeddings)
    # torch.save(embeddings.detach(), save_path + '/Public_Normalized_Embedding_tensor.pt')
    df_influence.to_csv(save_path + '/Public_Influence_Features.csv')
    df_labels.to_csv(save_path + '/Public_Labels.csv')
    df_graphs.to_csv(save_path + '/Public_Graphs.csv')
    df_distances.to_csv(save_path + '/Public_Distances.csv')
    df_own_company.to_csv(save_path + '/Public_Own_Company_flag.csv')
    df_data_splits.to_csv(save_path + '/Public_Dataset_splits.csv')
    df_family_flag.to_csv(save_path + '/Public_Family_flag.csv')


def normalize_embedding(x):
    epsilon = 1e-5
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    y = (x - mu) / np.sqrt(var + epsilon)
    return y


if __name__ == '__main__':
    datasets: List[str] = []
    for horizon in ['Lead-lag', 'Simultaneous']:
        for period in ['D', 'W']:
            for direction in ['Buy', 'Sell']:
                datasets.append(f'{horizon}_{period}_{direction}')

    for dataset in tqdm(datasets, desc='Anonymize datasets'):
        dataset_path: str = DATASET_PATH.format(dataset=dataset)
        datasplit_path: str = DATASPLIT_PATH.format(dataset=dataset)

        NODE_SECURITY_DISTANCE: Dict[Tuple[int, int], int]
        FAMNODE_TO_ISIN: Dict[int, Set[str]]

        anonymize_data(dataset_path=dataset_path, datasplit_path=datasplit_path,
                    node_security_distance=NODE_SECURITY_DISTANCE,
                    family_node_security_list_map=FAMNODE_TO_ISIN)
