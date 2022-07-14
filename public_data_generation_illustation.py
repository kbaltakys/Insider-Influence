from collections import Counter
from glob import glob
import json
import os
import shutil
from typing import Dict, Union
from joblib import Parallel, delayed
import numpy as np
from scipy import sparse
import networkx as nx
import pandas as pd
from tqdm import tqdm

START_DATE = '2005-01-01'
END_DATE = '2009-12-31'
EGO_RADIUS = 4
MINIMUM_ACTIVE_NEIGHBOURS = 1
RWR_JUMP_PROBABILITY = 0.8
NEGATIVE_POSITIVE_LABEL_RATIO = [3, 1]  # '3-1'

SECURITY_LIST = [u'FI0009000681', u'FI0009007132', u'FI0009902530',
                 u'FI0009005987', u'FI0009003305', u'FI0009007884',
                 u'FI0009007835', u'SE0000667925', u'FI0009013296',
                 u'FI0009002422', u'FI0009800643', u'FI0009003727',
                 u'FI0009005318', u'FI0009003552', u'FI0009013403',
                 u'FI0009005961', u'FI0009007694', u'FI0009004824',
                 u'FI0009000202', u'FI0009000459', u'FI0009000285',
                 u'FI0009005870', u'FI0009000251', u'FI0009013429',
                 u'FI0009000665', u'FI0009000277', u'FI0009014575',
                 u'FI0009801310', u'FI0009002158', u'FI0009008221']


class OnlyPositiveLabels(Exception):
    pass


class OnlyNegativeLabels(Exception):
    pass


class DataSet:

    def __init__(self, path: str, verbose: bool = True):
        self.paths = DataSet.load_data_paths(path)
        if verbose:
            print(f'Loading data set files. "{self.paths["root"]}"')
        self.mapping_node_to_owner = DataSet.load_json(
            self.paths['node_labels'])
        if verbose:
            print(f'Network contains {len(self.mapping_node_to_owner)} nodes.')
        self.node_neighbourhoods = DataSet.load_json(
            self.paths['node_neighbourhoods'])

        self.influence_features = dict()
        self.labels = dict()
        self.periods = dict()
        self.vertex_ids = dict()
        self.number_of_samples = dict()

        for frequency in ['D', 'W']:
            if verbose:
                print(f'Loading data with freq="{frequency}"')
            self.influence_features[frequency] = \
                DataSet.load_influence_features(
                    self.paths[frequency]['influence_features'], verbose)
            self.labels[frequency] = DataSet.load_labels(
                self.paths[frequency]['labels'], verbose)
            self.periods[frequency] = DataSet.load_periods(
                self.paths[frequency]['periods'], verbose)
            self.vertex_ids[frequency] = DataSet.load_vertex_ids(
                self.paths[frequency]['vertex_ids'], verbose)

            self.number_of_samples[frequency] = len(self.periods[frequency])

        self.check_data()
        self.number_of_securities = self.labels[frequency].shape[1]

        if verbose:
            print('Data successfully loaded, integrity verified.')

    @classmethod
    def load_data_paths(cls, path: str) -> str:
        paths = dict()

        assert os.path.isdir(path), f'Not valid data set path: {path}!'
        paths['root'] = path

        assert os.path.isfile(os.path.join(path, 'node_labels.json')), \
            'Data set does not contain node-owner mapping!'
        paths['node_labels'] = os.path.join(path, 'node_labels.json')

        assert os.path.isfile(os.path.join(path, 'node_neighbourhoods.json')), \
            'Data set does not contain node neighbourhood mapping!'
        paths['node_neighbourhoods'] = os.path.join(path,
                                                    'node_neighbourhoods.json')

        for frequency in ['D', 'W']:
            assert os.path.isdir(os.path.join(path, frequency)), \
                f'Data set does not contain freq="{frequency}" information!'
            paths[frequency] = dict()

            assert os.path.isfile(
                os.path.join(path, frequency, 'influence_feature.npy')), \
                f'Data set (freq="{frequency}") ' \
                f'does not contain influence features!'
            paths[frequency]['influence_features'] = \
                os.path.join(path, frequency, 'influence_feature.npy')

            assert os.path.isfile(os.path.join(path, frequency, 'label.npy')), \
                f'Data set (freq="{frequency}") ' \
                f'does not contain labels!'
            paths[frequency]['labels'] = os.path.join(
                path, frequency, 'label.npy')

            assert os.path.isfile(
                os.path.join(path, frequency, 'periods.npy')), \
                f'Data set (freq="{frequency}") ' \
                f'does not contain period information!'
            paths[frequency]['periods'] = os.path.join(
                path, frequency, 'periods.npy')

            assert os.path.isfile(
                os.path.join(path, frequency, 'vertex_id.npy')), \
                f'Data set (freq="{frequency}") ' \
                f'does not contain vertex_ids!'
            paths[frequency]['vertex_ids'] = \
                os.path.join(path, frequency, 'vertex_id.npy')
        return paths

    @classmethod
    def load_json(cls, path: str) -> dict:
        with open(path, 'r') as f:
            d = json.load(f)
        return {int(key): value for key, value in d.items()}

    @classmethod
    def load_influence_features(cls, path: str, verbose: bool = True):
        if verbose:
            print('Loading influence features ...', end=' ')
        influence = np.load(path, allow_pickle=True)
        influence = np.array(
            [sparse_matrix.toarray() for sparse_matrix in influence])
        if verbose:
            print(f'\t DONE! Shape: {influence.shape}')
        return influence

    @classmethod
    def load_labels(cls, path: str, verbose: bool = True):
        # TODO: load sparse labels too.
        if verbose:
            print('Loading labels ...', end=' ')
        labels = np.load(path)
        if verbose:
            print(f'\t\t\t\t DONE! Shape: {labels.shape}')
        return labels

    @classmethod
    def load_periods(cls, path: str, verbose: bool = True):
        if verbose:
            print('Loading periods ...', end=' ')
        periods = np.load(path, allow_pickle=True)
        if verbose:
            print(f'\t\t\t DONE! Shape: {periods.shape}')
        return periods

    @classmethod
    def load_vertex_ids(cls, path: str, verbose: bool = True):
        if verbose:
            print('Loading vertex ids ...', end=' ')
        vertex_ids = np.load(path)
        if verbose:
            print(f'\t\t\t DONE! Shape: {vertex_ids.shape}')
        return vertex_ids

    def check_data(self):
        number_of_securities = dict()
        for frequency in ['D', 'W']:
            assert (len(self.influence_features[frequency]) ==
                    len(self.periods[frequency]) ==
                    len(self.vertex_ids[frequency]) ==
                    len(self.labels[frequency])), \
                'Unequal number of samples in the data set'

            influence_feature_securities = \
                self.influence_features[frequency].shape[2] - 1
            label_securities = self.labels[frequency].shape[1]
            assert influence_feature_securities == label_securities, \
                'Unequal number of securities in influence features and labels'

            assert self.influence_features[frequency][:, -1, -1].all(), \
                'Ego indicator not always equal to 1'

            assert not self.influence_features[frequency][:, :-1, -1].any(), \
                'Ego indicator assigned to neighbours'

            number_of_securities[frequency] = self.labels[frequency].shape[1]

        assert (number_of_securities['D'] == number_of_securities['W']), \
            'Number of securities in weekly and daily data sets is not equal!'


def load_insider_network_links() -> pd.DataFrame:
    pass
    return pd.DataFrame()


def create_network_from_links(all_links):
    return nx.from_pandas_edgelist(all_links, 'source_id', 'target_id',
                                   edge_attr='weight',
                                   create_using=nx.Graph())


def randomize_network(G: nx.Graph, seed: int) -> nx.Graph:
    degree_sequence = [degree for node, degree in G.degree]
    H = nx.configuration_model(degree_sequence, seed=seed)
    # Remove parallel edges
    H = nx.Graph(H)
    # Remove self loops
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    nx.set_node_attributes(H, {h_node: g_node for h_node, g_node in
                               enumerate(G.nodes())}, 'label')
    nx.set_edge_attributes(H, 1, 'weight')
    return H


def anonymize_network(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    # Anonymization happens here
    return H


def write_adj_list(G: nx.Graph, path: str):
    with open(path, 'w') as f:
        for line in nx.generate_adjlist(G):
            f.write(line + '\n')


def write_node_labels(G: nx.Graph, path: str):
    label_dict = nx.get_node_attributes(G, 'label')
    with open(path, 'w') as f:
        json.dump(label_dict, f, indent=4)


def write_gexf_network(G: nx.Graph, path: str):
    nx.write_gexf(G, path)


def calculate_vertex_features(G):
    features = pd.DataFrame(index=list(G.nodes()))

    coreness_dict = nx.core_number(G)
    features['coreness'] = features.index.to_series().map(coreness_dict)

    pagerank_dict = nx.pagerank(G)
    features['pagerank'] = features.index.to_series().map(pagerank_dict)

    hubs_dict, authority_dict = nx.hits(G)
    features['hubs'] = features.index.to_series().map(hubs_dict)
    features['authority'] = features.index.to_series().map(
        authority_dict)

    eigenvector_dict = nx.eigenvector_centrality(G)
    features['eigenvector'] = features.index.to_series().map(
        eigenvector_dict)

    clustering_dict = nx.clustering(G)
    features['clustering'] = features.index.to_series().map(
        clustering_dict)

    rarity_dict = G.degree()
    features['rarity'] = features.index.to_series().map(rarity_dict)
    features['rarity'] = 1. / features['rarity']

    return features


def write_vertex_features(G: nx.Graph, path: str):
    vertex_features = calculate_vertex_features(G)
    np.save(path, vertex_features.T.values)
    vertex_features.to_json(path + '.json', orient='index', indent=4)


def make_embedding(path, graph_input, graph_output: str = f"deepwalk.emb_64",
                   verbose: bool = False):
    path_to_adjacency_list = os.path.join(path, graph_input)
    path_to_embedding = os.path.join(path, graph_output)

    if verbose:
        print('Creating embedding layer')
        print(f'deepwalk --input "{path_to_adjacency_list}" '
              f'--output "{path_to_embedding}" &')
        print(os.getcwd())
    if os.system(f'deepwalk --input "{path_to_adjacency_list}" '
                 f'--output "{path_to_embedding}"'):
        raise Exception('Deepwalk failed')


def random_walk_with_restart(G_full: nx.Graph, starting_node: [int, str],
                             restart_probability: float,
                             neighbourhood_size: int,
                             radius: int) -> list:
    G = nx.ego_graph(G_full, starting_node, radius)
    if len(G) < neighbourhood_size:
        return list()
    neighbourhood = set()
    neighbourhood.add(int(starting_node))
    current_node = starting_node
    while len(neighbourhood) < neighbourhood_size:
        p_jump = np.random.random()
        if p_jump < restart_probability:
            current_node = starting_node
        neighbors = list(nx.neighbors(G, current_node))
        weights = np.array([attributes['weight'] for node, attributes in
                            G[current_node].items()], dtype=float)
        weights /= weights.sum()
        current_node = np.random.choice(neighbors, p=weights)
        neighbourhood.add(int(current_node))
    neighbourhood.remove(int(starting_node))
    return sorted(neighbourhood) + [int(starting_node)]


def parallel_rwr(G: nx.Graph, radius: int,
                 neighbourhood_size: int,
                 restart_probability: float = 0.8) -> dict:
    nodes = G.nodes()
    progress_bar = tqdm(nodes, desc='Walking neighbourhoods',
                        leave=False)
    neighbourhoods = Parallel(n_jobs=-1, prefer='processes')(
        delayed(random_walk_with_restart)(
            G_full=G,
            starting_node=ego,
            restart_probability=restart_probability,
            neighbourhood_size=neighbourhood_size,
            radius=radius)
        for ego in progress_bar)
    results = dict(zip(nodes, neighbourhoods))
    return results


def write_ego_neighbourhood_adjacencies(G: nx.Graph, ego_neighbourhoods: dict,
                                        path: str):
    ego_adjacencies = dict()
    for ego in ego_neighbourhoods:
        neighbourhood = ego_neighbourhoods[ego]
        if neighbourhood:
            ego_network = nx.subgraph(G, neighbourhood)
            adj = nx.adjacency_matrix(
                ego_network, nodelist=ego_neighbourhoods[ego])
            ego_adjacencies[ego] = adj
    np.save(path, ego_adjacencies)


def write_ego_neighbourhoods(neighbourhoods: dict, path: str):
    with open(path, 'w') as f:
        json.dump(neighbourhoods, f, indent=4)


def generate_one_dataset(random_seed: int, G: nx.Graph, save_path: str,
                         random: bool = True):
    # 1. Create a network, random or anonymized
    if random:
        iteration_path = os.path.join(save_path, f'random_{random_seed}')
        network = randomize_network(G, seed=random_seed)
    else:
        iteration_path = os.path.join(save_path, f'real_{random_seed}')
        network = anonymize_network(G)
    os.makedirs(iteration_path, exist_ok=True)
    write_adj_list(network,
                   os.path.join(iteration_path, 'adj_list.txt'))
    write_node_labels(network,
                      os.path.join(iteration_path, 'node_labels.json'))
    write_gexf_network(network,
                       os.path.join(iteration_path, 'node_labels.gexf'))
    write_vertex_features(network,
                          os.path.join(iteration_path, 'vertex_feature'))

    make_embedding(path=iteration_path, graph_input='adj_list.txt')

    ego_neighbourhoods = parallel_rwr(
        G=network, radius=4, neighbourhood_size=50,
        restart_probability=0.8)
    write_ego_neighbourhoods(ego_neighbourhoods,
                             os.path.join(iteration_path,
                                          'node_neighbourhoods.json'))

    write_ego_neighbourhood_adjacencies(
        network, ego_neighbourhoods,
        os.path.join(iteration_path, 'vertex_adjacency.npy'))


def generate_series_of_networks(save_path: str, number_of_datasets: int,
                                random: bool = True):
    real_network_links = load_insider_network_links()
    real_insider_network = create_network_from_links(real_network_links)
    for iteration in tqdm(
            range(number_of_datasets), total=number_of_datasets,
            desc=f'Creating {"randomized" if random else "real"} data set'):
        generate_one_dataset(random_seed=iteration,
                             G=real_insider_network,
                             save_path=save_path,
                             random=random)


def load_insider_trades() -> pd.DataFrame:
    pass
    return pd.DataFrame()


def filter_trades(trades) -> pd.DataFrame:
    trades = trades[trades['isin'].isin(SECURITY_LIST) & (
        START_DATE <= trades['trading_date']) & (
        trades[
            'trading_date'] <= END_DATE)
    ]
    return trades


def load_neighbourhoods(path: str) -> dict:
    with open(path, 'r') as f:
        neighbourhood_data = json.load(f)
    return {int(key): value for key, value in neighbourhood_data.items()}


def load_node_map(path: str) -> dict:
    with open(path, 'r') as f:
        node_map = json.load(f)
    return {value: int(key) for key, value in node_map.items()}


def remap_owners(trades: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    trades = trades.loc[trades.owner_id.isin(mapping), :]
    node_ids = trades.owner_id.replace(mapping)
    trades['node_id'] = node_ids
    return trades


def extract_node_activation(trades: pd.DataFrame, frequency: str):
    net_trades = trades.groupby(
        ['node_id', 'isin', 'trading_date']).vol.sum()

    if frequency == 'W':
        net_trades = net_trades.groupby(level=['node_id', 'isin']).resample(
            'W', level='trading_date').sum()
    net_trades = net_trades[net_trades != 0]
    net_trades = net_trades.reorder_levels(['node_id', 'trading_date', 'isin'])
    net_trades.sort_index(level=['node_id', 'trading_date', 'isin'],
                          inplace=True)

    node_activation_ = net_trades.map(
        lambda x: -1 if x < 0 else 1 if x > 0 else 0)
    return node_activation_


def get_trading_periods(frequency: str = None):
    omxhpi = pd.read_csv('OMXHPI.csv', parse_dates=['Date'],
                         usecols=['Date']).Date
    omxhpi = omxhpi[omxhpi.between(START_DATE, END_DATE)]
    trading_days = [pd.to_datetime(day) for day in sorted(omxhpi.values)]
    if frequency == 'W':
        trading_periods = pd.date_range(min(trading_days), max(trading_days),
                                        freq=frequency).tolist()
        return trading_periods
    else:
        return trading_days


def get_ego_features(ego, time_index, ego_neighbourhoods, activation):
    neighbourhood_activation = activation.reindex(
        index=ego_neighbourhoods[ego])
    neighbourhood_activation.loc[ego, pd.IndexSlice[:, 'ego']] = 1
    new_ego_labels = neighbourhood_activation.loc[ego].unstack().reindex(
        columns=SECURITY_LIST).loc[time_index[1:]]
    # features drop last
    new_features = np.array(
        [sparse.csr_matrix(data.fillna(0)) for period, data
         in
         neighbourhood_activation[time_index[:-1]].groupby(
             level=['trading_date'], axis=1)])
    new_vertex_ids = np.tile(ego_neighbourhoods[ego], (len(time_index) - 1, 1))
    new_periods = np.array(time_index[1:])
    # Keep only the samples with active neighbours
    mask = [feature.nnz > 1 for feature in new_features]

    return {'vertex_ids': new_vertex_ids[mask],
            'features': new_features[mask],
            'labels': new_ego_labels[mask],
            'periods': new_periods[mask]}


def match_labels(a: np.array, labels: Union[list, int]) -> np.array:
    if not isinstance(labels, list):
        return a == labels
    else:
        mask = np.ones_like(a, dtype=bool)
        for label in labels:
            mask = np.multiply(mask, a != label)
    return ~ mask


def fix_label_ratio(labels, security_ids, influence_features, vertex_ids,
                    periods, verbose: bool = True):
    counter = Counter(labels)
    if verbose:
        print('Label counts:', counter)

    if 0 not in counter:
        raise OnlyPositiveLabels
    if 1 not in counter:
        raise OnlyNegativeLabels

    neg_idx = np.argwhere(labels == 0).squeeze()
    pos_idx = np.argwhere(labels == 1).squeeze()
    neg_data, pos_data = float(counter[0]) / counter[1], 1
    neg_required, pos_required = NEGATIVE_POSITIVE_LABEL_RATIO

    if verbose:
        print(f'Desired label ratio {neg_required} : {pos_required}')
        print(f'Starting label ratio: {neg_data:.2f} : {pos_data}')

    if neg_data / pos_data > neg_required / pos_required:
        # reduce negatives
        neg_count = int(len(pos_idx) * neg_required / pos_required)
        neg_idx = np.random.choice(neg_idx, size=neg_count, replace=False)
    else:
        # reduce positives
        pos_count = int(len(neg_idx) * pos_required / neg_required)
        pos_idx = np.random.choice(pos_idx, size=pos_count, replace=False)

    idx = sorted(np.hstack((neg_idx, pos_idx)))

    labels = labels[idx]
    periods = periods[idx]
    vertex_ids = vertex_ids[idx]
    influence_features = influence_features[idx]
    security_ids = security_ids[idx]
    if verbose:
        counter = Counter(labels)
        neg_data, pos_data = float(counter[0]) / counter[1], 1
        print(f'Ending label ratio {neg_data:.2f} : {pos_data}')
    return labels, security_ids, influence_features, vertex_ids, periods


def create_features2(activation: pd.DataFrame,
                     ego_neighbourhoods: dict,
                     frequency: str):
    time_index = get_trading_periods(frequency=frequency)

    column_index = pd.MultiIndex.from_product(
        [time_index, SECURITY_LIST + ['ego']], names=['trading_date', 'isin'])
    activation_unstacked = activation.unstack([1, 2])
    activation_unstacked = activation_unstacked.reindex(column_index, axis=1)

    results = Parallel(n_jobs=-1, prefer='processes')(
        delayed(get_ego_features)(ego, time_index, ego_neighbourhoods,
                                  activation_unstacked) for ego in
        tqdm(list(ego_neighbourhoods), desc=f'Egos ({frequency})', leave=False)
        if ego_neighbourhoods[ego])

    vertex_ids_ = np.vstack([result['vertex_ids'] for result in results])
    features_ = np.hstack([result['features'] for result in results])
    labels_ = np.vstack([result['labels'] for result in results])
    periods_ = np.hstack([result['periods'] for result in results])

    return vertex_ids_, features_, labels_, periods_


def create_influence_data(trades: pd.DataFrame, path: str):
    neighbourhood_path = os.path.join(path, 'node_neighbourhoods.json')
    node_map_path = os.path.join(path, 'node_labels.json')
    neighbourhoods = load_neighbourhoods(neighbourhood_path)
    node_to_owner = load_node_map(node_map_path)
    remapped_trade_data = remap_owners(trades.copy(), node_to_owner)
    for frequency in ['D', 'W']:
        node_activation = extract_node_activation(remapped_trade_data,
                                                  frequency=frequency)
        vertex_ids, features, labels, periods = create_features2(
            node_activation,
            neighbourhoods,
            frequency=frequency)
        os.makedirs(os.path.join(path, frequency), exist_ok=True)
        np.save(os.path.join(path, frequency, 'vertex_id.npy'), vertex_ids)
        np.save(os.path.join(path, frequency, 'influence_feature.npy'),
                features)
        np.save(os.path.join(path, frequency, 'label.npy'), labels)
        np.save(os.path.join(path, frequency, 'periods.npy'), periods)


def copy_embedding(embedding_location: str, save_path: str):
    shutil.copy2(
        os.path.join(embedding_location, 'deepwalk.emb_64'),
        save_path + '/deepwalk.emb_64')


def generate_data_set(dataset: DataSet, verbose: bool = True,
                      remove_contemporaneous_signals: bool = False):
    security_ids = None
    if verbose:
        print('Generating datasets.')
    for frequency in ['D', 'W']:
        for prediction in ['Simultaneous', 'Lead-lag']:
            for transaction_type in ['Sell', 'Buy']:
                if verbose:
                    print(f'\nfreq: {frequency}, '
                          f'prediction: {prediction}, '
                          f'transaction type: {transaction_type}.')

                labels = dataset.labels[frequency].copy()
                influence_features = dataset.influence_features[
                    frequency].copy()
                periods = dataset.periods[frequency].copy()
                vertex_ids = dataset.vertex_ids[frequency].copy()

                if prediction == 'Simultaneous':
                    labels = influence_features[:, -1, :-1].copy()
                    influence_features[:, -1, :-1] = 0
                # Leave no information about current ego behaviour!
                elif remove_contemporaneous_signals:
                    influence_features[:, -1, :-1] = 0

                if transaction_type == 'Buy':
                    behaviour_code = 1
                elif transaction_type == 'Sell':
                    behaviour_code = -1
                else:
                    raise NotImplemented

                behaviour_labels = np.where(
                    match_labels(labels, behaviour_code), 1, np.nan)

                influence_features[:, :, :-1] = np.where(
                    match_labels(influence_features[:, :, :-1],
                                 behaviour_code),
                    1, 0)

                # Checking which of the desired ego behaviours were influenced
                # by same type of behaviours in the neighbourhood.
                labels = np.equal(
                    influence_features[:, :-1, :-1],
                    behaviour_labels.reshape(
                        -1, 1, dataset.number_of_securities)
                ).any(axis=1).astype(np.int64)  # type: ignore

                if verbose:
                    print('Splitting security data set:', end=' ')

                security_features = np.dsplit(
                    influence_features, influence_features.shape[2])
                ego_indicator = security_features.pop(-1)

                keep_ids = np.zeros((dataset.number_of_securities,
                                     dataset.number_of_samples[frequency]),
                                    dtype=bool)

                for security_id in range(dataset.number_of_securities):
                    keep_ids[security_id] = np.count_nonzero(
                        security_features[security_id][:, :-1].squeeze(),
                        axis=1) >= MINIMUM_ACTIVE_NEIGHBOURS
                    if verbose:
                        print(f'{security_id},', end=' ')

                if verbose:
                    print('\nCombining security samples:')
                influence_features = np.vstack([
                    np.dstack(
                        (security_activation, ego_indicator))[
                        keep_ids[security_id]]
                    for security_id, security_activation in
                    enumerate(security_features)])
                if verbose:
                    print(f'\tinfluence features \t{influence_features.shape}')

                security_ids = np.repeat(range(dataset.number_of_securities),
                                         keep_ids.sum(axis=1))
                if verbose:
                    print(f'\tsecurity ids \t\t{security_ids.shape}')

                labels = labels.T[keep_ids]
                if verbose:
                    print(f'\tlabels \t\t\t\t{labels.shape}')

                vertex_ids = vertex_ids[keep_ids.nonzero()[1]]
                if verbose:
                    print(f'\tvertex ids \t\t\t{vertex_ids.shape}')
                periods = periods[keep_ids.nonzero()[1]]
                if verbose:
                    print(f'\tperiods \t\t\t\t{periods.shape}')

                labels, security_ids, influence_features, \
                    vertex_ids, periods = fix_label_ratio(
                        labels, security_ids, influence_features, vertex_ids,
                        periods, verbose=verbose)

                save_path = os.path.join(
                    dataset.paths['root'],
                    'derivatives',
                    f'{prediction}_{frequency}_{transaction_type}')
                os.makedirs(save_path, exist_ok=True)

                shutil.copy2(
                    os.path.join(dataset.paths['root'], 'vertex_feature.npy'),
                    save_path + '/vertex_feature.npy')

                shutil.copy2(
                    os.path.join(dataset.paths['root'],
                                 'vertex_adjacency.npy'),
                    save_path + '/vertex_adjacency.npy')

                copy_embedding(dataset.paths['root'], save_path)

                np.save(save_path + '/vertex_id', vertex_ids)
                np.save(save_path + '/influence_feature', influence_features)
                np.save(save_path + f'/label', labels)
                np.save(save_path + f'/security_ids', security_ids)
                np.save(save_path + f'/periods', periods)


if __name__ == '__main__':

    for RANDOM, NUMBER, NAME in zip([False, True],
                                    [1, 100],
                                    ['real_data', 'random']):
        data_path = 'path/to/data/set'
        generate_series_of_networks(data_path, number_of_datasets=NUMBER,
                                    random=RANDOM)

        trade_data = load_insider_trades()
        trade_data = filter_trades(trade_data)
        for path_to_data in glob(data_path + '*'):

            create_influence_data(trade_data, path_to_data)
            data = DataSet(path_to_data)
            generate_data_set(data)
