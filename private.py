
import json
import os
import numpy as np
import pandas as pd
import torch
import socket


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

def load_owner_security_distance():
    try:
        owner_security_distance = np.load(
            './data/input/owner_security_distance.npy',
            allow_pickle=True).item()
    except FileNotFoundError as e:
        print('Run func `get_owner_security_distance` in analyse_model_predictions.py')
        raise e
    return owner_security_distance


def load_owner_companies():
    try:
        owner_company_map = np.load('./data/input/owner_companies.npy',
                                    allow_pickle=True).item()
        family_in_company_map = np.load('./data/input/family_in_company.npy',
                                        allow_pickle=True).item()
    except FileNotFoundError as e:
        print('Run func `get_owner_companies` in analyse_model_predictions.py')
        raise e

    company_to_isin = pd.read_csv('./data/input/tab_company_isin.csv',
                                  delimiter=';', index_col=0)
    company_to_isin = company_to_isin.groupby(
        ['company']).ISIN_new.unique().map(set).to_dict()

    owner_to_isin = {}

    for owner in owner_company_map:
        owner_to_isin[owner] = set()
        for company in owner_company_map[owner]:
            owner_to_isin[owner].update(company_to_isin[company])

    family_to_isin = {}
    for member in family_in_company_map:
        family_to_isin[member] = set()
        for company in family_in_company_map[member]:
            family_to_isin[member].update(company_to_isin[company])

    return owner_to_isin, family_to_isin


COMPUTER_NAME = socket.gethostname()
PROJECT_DATA_FOLDER = (
    "/Volumes/WD Elements 2621 Media/Insider Influence/data/"
    if COMPUTER_NAME == 'wks-88866-mac.ad.tuni.fi' else
    './data/')
# PROJECT_DATA_FOLDER = '/Volumes/WD Elements 2621 Media/Insider Influence/data/'

RAW_DATA_DIR = PROJECT_DATA_FOLDER + 'real_data/real_0/'
DATASET_PATH = RAW_DATA_DIR + 'derivatives/{dataset}/'
DATASPLIT_PATH = DATASET_PATH + 'data_split/train75_valid12.5_test12.5_20.npy'
ID_SECURITY_MAP = dict(enumerate(SECURITY_LIST))
SECURITY_ID_MAP = {security: isin for isin,
                   security in ID_SECURITY_MAP.items()}

OWNER_SECURITY_DISTANCE = load_owner_security_distance()

owner_path = os.path.join(RAW_DATA_DIR, 'node_labels.json')
with open(owner_path, 'r') as node_labels:
    node_to_owner = json.load(node_labels)
node_to_owner = {int(owner): node_to_owner[owner]
                 for owner in node_to_owner}
owner_to_node = {node_to_owner[owner]: int(
    owner) for owner in node_to_owner}

NODE_SECURITY_DISTANCE = {
    (owner_to_node[owner], SECURITY_ID_MAP[security]): distance
    for (owner, security), distance in OWNER_SECURITY_DISTANCE.items() if security in SECURITY_ID_MAP}

owner_to_isin, family_to_isin = load_owner_companies()

node_to_isin = {
    owner_to_node[owner]: security_set for owner, security_set in owner_to_isin.items()}
FAMNODE_TO_ISIN = {
    owner_to_node[family]: security_set for family, security_set in family_to_isin.items()}

PATH_TO_MODEL = './results/random_data_splits_for_best_dataset_models_fixed_threshold_fixed'

if COMPUTER_NAME == 'wks-88866-mac.ad.tuni.fi':
    pass