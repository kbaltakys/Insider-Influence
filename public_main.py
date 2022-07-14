#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:06:30 2022

@author: baltakys
"""

import copy
import torch.nn as nn
import json
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from public_earlystopping import EarlyStopping
from public_data_loader import return_train_valid_test_sets
from public_gcn import BatchGCN
from public_gat import BatchGAT
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, \
    precision_recall_curve


def train_model(model, dataloader, args: dict, use_cuda: bool, patience: int,
                epochs: int, result_dir: str, verbose: bool = True):
    print_every = 5
    tensorboard_logger = SummaryWriter(result_dir)
    if args['class_weight_balanced']:
        class_weight = dataloader['test'].dataset.get_class_weight()
    else:
        class_weight = torch.ones(dataloader['test'].dataset.n_classes)

    if use_cuda:
        device = torch.device("cuda:1")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(args['seed'])
        model.cuda(device)
        class_weight = class_weight.cuda(device)
        # print('Models moved to GPU.')
    else:
        device = torch.device("cpu")
        torch.manual_seed(args['seed'])
        # print('Only CPU available.')

    params = [{'params': model.layer_stack.parameters()}]

    optimizer = optim.Adagrad(params, lr=args['lr'], weight_decay=1e-3)

    train_loader = dataloader['train']
    valid_loader = dataloader['valid']
    test_loader = dataloader['test']

    param_path = os.path.join(result_dir, 'checkpoint.pt')

    # Defining the early stopping monitor
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   path=param_path, trace_func=print)

    # Loss function with class weights
    criterion = torch.nn.NLLLoss(class_weight)

    progress_bar = tqdm(total=epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = 0.
        train_totals = 0.

        for _, (data, target) in enumerate(train_loader):
            batch_size = data[0].size(0)

            if use_cuda:
                target = target.cuda(DEVICE)
                data = [tensor.cuda(DEVICE) for tensor in data]

            optimizer.zero_grad()
            output = model(data[:2], data[-1])
            loss_train = criterion(output, target)
            train_losses += batch_size * loss_train.item()
            train_totals += batch_size
            loss_train.backward()
            optimizer.step()

        train_loss = train_losses / train_totals
        progress_bar.update()
        print("train loss in this epoch %f %f", epoch, train_loss)
        tensorboard_logger.add_scalar('Loss/train',
                                      train_loss, epoch)
        # =========================================================================
        #   VALIDATE MODEL
        # =========================================================================

        valid_loss, best_thr, valid_stats = evaluate(model, class_weight,
                                                     valid_loader)
        print(f" epoch: {epoch} train_loss: {train_loss:.5f}, "
                    f"valid_loss: {valid_loss:.5f}, best_thr: {best_thr}")

        tensorboard_logger.add_scalar('Loss/valid',
                                      valid_loss, epoch)

        tensorboard_logger.add_scalar('Precision/raw',
                                      valid_stats['prec'][0], epoch)
        tensorboard_logger.add_scalar('Precision/threshold',
                                      valid_stats['prec'][1], epoch)

        tensorboard_logger.add_scalar('Recall/raw',
                                      valid_stats['rec'][0], epoch)
        tensorboard_logger.add_scalar('Recall/threshold',
                                      valid_stats['rec'][1], epoch)

        tensorboard_logger.add_scalar('F1/raw',
                                      valid_stats['f1'][0], epoch)
        tensorboard_logger.add_scalar('F1/threshold',
                                      valid_stats['f1'][1], epoch)

        early_stopping(valid_loss, model)

        if epoch % print_every == 0:
            progress_bar.set_description(
                f'Epoch {epoch}/{epochs} Train Loss: {train_loss:.5f} Valid Loss: {valid_loss:.5f}')

        if early_stopping.early_stop:
            if verbose:
                progress_bar.set_description(
                    "Early stopping on epoch {}".format(epoch))

            break
    # TODO: should the validation be done after the best model is loaded?
    model.load_state_dict(torch.load(param_path))
    model.eval()

    _, best_thr, _ = evaluate(model, class_weight, valid_loader)

    print(f" epoch: {epoch} train_loss: {train_loss}, "
                f"valid_loss: {valid_loss}, best_thr: {best_thr}")
    test_loss, _, test_stats = evaluate(model, class_weight,
                                        test_loader, best_thr=best_thr)
    tensorboard_logger.add_scalar('Loss/test', test_loss, epoch)
    tensorboard_logger.add_pr_curve('Test', test_stats['labels'],
                                    test_stats['predictions'])

    tensorboard_logger.add_hparams(
        hparam_dict={
            'lr': args['lr'],
            'batch': args['batch_size'],
            'model': args['model'],
            'hidden-units': args['hidden_units'] +
                            (
                                ' heads-' + args['heads'] if args['model'] == 'gat' else ''),
            'patience': patience,
            'train-size': args['train_ratio'],
            'valid-ratio': args['valid_ratio'],
            'drop-out': args['dropout'],
            'epochs': epochs,
            'last-epoch': epoch,
            'seed': args['seed'],
            'data-split-seed': args['data_split_seed'],
            'weight-decay': 1e-3,
            'class_weight_balanced': args['class_weight_balanced'],
            'best_threshold': best_thr
        },
        metric_dict={
            'test_loss': test_loss,
            'test_auc': test_stats['auc'],
            'test_prec': test_stats['prec'][0],
            'test_rec': test_stats['rec'][0],
            'test_f1': test_stats['f1'][0],
            'test_prec_thr': test_stats['prec'][1],
            'test_rec_thr': test_stats['rec'][1],
            'test_f1_thr': test_stats['f1'][1]})
    tensorboard_logger.close()
    return model


def evaluate(model, class_weight, loader, best_thr=None):
    model.eval()
    total = 0.
    loss = 0.
    y_true, y_pred, y_score = [], [], []
    class_weight = class_weight.to(DEVICE)

    for _, (data, target) in enumerate(loader):
        # graph, features, labels, vertices = batch
        bs = data[0].size(0)

        target = target.cuda(DEVICE)  # labels
        data = [tensor.cuda(DEVICE) for tensor in
                data]

        output = model(data[:2], data[-1])
        loss_batch = F.nll_loss(output, target, class_weight)
        loss += bs * loss_batch.item()
        y_true += target.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    if best_thr is None:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]

    y_score = np.array(y_score)
    y_pred_thr = np.zeros_like(y_score)
    y_pred_thr[y_score > best_thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary")
    prec_th, rec_th, f1_th, _ = precision_recall_fscore_support(
        y_true, y_pred_thr, average="binary")

    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    acc_th = accuracy_score(y_true, y_pred_thr)

    return loss / total, best_thr, {
        'auc': auc,
        'acc': [acc, acc_th],
        'prec': [prec, prec_th],
        'rec': [rec, rec_th],
        'f1': [f1, f1_th],
        'predictions': torch.tensor(np.exp(y_score)),
        'labels': torch.tensor(y_true),
        'predicted_labels': y_pred_thr
    }


def evaluate_predictions(df):
    prec, rec, f1, _ = precision_recall_fscore_support(
        df.label, df.prediction, average="binary", zero_division=0)

    acc = accuracy_score(df.label, df.prediction)

    if df.label.nunique() == 2:
        auc = roc_auc_score(df.label, df.score)
    elif df.label.nunique() == 1:
        auc = None
    else:
        raise NotImplementedError('Unexpected number of different labels')
    return pd.Series(
        {'prec': prec, 'rec': rec, 'f1': f1, 'acc': acc, 'auc': auc,
         'n_pos_labels': df.label.sum() / df.shape[0],
         'n_samples': df.shape[0]})


def get_parameters(horizon: str, frequency: str, direction: str,  architecture: str):

    with open('./models/gat_gcn_parameters.json', 'r') as file:
        model_parameters = json.load(file)

    params = model_parameters[f'{architecture}_{horizon}_{frequency}_{direction}']
    args = {'data_split_seed': 20,
            'seed': 43,
            'batch_size': int(model_parameters[f'{architecture}_{horizon}_{frequency}_{direction}']['batch']),
            'public_file_dir': f'./data/{horizon}_{frequency}_{direction}/',
            'shuffle': False,
            'train_ratio': 75,
            'valid_ratio': 12.5,
            'model': architecture,
            'hidden_units': params['hidden-units'],
            'heads': params['heads'],
            'dropout': params['drop-out'],
            'class_weight_balanced': True,
            'lr': params['lr']}

    return args


if __name__ == '__main__':
    use_cuda = False
    train = False
    if use_cuda:
        DEVICE = torch.device("cuda:1")
    else:
        DEVICE = torch.device("cpu")

    datasets: List[Tuple[str, str, str, str]] = []
    for architecture in ['gat', 'gcn']:
        for horizon in ['Lead-lag', 'Simultaneous']:
            for frequency in ['D', 'W']:
                for direction in ['Buy', 'Sell']:
                    datasets.append(
                        (architecture, horizon, frequency, direction)
                    )

    # Main result for best GCN and GAT architectures
    table_5 = pd.DataFrame(index=pd.MultiIndex.from_tuples(datasets))

    # Only samples with non-own securities
    table_8 = pd.DataFrame(index=pd.MultiIndex.from_tuples(datasets))

    # Only only samples with non-own securities traded by insiders themselves
    table_9 = pd.DataFrame(index=pd.MultiIndex.from_tuples(datasets))

    prediction_list: List[pd.DataFrame] = []
    for architecture, horizon, frequency, direction in datasets:

        args = get_parameters(horizon, frequency, direction, architecture)

        np.random.seed(args['data_split_seed'])
        torch.manual_seed(args['data_split_seed'])
        data_loader = return_train_valid_test_sets(
            args, batch_size=args['batch_size'])

        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])

        n_neighbors = data_loader['test'].dataset.n_neighbors
        n_classes = data_loader['test'].dataset.get_num_class()

        feature_dim = data_loader['test'].dataset.get_feature_dimension()
        n_units = [feature_dim] + \
            [int(x) for x in args['hidden_units'].strip().split(",")] + \
            [data_loader['test'].dataset.n_classes]

        # Model and optimizer
        if args['model'] == "gcn":
            model = BatchGCN(#pretrained_emb=embedding,
                             n_neighbors=n_neighbors,
                             n_units=n_units,
                             dropout=args['dropout']
                             )
        elif args['model'] == "gat":
            n_heads = [int(x) for x in args['heads'].strip().split(",")]
            model = BatchGAT(#pretrained_emb=embedding,
                             n_units=n_units,
                             n_heads=n_heads,
                             dropout=args['dropout'],
                             )
        else:
            raise NotImplementedError

        if train:
            # dataiter = iter(data_loader['train'])
            # data, target = next(dataiter)
            result_dir = (
                './models/'
                f'new_{architecture}_{horizon}_{frequency}_{direction}/')
            trained_model = train_model(
                model=model,
                dataloader=data_loader,
                args=args,
                use_cuda=use_cuda,
                patience=10 ,
                epochs=500,
                result_dir=result_dir)
        else:
            path_model_checkpoint = (
                './models/'
                f'{architecture}_{horizon}_{frequency}_{direction}/'
                'checkpoint.pt')
            model.load_state_dict(torch.load(path_model_checkpoint))

        model.to(DEVICE)
        model.eval()

        test_loader = data_loader['test']
        test_loader.sampler.shuffle = False
        data_loader['valid'].sampler.shuffle = False

        if args['class_weight_balanced']:
            class_weight = test_loader.dataset.get_class_weight()
        else:
            class_weight = torch.ones(test_loader.dataset.n_classes)

        valid_loss, best_thr, valid_stats = \
            evaluate(model, class_weight, data_loader['valid'])

        distances = []
        family_flags = []
        own_company_flags = []
        for data, _ in test_loader:
            (_, _, batch_distances, batch_family_flags,
             batch_own_company_flags, _) = data
            distances.append(batch_distances.numpy())
            family_flags.append(batch_family_flags.numpy())
            own_company_flags.append(batch_own_company_flags.numpy())
        distances = np.hstack(distances)
        family_flags = np.hstack(family_flags)
        own_company_flags = np.hstack(own_company_flags)

        _, _, stats = evaluate(model, class_weight,
                               test_loader, best_thr=best_thr)

        table_5.at[(architecture, horizon, frequency, direction),
                   'F1-score'] = stats['f1'][1]
        table_5.at[(architecture, horizon, frequency,
                    direction), 'AUC'] = stats['auc']

        predictions = pd.DataFrame(
            [own_company_flags,
             family_flags,
             distances,
             stats['predicted_labels'],
             stats['labels'].numpy(),
             stats['predictions'].numpy()],
            index=['own_company_flag',
                   'family_flag',
                   'distance',
                   'prediction',
                   'label',
                   'score']).T

        predictions['best_thr'] = np.exp(best_thr)
        predictions['dataset'] = f'{horizon}_{frequency}_{direction}'
        predictions['architecture'] = architecture
        predictions['dataset_seed'] = args['data_split_seed']
        predictions['seed'] = args['seed']
        prediction_list.append(predictions)

        non_own_companies = evaluate_predictions(
            predictions[predictions.own_company_flag == 0])
        table_8.at[(architecture, horizon, frequency, direction),
                   'F1-score'] = non_own_companies.f1
        table_8.at[(architecture, horizon, frequency, direction),
                   'AUC'] = non_own_companies.auc

        insiders_non_own_companies = evaluate_predictions(
            predictions[(predictions.own_company_flag == 0) &
            (predictions.family_flag == 0)])
        table_9.at[(architecture, horizon, frequency, direction),
                   'F1-score'] = insiders_non_own_companies.f1
        table_9.at[(architecture, horizon, frequency, direction),
                   'AUC'] = insiders_non_own_companies.auc

    combined_predictions = pd.concat(prediction_list, axis=0)
    combined_predictions = combined_predictions[
        combined_predictions.architecture == 'gat']

    # Performances for samples with different distances between the traded and company
    table_10 = pd.DataFrame(index=pd.MultiIndex.from_product(
        [['insider', 'family'], range(4)]))

    for family_flag, investor_type in enumerate(['insider', 'family']):
        for distance in range(5):
            if family_flag:
                if distance == 1:
                    continue
                elif distance > 1:
                    adj_distance = distance - 1
                else:
                    adj_distance = distance
            else:
                if distance < 4:
                    adj_distance = distance
                else:
                    continue
            distance_perofmance = evaluate_predictions(
                combined_predictions[
                    (combined_predictions.distance == distance) &
                    (combined_predictions.family_flag == family_flag)])
            table_10.at[(investor_type, adj_distance),
                        'F1-score'] = distance_perofmance.f1
            table_10.at[(investor_type, adj_distance),
                        'AUC'] = distance_perofmance.auc

    table_5 = table_5.round(2).unstack([1, 2, 3]).stack(
        0).sort_index(ascending=[True, False])
    print('TABLE 5:\n', table_5)

    table_8 = table_8.round(2).unstack([1, 2, 3]).stack(
        0).sort_index(ascending=[True, False])
    print('TABLE 8:\n', table_8)

    table_9 = table_9.round(2).unstack([1, 2, 3]).stack(
        0).sort_index(ascending=[True, False])
    print('TABLE 9:\n', table_9)

    table_10 = table_10.round(2).unstack(1).stack(
        0).sort_index(ascending=[False, False])
    print('TABLE 10:\n', table_10)
