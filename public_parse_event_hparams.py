'''
As found on : https://github.com/tensorflow/tensorboard/issues/3091
'''

import socket
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.plugins.hparams import plugin_data_pb2
from tensorboard.backend.event_processing import event_accumulator


def read_hyper_params(path: str):
    si = summary_iterator(path)

    count = 0
    hyper_param_count = 0
    for event in si:
        for value in event.summary.value:
            count += 1
            proto_bytes = value.metadata.plugin_data.content
            plugin_data = plugin_data_pb2.HParamsPluginData.FromString(
                proto_bytes)
            if plugin_data.HasField("experiment"):
                pass
            elif plugin_data.HasField("session_start_info"):
                hyper_param_count += 1
                hyper_params = dict(plugin_data.session_start_info.hparams)
                for key, value in hyper_params.items():
                    if value.HasField('string_value'):
                        hyper_params[key] = value.string_value
                    elif value.HasField('number_value'):
                        hyper_params[key] = value.number_value
                    else:
                        raise Exception('Unknown hyper parameter type')

                # print(
                #     "Got session start info with concrete hparam values: %r"
                #     % (hyper_params,)
                # )
    if hyper_param_count > 1:
        raise Exception('More hyper_params then expected.')

    return hyper_params


def read_metrics(path):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    measures = {metric: ea.Scalars(metric)[0].value
                for metric in ea.Tags()['scalars']}
    return measures


if __name__ == '__main__':

    computer_name = socket.gethostname()
    if computer_name == 'wks-88866-mac.ad.tuni.fi':

        path = '/Volumes/WD Elements 2621 Media/Insider Influence/results/' \
            'randomized_networks_for_best_dataset_models/Simultaneous_W_Sell/' \
            'random_0/Nov28_05-01-06/1638068475.9151757/' \
            'events.out.tfevents.1638068475.viikunakyyhky.58911.1'
    else:
        path = './results/test_Feb_03_1/1643906572.8939831/events.out.tfevents.1643906572.viikunakyyhky.78357.1'



    metrics = read_metrics(path)

    parameters = read_hyper_params(path)
