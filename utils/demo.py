from collections import ChainMap
import argparse
import os
import random
import sys

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from . import model as mod
from .manage_audio import AudioPreprocessor

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def evaluate(config):
    _, _, test_set = mod.SpeechDataset.splits(config)
    test_loader = data.DataLoader(
        test_set,
        batch_size=1,
        collate_fn=test_set.collate_fn)

    model = config['model_class'](config)
    model.load(config['weights'])
    model.eval()

    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        scores = model(model_in)
        return torch.max(scores, 1)[1].view(1).data.numpy()[0]

    raise Exception('No test data')

def process():
    global_config = dict(cache_size=0, model='res8')
    builder = ConfigBuilder(
        mod.find_config(global_config['model']),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    config = builder.config_from_argparse(parser)
    config['model_class'] = mod.find_model(global_config['model'])
    config['data_folder'] = 'tmp/hi_koov_test'
    config['wanted_words'] = ['hi_koov']
    config['n_labels'] = 3
    config["train_pct"] = 0
    config["dev_pct"] = 0
    config['test_pct'] = 100
    config['weights'] = 'hi_koov_weights.pt'
    return evaluate(config)

def main():
    result = process()
    if result == 2:
        print('Hi KOOV!')
    else:
        print('not matched')

if __name__ == "__main__":
    main()
