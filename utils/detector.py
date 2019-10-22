import argparse
from collections import ChainMap
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn as nn

from . import model as mod


class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(
                    value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value,
                                    nargs="+", type=type(value[0]))
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


class Detector:
    def __init__(self, dataFolder='tmp/hi_koov_demo'):
        global_config = dict(cache_size=0, model='res8')
        builder = ConfigBuilder(
            mod.find_config(global_config['model']),
            mod.SpeechDataset.default_config(),
            global_config)
        parser = builder.build_argparse()
        self.config = builder.config_from_argparse(parser)
        self.config['model_class'] = mod.find_model(global_config['model'])
        self.config['data_folder'] = dataFolder
        self.config['wanted_words'] = ['hi_koov']
        self.config['noise_prob'] = 0
        self.config['n_labels'] = 3
        self.config['train_pct'] = 0
        self.config['dev_pct'] = 0
        self.config['test_pct'] = 100
        self.config['weights'] = 'model/model-res8-mfcc-91.pt'

        self.model = self.config['model_class'](self.config)
        self.model.load(self.config['weights'])
        self.model.eval()

        # self.quantized_model = torch.quantization.quantize_dynamic(
        #     self.model, {nn.Conv2d, nn.AvgPool2d, nn.BatchNorm2d, nn.Linear}, dtype=torch.qint8
        # )

    def evaluate(self):
        _, _, testSet = mod.SpeechDataset.splits(self.config)
        testLoader = data.DataLoader(testSet, batch_size=1, collate_fn=testSet.collate_fn)

        for x, _ in testLoader:
            x = Variable(x, requires_grad=False)
            scores = self.model(x)
            # scores = self.quantized_model(x)
            return torch.max(scores, 1)[1].view(1).data.numpy()[0]

    def getDataFolder(self):
        return self.config['data_folder']
