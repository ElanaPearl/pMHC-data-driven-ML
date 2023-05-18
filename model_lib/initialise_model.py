import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_lib.bert import BERT
from model_lib.models import *
from model_lib.utils import weights_init
from collections import OrderedDict
# import fb_esm as esm

def get_bert(args, vocab_size):
    bert = BERT(
        vocab_size,
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads,
        max_len=args.seq_len,
        dropout=args.dropout,
        emb_type=args.emb_type,
        activation=args.activation
    )

    return bert


def get_classifier(bert, args, vocab_size, num_classes, device):
    model = PairwiseClassifierModel(
            bert,
            num_classes,
            args.dropout,
            seq_len=args.seq_len,
            esm=False)
    model.apply(weights_init)

    model = model.to(device)

    return model


def initialise_model(args, vocab_size, num_classes, device):
    print("Building BERT model")
    bert = get_bert(args, vocab_size)

    return get_classifier(
            bert,
            args,
            vocab_size,
            num_classes,
            device)


######## Unit Test ########


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class FakeArgs(object):
        """Fake Arg Class"""
        hidden=512 # hidden size of transformer model.
        layers=12  # number of layers of bert.
        attn_heads=8  # number of attention heads in transformer.
        seq_len=1024  # maximum sequence length.
        dropout=0.1  # dropout rate
        emb_type='conv'  # embedding type  ['lookup, 'conv', 'continuous', 'both', 'pair']
        activation='gelu'  # activation function
    sample_args = FakeArgs()

    sample_vocab_size = 29
    sample_num_classes = 1
    sample_device = "cuda:0"

    model, criterion = initialise_model(
        sample_args,
        sample_vocab_size,
        sample_num_classes,
        sample_device,
    )
    print("\nModel and Loss initialised.")

########
