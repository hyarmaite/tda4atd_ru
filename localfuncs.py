from collections import defaultdict
import itertools
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import gc
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from multiprocessing import Pool

from stats_count import *
from grab_weights import grab_attention_weights, text_preprocessing

import os
import shutil
import timeit
import ripser_count

import warnings
from math import ceil

import json
import pickle

import torch
import torch.nn as nn

from multiprocessing import Process, Queue

def subprocess_wrap(queue, function, args):
    queue.put(function(*args))
#     print("Putted in Queue")
    queue.close()
    exit()

def get_only_barcodes(adj_matricies, ntokens_array, dim, lower_bound):
    """Get barcodes from adj matricies for each layer, head"""
    barcodes = {}
    layers, heads = range(adj_matricies.shape[1]), range(adj_matricies.shape[2])
    for (layer, head) in itertools.product(layers, heads):
        matricies = adj_matricies[:, layer, head, :, :]
        barcodes[(layer, head)] = ripser_count.get_barcodes(matricies, ntokens_array, dim, lower_bound, (layer, head))
    return barcodes

def format_barcodes(barcodes):
    """Reformat barcodes to json-compatible format"""
    return [{d: b[d].tolist() for d in b} for b in barcodes]

def save_barcodes(barcodes, filename):
    """Save barcodes to file"""
    formatted_barcodes = defaultdict(dict)
    for layer, head in barcodes:
        formatted_barcodes[layer][head] = format_barcodes(barcodes[(layer, head)])
    json.dump(formatted_barcodes, open(filename, 'w'))

def unite_barcodes(barcodes, barcodes_part):
    """Unite 2 barcodes"""
    for (layer, head) in barcodes_part:
        barcodes[(layer, head)].extend(barcodes_part[(layer, head)])
    return barcodes

#def split_matricies_and_lengths(adj_matricies, ntokens, number_of_splits):
#    splitted_ids = np.array_split(np.arange(ntokens.shape[0]), number_of_splits)
#    splitted = [(adj_matricies[ids], ntokens[ids]) for ids in splitted_ids]
#    return splitted

def reformat_barcodes(barcodes):
    """Return barcodes to their original format"""
    formatted_barcodes = []
    for barcode in barcodes:
        formatted_barcode = {}
        for dim in barcode:
            formatted_barcode[int(dim)] = np.asarray(
                [(b, d) for b,d in barcode[dim]], dtype=[('birth', '<f4'), ('death', '<f4')]
            )
        formatted_barcodes.append(formatted_barcode)
    return formatted_barcodes

# What is calculated in "f(v)". You can add any other function from the array with vertex degrees.

def function_for_v(list_of_v_degrees_of_graph):
    return sum(map(lambda x: np.sqrt(x*x), list_of_v_degrees_of_graph))

def split_matricies_and_lengths(adj_matricies, ntokens_array, num_of_workers):
    print(f'SHAPE: {adj_matricies.shape}')
    print(f'SHAPE: {ntokens_array.shape}')
    print(f'Tokens: {ntokens_array}')
    splitted_adj_matricies = np.array_split(adj_matricies, num_of_workers)
    splitted_ntokens = np.array_split(ntokens_array, num_of_workers)
    assert all([len(m)==len(n) for m, n in zip(splitted_adj_matricies, splitted_ntokens)]), "Split is not valid!"
    return zip(splitted_adj_matricies, splitted_ntokens)

def matrix_distance(matricies, template, broadcast=True):
    """
    Calculates the distance between the list of matricies and the template matrix.
    Args:

    -- matricies: np.array of shape (n_matricies, dim, dim)
    -- template: np.array of shape (dim, dim) if broadcast else (n_matricies, dim, dim)

    Returns:
    -- diff: np.array of shape (n_matricies, )
    """
    diff = np.linalg.norm(matricies-template, ord='fro', axis=(1, 2))
    div = np.linalg.norm(matricies, ord='fro', axis=(1, 2))**2
    if broadcast:
        div += np.linalg.norm(template, ord='fro')**2
    else:
        div += np.linalg.norm(template, ord='fro', axis=(1, 2))**2
    return diff/np.sqrt(div)

def attention_to_self(matricies):
    """
    Calculates the distance between input matricies and identity matrix,
    which representes the attention to the same token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.eye(n)
    return matrix_distance(matricies, template_matrix)

def attention_to_next_token(matricies):
    """
    Calculates the distance between input and E=(i, i+1) matrix,
    which representes the attention to the next token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=1, dtype=matricies.dtype), k=1)
    return matrix_distance(matricies, template_matrix)

def attention_to_prev_token(matricies):
    """
    Calculates the distance between input and E=(i+1, i) matrix,
    which representes the attention to the previous token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=-1, dtype=matricies.dtype), k=-1)
    return matrix_distance(matricies, template_matrix)

def attention_to_beginning(matricies):
    """
    Calculates the distance between input and E=(i+1, i) matrix,
    which representes the attention to [CLS] token (beginning).
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.zeros((n, n))
    template_matrix[:, 0] = 1.0
    return matrix_distance(matricies, template_matrix)

def attention_to_ids(matricies, list_of_ids, token_id):
    """
    Calculates the distance between input and ids matrix,
    which representes the attention to some particular tokens,
    which ids are in the `list_of_ids` (commas, periods, separators).
    """

    batch_size, n, m = matricies.shape
    EPS = 1e-7
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
#     assert len(list_of_ids) == batch_size, f"List of ids length doesn't match the dimension of the matrix"
    template_matrix = np.zeros_like(matricies)
    ids = np.argwhere(list_of_ids == token_id)
    if len(ids):
        batch_ids, row_ids = zip(*ids)
        template_matrix[np.array(batch_ids), :, np.array(row_ids)] = 1.0
        template_matrix /= (np.sum(template_matrix, axis=-1, keepdims=True) + EPS)
    return matrix_distance(matricies, template_matrix, broadcast=False)

def count_template_features(matricies, feature_list=['self', 'beginning', 'prev', 'next', 'comma', 'dot'], ids=None):
    features = []
    comma_id = 1010
    dot_id = 1012
    for feature in feature_list:
        if feature == 'self':
            features.append(attention_to_self(matricies))
        elif feature == 'beginning':
            features.append(attention_to_beginning(matricies))
        elif feature == 'prev':
            features.append(attention_to_prev_token(matricies))
        elif feature == 'next':
            features.append(attention_to_next_token(matricies))
        elif feature == 'comma':
            features.append(attention_to_ids(matricies, ids, comma_id))
        elif feature == 'dot':
            features.append(attention_to_ids(matricies, ids, dot_id))
    return np.array(features)

def calculate_features_t(adj_matricies, template_features, ids=None):
    """Calculate template features for adj_matricies"""
    features = []
    for layer in range(adj_matricies.shape[1]):
        features.append([])
        for head in range(adj_matricies.shape[2]):
            matricies = adj_matricies[:, layer, head, :, :]
            lh_features = count_template_features(matricies, template_features, ids) # samples X n_features
            features[-1].append(lh_features)
    return np.asarray(features) # layer X head X n_features X samples

# '.' id 1012
# ',' id 1010
def get_list_of_ids(sentences, tokenizer):
    inputs = tokenizer.batch_encode_plus([text_preprocessing(s) for s in sentences],
                                       add_special_tokens=True,
                                       max_length=MAX_LEN,             # Max length to truncate/pad
                                       pad_to_max_length=True,         # Pad sentence to max length)
                                       truncation=True
                                      )
    return np.array(inputs['input_ids'])
