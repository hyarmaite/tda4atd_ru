from collections import defaultdict
import itertools

import GPUtil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from multiprocessing import Pool

from stats_count import *
from grab_weights import grab_attention_weights, text_preprocessing

import os
import shutil
import ripser_count
from localfuncs import *

from math import ceil

import json
import pickle

import torch
import torch.nn as nn

from multiprocessing import Process, Queue

print("In subprocess (next two attention maps)")
GPUtil.showUtilization()

np.random.seed(42) # For reproducibility.

max_tokens_amount  = 128 # The number of tokens to which the tokenized text is truncated / padded.
stats_cap          = 500 # Max value that the feature can take. Is NOT applicable to Betty numbers.
    
layers_of_interest = [i for i in range(12)]  # Layers for which attention matrices and features on them are 
                                             # calculated. For calculating features on all layers, leave it be
                                             # [i for i in range(12)].
stats_name = "s_e_v_c_b0b1" # The set of topological features that will be count (see explanation below)

thresholds_array = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75] # The set of thresholds
thrs = len(thresholds_array)                           # ("t" in the paper)

model_path = tokenizer_path = "bert-base-uncased"  

# You can use either standard or fine-tuned BERT. If you want to use fine-tuned BERT to your current task, save the
# model and the tokenizer with the commands tokenizer.save_pretrained(output_dir); 
# bert_classifier.save_pretrained(output_dir) into the same directory and insert the path to it here.

subset = "train_ru"           # .csv file with the texts, for which we count topological features
input_dir = "small_gpt_web/"  # Name of the directory with .csv file
output_dir = "small_gpt_web/" # Name of the directory with calculations results

prefix = output_dir + subset

r_file     = output_dir + 'attentions/' + subset  + "_all_heads_" + str(len(layers_of_interest)) + "_layers_MAX_LEN_" + \
             str(max_tokens_amount) + "_" + model_path.split("/")[-1]

barcodes_file = output_dir + 'barcodes/' + subset  + "_all_heads_" + str(len(layers_of_interest)) + "_layers_MAX_LEN_" + \
             str(max_tokens_amount) + "_" + model_path.split("/")[-1]

ripser_file = output_dir + 'features/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
                 + "_MAX_LEN_" + str(max_tokens_amount) + \
                 "_" + model_path.split("/")[-1] + "_ripser" + '.npy'

ntokens_array = pickle.load(open('/home/amshtareva/ntokens_train.obj', 'rb'))

print("Tokens read")

adj_matricies = []
adj_filenames = []
batch_size = 10 # batch size
number_of_batches = ceil(len(ntokens_array) / batch_size)
DUMP_SIZE = 100 #umber of batches to be dumped
number_of_files = ceil(number_of_batches / DUMP_SIZE)
number_of_batches_single = ceil(len(ntokens_array) / batch_size)
single_set = ceil(number_of_batches_single / DUMP_SIZE)

#assert torch.cuda.is_available() == True

dim = 1
lower_bound = 1e-3

features_array = pickle.load(open('/home/amshtareva/features.obj', 'rb'))

mod = 2000
number_of_batches_single = ceil(mod / batch_size)
single_set = ceil(number_of_batches_single / DUMP_SIZE)

ripser_feature_names=[
    'h0_s', 
    'h0_e',
    'h0_t_d', 
    'h0_n_d_m_t0.75',
    'h0_n_d_m_t0.5',
    'h0_n_d_l_t0.25',
    'h1_t_b',
    'h1_n_b_m_t0.25',
    'h1_n_b_l_t0.95', 
    'h1_n_b_l_t0.70',  
    'h1_s',
    'h1_e',
    'h1_v',
    'h1_nb'
]
feature_list = ['self', 'beginning', 'prev', 'next', 'comma', 'dot']

features_array = pickle.load(open('/home/amshtareva/features.obj', 'rb'))
component = ceil(len(features_array)/2)
iterv = number_of_batches-component*number_of_batches_single

queue = Queue()
num_of_workers = 60
pool = Pool(num_of_workers)

adj_filenames = [
    output_dir + 'attentions/' + filename 
    for filename in os.listdir(output_dir + 'attentions/') if r_file in (output_dir + 'attentions/' + filename)
]

# sorted by part number
adj_filenames = sorted(adj_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip())) 
print(adj_filenames)

for i, filename in enumerate(adj_filenames):
    adj_matricies = np.load(filename, allow_pickle=True)
    ntokens = ntokens_array[(i+component*single_set)*batch_size*DUMP_SIZE : (i+component*single_set+1)*batch_size*DUMP_SIZE]
    splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers) # 2 or 20.

    barcodes = defaultdict(list)
    print(f"Matricies loaded from: {filename}")
    for matricies, ntokens in splitted:
        p = Process(
            target=subprocess_wrap,
            args=(
                queue,
                get_only_barcodes,
                (matricies, ntokens, dim, lower_bound)
            )
        )
        GPUtil.showUtilization()
        p.start()
        barcodes_part = queue.get() # block until putted and get barcodes from the queue
        #print("Features got.")
        p.join() # release resources
        #print("The process is joined.")
        p.close() # releasing resources of ripser
        #print("The proccess is closed.")

        barcodes = unite_barcodes(barcodes, barcodes_part)
    part = filename.split('_')[-1].split('.')[0]
    save_barcodes(barcodes, barcodes_file + '_' + part + '.json')
    
bar_filenames = [
    output_dir + 'barcodes/' + filename
    for filename in os.listdir(output_dir + 'barcodes/') if r_file.split('/')[-1] == filename.split('_part')[0]
]
bar_filenames = sorted(bar_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip())) 
print(bar_filenames)

for filename in bar_filenames:
    barcodes = json.load(open(filename))
    print(f"Barcodes loaded from: {filename}", flush=True)
    features_part = []
    for layer in barcodes:
        features_layer = []
        for head in barcodes[layer]:
            ref_barcodes = reformat_barcodes(barcodes[layer][head])
            features = ripser_count.count_ripser_features(ref_barcodes, ripser_feature_names)
            features_layer.append(features)
        features_part.append(features_layer)
    features_array.append(np.asarray(features_part))
    
for filename in os.listdir(output_dir + 'attentions/'):
    filepath = os.path.join(output_dir + 'attentions/', filename)
    try:
        shutil.rmtree(filepath)
    except OSError:
        os.remove(filepath)
    
for filename in os.listdir(output_dir + 'barcodes/'):
    filepath = os.path.join(output_dir + 'barcodes/', filename)
    try:
        shutil.rmtree(filepath)
    except OSError:
        os.remove(filepath)

print(f"DUMPING: {str(len(features_array))}")
pickle.dump(features_array, open('/home/amshtareva/features.obj', 'wb'))