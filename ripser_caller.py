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
import sys
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

subset = "valid_5k"           # .csv file with the texts, for which we count topological features
input_dir = "small_gpt_web/"  # Name of the directory with .csv file
output_dir = "small_gpt_web/" # Name of the directory with calculations results

prefix = output_dir + subset

r_file     = output_dir + 'attentions/' + subset  + "_all_heads_" + str(len(layers_of_interest)) + "_layers_MAX_LEN_" + \
             str(max_tokens_amount) + "_" + model_path.split("/")[-1]
euc_final_emb_file = output_dir + 'final_embeddings/euclidean_distance/' + subset
sph_final_emb_file = output_dir + 'final_embeddings/spherical_distance/' + subset

euc_start_emb_file = output_dir + 'start_embeddings/euclidean_distance/' + subset
sph_start_emb_file = output_dir + 'start_embeddings/spherical_distance/' + subset

barcodes_file = output_dir + 'barcodes/' + subset  + "_all_heads_" + str(len(layers_of_interest)) + "_layers_MAX_LEN_" + \
             str(max_tokens_amount) + "_" + model_path.split("/")[-1]
barcodes_euc_final_file = output_dir + 'barcodes/euc_final/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
                 + "_MAX_LEN_" + str(max_tokens_amount) + \
                 "_" + model_path.split("/")[-1]
barcodes_euc_start_file = output_dir + 'barcodes/euc_start/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
                 + "_MAX_LEN_" + str(max_tokens_amount) + \
                 "_" + model_path.split("/")[-1]
barcodes_sph_final_file = output_dir + 'barcodes/sph_final/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
                 + "_MAX_LEN_" + str(max_tokens_amount) + \
                 "_" + model_path.split("/")[-1]
barcodes_sph_start_file = output_dir + 'barcodes/sph_start/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
                 + "_MAX_LEN_" + str(max_tokens_amount) + \
                 "_" + model_path.split("/")[-1]

ripser_file = output_dir + 'features/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
                 + "_MAX_LEN_" + str(max_tokens_amount) + \
                 "_" + model_path.split("/")[-1] + "_ripser" + '.npy'

ntokens_array = pickle.load(open('ntokens_train.obj', 'rb'))

print("Tokens read")

adj_matricies = []
adj_filenames = []
final_euc_matricies = []
final_euc_filenames = []
final_sph_matricies = []
final_sph_filenames = []
start_euc_matricies = []
start_euc_filenames = []
start_sph_matricies = []
start_sph_filenames = []
batch_size = 10 # batch size
number_of_batches = ceil(len(ntokens_array) / batch_size)
DUMP_SIZE = 100 #umber of batches to be dumped
number_of_files = ceil(number_of_batches / DUMP_SIZE)
number_of_batches_single = ceil(len(ntokens_array) / batch_size)
single_set = ceil(number_of_batches_single / DUMP_SIZE)

#assert torch.cuda.is_available() == True

dim = 1
lower_bound = 1e-3

mod = 2000
number_of_batches_single = ceil(mod / batch_size)
single_set = ceil(number_of_batches_single / DUMP_SIZE)

hfeat = []
for i in range (2,10):
     hfeat.append("h" + str(i) + "_t_b")
     hfeat.append("h" + str(i) + "_n_b_m_t0.1")
     hfeat.append("h" + str(i) + "_n_b_l_t0.2")
     hfeat.append("h" + str(i) + "_n_b_l_t0.3")
     hfeat.append("h" + str(i) + "_n_b_l_t0.4")
     hfeat.append("h" + str(i) + "_n_b_l_t0.5")
     hfeat.append("h" + str(i) + "_n_b_l_t0.6")
     hfeat.append("h" + str(i) + "_n_b_l_t0.7")
     hfeat.append("h" + str(i) + "_n_b_l_t0.8")
     hfeat.append("h" + str(i) + "_n_b_l_t0.9")
     hfeat.append("h" + str(i) + "_s")
     hfeat.append("h" + str(i) + "_e")
     hfeat.append("h" + str(i) + "_v")
     hfeat.append("h" + str(i) + "_nb")

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
ripser_all_feature_names = ripser_feature_names + hfeat
feature_list = ['self', 'beginning', 'prev', 'next', 'comma', 'dot']

try:
    features_array = pickle.load(open('features.obj', 'rb'))
    features = np.concatenate(features_array, axis=2)
except:
    pickle.dump([], open('features.obj', 'wb'))
    features_array = []
    features = np.array([])

try:
    features_enc_final_array = pickle.load(open('features_enc_final.obj', 'rb'))
    features_enc_final = np.concatenate(features_enc_final_array, axis=2)
except:
    pickle.dump([], open('features_enc_final.obj', 'wb'))
    features_enc_final_array = []
    features_enc_final = np.array([])
try:
    features_enc_start_array = pickle.load(open('features_enc_start.obj', 'rb'))
    features_enc_start = np.concatenate(features_enc_start_array, axis=2)
except:
    pickle.dump([], open('features_enc_start.obj', 'wb'))
    features_enc_start_array = []
    features_enc_start = np.array([])

try:
    features_sph_final_array = pickle.load(open('features_sph_final.obj', 'rb'))
    features_sph_final = np.concatenate(features_sph_final_array, axis=2)
except:
    pickle.dump([], open('features_sph_final.obj', 'wb'))
    features_sph_final_array = []
    features_sph_final = np.array([])
try:
    features_sph_start_array = pickle.load(open('features_sph_start.obj', 'rb'))
    features_sph_start = np.concatenate(features_sph_start_array, axis=2)
except:
    pickle.dump(np.array([]), open('features_sph_start.obj', 'wb'))
    features_sph_start_array = []
    features_sph_start = np.array([])

component = ceil(len(features_array)/2)
iterv = number_of_batches-component*number_of_batches_single

queue = Queue()
num_of_workers = 60
pool = Pool(num_of_workers)

print(os.listdir(output_dir + 'attentions/'))
print(r_file)
adj_filenames = [
    output_dir + 'attentions/' + filename
    for filename in os.listdir(output_dir + 'attentions/') if r_file in (output_dir + 'attentions/' + filename)
]
print(adj_filenames)

# sorted by part number
adj_filenames = sorted(adj_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))
print(adj_filenames)

euc_final_filenames = [
    output_dir + 'final_embeddings/euclidean_distance/' + filename
    for filename in os.listdir(output_dir + 'final_embeddings/euclidean_distance/') if euc_final_emb_file in (output_dir + 'final_embeddings/euclidean_distance/' + filename)
]

# sorted by part number
euc_final_filenames = sorted(euc_final_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))

euc_start_filenames = [
    output_dir + 'start_embeddings/euclidean_distance/' + filename
    for filename in os.listdir(output_dir + 'start_embeddings/euclidean_distance/') if euc_start_emb_file in (output_dir + 'start_embeddings/euclidean_distance/' + filename)
]

# sorted by part number
euc_start_filenames = sorted(euc_start_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))

sph_final_filenames = [
    output_dir + 'final_embeddings/euclidean_distance/' + filename
    for filename in os.listdir(output_dir + 'final_embeddings/spherical_distance/') if sph_final_emb_file in (output_dir + 'final_embeddings/spherical_distance/' + filename)
]

# sorted by part number
sph_final_filenames = sorted(sph_final_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))

sph_start_filenames = [
    output_dir + 'start_embeddings/euclidean_distance/' + filename
    for filename in os.listdir(output_dir + 'start_embeddings/spherical_distance/') if sph_start_emb_file in (output_dir + 'start_embeddings/spherical_distance/' + filename)
]

# sorted by part number
sph_start_filenames = sorted(sph_start_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))

for i, filenames in enumerate(zip(adj_filenames, euc_final_filenames, euc_start_filenames, sph_start_filenames, euc_start_filenames)):
    adj, euc_final, euc_start, sph_start, sph_final = filenames
    adj_matricies = np.load(adj, allow_pickle=True)
    euc_start_matricies = np.load(euc_start, allow_pickle=True)
    euc_final_matricies = np.load(euc_final, allow_pickle=True)
    sph_final_matricies = np.load(sph_final, allow_pickle=True)
    sph_start_matricies = np.load(sph_start, allow_pickle=True)

    for filename, file, matrices in [ (adj, barcodes_file, adj_matricies)
                                    , (euc_final, barcodes_euc_final_file, euc_final_matricies)
                                    , (euc_start, barcodes_euc_start_file, euc_start_matricies)
                                    , (sph_final, barcodes_sph_final_file, sph_final_matricies)
                                    , (sph_start, barcodes_sph_start_file, sph_start_matricies)
                                    ]:
        ntokens = ntokens_array[(i+component*single_set)*batch_size*DUMP_SIZE : (i+component*single_set+1)*batch_size*DUMP_SIZE]
        splitted = split_matricies_and_lengths(matrices, ntokens, num_of_workers) # 2 or 20.
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
            print(matrices.shape)
            sys.stdout.flush()
            p.start()
            barcodes_part = queue.get() # block until putted and get barcodes from the queue
            #print("Features got.")
            p.join() # release resources
            #print("The process is joined.")
            p.close() # releasing resources of ripser
            #print("The proccess is closed.")

            barcodes = unite_barcodes(barcodes, barcodes_part)
        part = filename.split('_')[-1].split('.')[0]
        save_barcodes(barcodes, file + '_' + part + '.json')

bar_filenames = [
    output_dir + 'barcodes/' + filename
    for filename in os.listdir(output_dir + 'barcodes/') if r_file.split('/')[-1] == filename.split('_part')[0]
]
bar_filenames = sorted(bar_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))
print(bar_filenames)

print(os.listdir(output_dir + 'barcodes/euc_final/'))
print(r_file.split('/')[-1])
print(filename.split('_part')[0])
bar_filenames_euc_final = [
    output_dir + 'barcodes/euc_final/' + filename
    for filename in os.listdir(output_dir + 'barcodes/euc_final/') if r_file.split('/')[-1] == filename.split('_part')[0]
]
print(bar_filenames_euc_final)
bar_filenames_euc_final = sorted(bar_filenames_euc_final, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))

bar_filenames_euc_start = [
    output_dir + 'barcodes/euc_start/' + filename
    for filename in os.listdir(output_dir + 'barcodes/euc_start/') if r_file.split('/')[-1] == filename.split('_part')[0]
]
bar_filenames_euc_start = sorted(bar_filenames_euc_start, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))

bar_filenames_sph_final = [
    output_dir + 'barcodes/sph_final/' + filename
    for filename in os.listdir(output_dir + 'barcodes/sph_final/') if r_file.split('/')[-1] == filename.split('_part')[0]
]
bar_filenames_sph_final = sorted(bar_filenames_sph_final, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))

bar_filenames_sph_start = [
    output_dir + 'barcodes/sph_start/' + filename
    for filename in os.listdir(output_dir + 'barcodes/sph_start/') if r_file.split('/')[-1] == filename.split('_part')[0]
]
bar_filenames_sph_start = sorted(bar_filenames_sph_start, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))

for filename, file_euc_final, file_euc_start, file_sph_final, file_sph_start in zip ( bar_filenames
                                                                                    , bar_filenames_euc_final
                                                                                    , bar_filenames_euc_start
                                                                                    , bar_filenames_sph_final
                                                                                    , bar_filenames_sph_start
                                                                                    ):
    barcodes_att = json.load(open(filename))
    barcodes_euc_final = json.load(open(file_euc_final))
    barcodes_euc_start = json.load(open(file_euc_start))
    barcodes_sph_final = json.load(open(file_sph_final))
    barcodes_sph_start = json.load(open(file_sph_start))
    print(f"Barcodes loaded", flush=True)

    for barcodes, array in [ (barcodes_att, features_array)
                           , (barcodes_euc_final, features_enc_final_array)
                           , (barcodes_euc_start, features_enc_start_array)
                           , (barcodes_sph_final, features_sph_final_array)
                           , (barcodes_sph_start, features_sph_start_array)
                           ]:
        features_part = []
        for layer in barcodes:
            features_layer = []
            for head in barcodes[layer]:
                ref_barcodes = reformat_barcodes(barcodes[layer][head])
                features = ripser_count.count_ripser_features(ref_barcodes, ripser_feature_names)
                features_layer.append(features)
            features_part.append(features_layer)
        print(len(features_layer))
        array.append(np.asarray(features_part))

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

os.system('mkdir small_gpt_web/barcodes/euc_final')
os.system('mkdir small_gpt_web/barcodes/euc_start')
os.system('mkdir small_gpt_web/barcodes/sph_final')
os.system('mkdir small_gpt_web/barcodes/sph_start')

print(f"DUMPING: {str(len(features_array))}")
pickle.dump(features_array, open('features.obj', 'wb'))
pickle.dump(features_enc_final_array, open('features_enc_final.obj', 'wb'))
pickle.dump(features_enc_start_array, open('features_enc_start.obj', 'wb'))
pickle.dump(features_sph_final_array, open('features_sph_final.obj', 'wb'))
pickle.dump(features_sph_start_array, open('features_sph_start.obj', 'wb'))
