import itertools
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import gc
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

from stats_count import *
from grab_weights import grab_attention_weights, text_preprocessing, grab_embeddings

import os
import ripser_count
from localfuncs import *

import warnings
from math import ceil

import pickle

import torch
import torch.nn as nn

print("Imports complete.")

warnings.filterwarnings('ignore')

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
# Name of the file for attention matrices weights

stats_file = output_dir + 'features/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers_" + stats_name \
             + "_lists_array_" + str(thrs) + "_thrs_MAX_LEN_" + str(max_tokens_amount) + \
             "_" + model_path.split("/")[-1] + '.npy'

barcodes_file = output_dir + 'barcodes/' + subset  + "_all_heads_" + str(len(layers_of_interest)) + "_layers_MAX_LEN_" + \
             str(max_tokens_amount) + "_" + model_path.split("/")[-1]
# Name of the file for topological features array

ripser_file = output_dir + 'features/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
                 + "_MAX_LEN_" + str(max_tokens_amount) + \
                 "_" + model_path.split("/")[-1] + "_ripser" + '.npy'
ripser_euc_final_file = output_dir + 'features/euc_final' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
                 + "_MAX_LEN_" + str(max_tokens_amount) + \
                 "_" + model_path.split("/")[-1] + "_ripser" + '.npy'
ripser_euc_start_file = output_dir + 'features/euc_start' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
                 + "_MAX_LEN_" + str(max_tokens_amount) + \
                 "_" + model_path.split("/")[-1] + "_ripser" + '.npy'
ripser_sph_final_file = output_dir + 'features/sph_final' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
                 + "_MAX_LEN_" + str(max_tokens_amount) + \
                 "_" + model_path.split("/")[-1] + "_ripser" + '.npy'
ripser_sph_start_file = output_dir + 'features/sph_start' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
                 + "_MAX_LEN_" + str(max_tokens_amount) + \
                 "_" + model_path.split("/")[-1] + "_ripser" + '.npy'

euc_final_emb_file = output_dir + 'final_embeddings/euclidean_distance/' + subset
sph_final_emb_file = output_dir + 'final_embeddings/spherical_distance/' + subset

euc_start_emb_file = output_dir + 'start_embeddings/euclidean_distance/' + subset
sph_start_emb_file = output_dir + 'start_embeddings/spherical_distance/' + subset
# Names of files for distance matrices for different embedding sets

try:
    data = pd.read_csv(input_dir + subset + ".csv").reset_index(drop=True)
except:
    #data = pd.read_csv(input_dir + subset + ".tsv", delimiter="\t")
    data = pd.read_csv(input_dir + subset + ".tsv", delimiter="\t", header=None)
    data.columns = ["0", "labels", "2", "sentence"]

batch_size = 10 # batch size
number_of_batches = ceil(len(data['sentence']) / batch_size)
DUMP_SIZE = 100 # number of batches to be dumped
number_of_files = ceil(number_of_batches / DUMP_SIZE)

sentences = list(map(len, map(lambda x: re.sub('\w', ' ', x).split(" "), data['sentence'])))
print("Average amount of words in example:", np.mean(sentences))
print("Max. amount of words in example:", np.max(sentences))
print("Min. amount of words in example:", np.min(sentences))

MAX_LEN = max_tokens_amount

tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
pickle.dump(tokenizer, open('tokenizer_obj','wb'))

def get_token_length(batch_texts):
    inputs = tokenizer.batch_encode_plus(batch_texts,
       return_tensors='pt',
       add_special_tokens=True,
       max_length=MAX_LEN,             # Max length to truncate/pad
       pad_to_max_length=True,         # Pad sentence to max length
       truncation=True
    )
    inputs = inputs['input_ids'].numpy()
    n_tokens = []
    indexes = np.argwhere(inputs == tokenizer.pad_token_id)
    for i in range(inputs.shape[0]):
        ids = indexes[(indexes == i)[:, 0]]
        if not len(ids):
            n_tokens.append(MAX_LEN)
        else:
            n_tokens.append(ids[0, 1])
    return n_tokens

data['tokenizer_length'] = get_token_length(data['sentence'].values)

ntokens_array = data['tokenizer_length'].values
pickle.dump(ntokens_array, open('ntokens_train.obj', 'wb'))

print("Tokens read")

batched_sentences = np.array_split(data['sentence'].values, number_of_batches)
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
number_of_batches = ceil(len(data['sentence']) / batch_size)
number_of_files = ceil(number_of_batches / DUMP_SIZE)
number_of_batches_single = ceil(len(data['sentence']) / batch_size)
single_set = ceil(number_of_batches_single / DUMP_SIZE)

assert number_of_batches == len(batched_sentences) # sanity check
assert torch.cuda.is_available() == True

dim = 1
lower_bound = 1e-3

mod = 2000
number_of_batches_single = ceil(mod / batch_size)
single_set = ceil(number_of_batches_single / DUMP_SIZE)

try:
    features_array = pickle.load(open('features.obj', 'rb'))
except:
    features_array = []
component = ceil(len(features_array)/2)
iterv = number_of_batches-component*number_of_batches_single

device='cuda'
model = BertForSequenceClassification.from_pretrained(model_path, output_hidden_states=True, output_attentions=True)
pickle.dump(model, open('model_obj','wb'))
model = nn.DataParallel(model)
model = model.to(device)

print("Entering loop.")

while iterv > 0:
#for component in range(4):
    for i in tqdm(range(min(number_of_batches_single, iterv)), desc="Weights calc"):
        attention_w = grab_attention_weights(model, tokenizer, batched_sentences[i+component*number_of_batches_single], max_tokens_amount, device)
        adj_matricies.append(attention_w)

        start_emb_euclid, start_emb_spher, res_emb_euclid, res_emb_spher = grab_embeddings(model, tokenizer, batched_sentences[i+component*number_of_batches_single], max_tokens_amount, device)
        final_euc_matricies.append(res_emb_euclid)
        final_sph_matricies.append(res_emb_spher)
        start_euc_matricies.append(start_emb_euclid)
        start_sph_matricies.append(start_emb_spher)


        if (i+1) % DUMP_SIZE == 0: # dumping
            print(f'Saving: shape {adj_matricies[0].shape}')
            adj_matricies = np.concatenate(adj_matricies, axis=1)
            adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
            # Carefully with boundaries
            filename = r_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
            adj_filenames.append(filename)
            np.save(filename, adj_matricies)
            adj_matricies = []

            filename1 = euc_final_emb_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
            final_euc_filenames.append(filename1)
            np.save(filename1, final_euc_matricies)
            final_euc_matricies = []

            filename2 = sph_final_emb_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
            final_sph_filenames.append(filename2)
            np.save(filename2, final_sph_matricies)
            final_sph_matricies = []

            filename3 = euc_start_emb_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
            start_euc_filenames.append(filename3)
            np.save(filename3, start_euc_matricies)
            start_euc_matricies = []

            filename4 = sph_start_emb_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
            start_sph_filenames.append(filename3)
            np.save(filename4, start_sph_matricies)
            start_sph_matricies = []


    if len(adj_matricies):
        print("Alert!")
        filename = r_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
        print(f'Saving: shape {adj_matricies[0].shape}')
        adj_matricies = np.concatenate(adj_matricies, axis=1)
        adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
        np.save(filename, adj_matricies)
        adj_matricies = []

        filename1 = euc_final_emb_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
        final_euc_filenames.append(filename1)
        np.save(filename1, final_euc_matricies)
        final_euc_matricies = []

        filename2 = sph_final_emb_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
        final_sph_filenames.append(filename2)
        np.save(filename2, final_sph_matricies)
        final_sph_matricies = []

        filename3 = euc_start_emb_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
        start_euc_filenames.append(filename3)
        np.save(filename3, start_euc_matricies)
        start_euc_matricies = []

        filename4 = sph_start_emb_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
        start_sph_filenames.append(filename3)
        np.save(filename4, start_sph_matricies)
        start_sph_matricies = []

    print("Results saved.")
    os.system('ls small_gpt_web/attentions')

    # Run computations in completely isolated runtime assuming it helps with memory problem.
    p = subprocess.Popen(["python3", "-u", "ripser_caller.py"], stdout = sys.stdout, stderr = sys.stderr)
    p.wait()

    adj_matricies = []
    iterv = iterv - number_of_batches_single
    component = component + 1

features_array = pickle.load(open('features.obj', 'rb'))
features = np.concatenate(features_array, axis=2)
features.shape

features_enc_final_array = pickle.load(open('features_enc_final.obj', 'rb'))
features_enc_final = np.concatenate(features_enc_final_array, axis=2)

features_enc_start_array = pickle.load(open('features_enc_start.obj', 'rb'))
features_enc_start = np.concatenate(features_enc_start_array, axis=2)

features_sph_final_array = pickle.load(open('features_sph_final.obj', 'rb'))
features_sph_final = np.concatenate(features_sph_final_array, axis=2)

features_sph_start_array = pickle.load(open('features_sph_start.obj', 'rb'))
features_sph_start = np.concatenate(features_sph_start_array, axis=2)

np.save(ripser_file, features)
np.save(ripser_euc_final_file, features_enc_final)
np.save(ripser_euc_start_file, features_enc_start)
np.save(ripser_sph_final_file, features_sph_final)
np.save(ripser_sph_start_file, features_sph_start)
