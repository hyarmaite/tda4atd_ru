import itertools
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import gc
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

from stats_count import *
from grab_weights import grab_attention_weights, text_preprocessing

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

model_path = tokenizer_path = "DeepPavlov/rubert-base-cased"

# You can use either standard or fine-tuned BERT. If you want to use fine-tuned BERT to your current task, save the
# model and the tokenizer with the commands tokenizer.save_pretrained(output_dir); 
# bert_classifier.save_pretrained(output_dir) into the same directory and insert the path to it here.

subset = "train_ru"           # .csv file with the texts, for which we count topological features
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

try:
    data = pd.read_csv(input_dir + subset + ".csv").reset_index(drop=True)
except:
    #data = pd.read_csv(input_dir + subset + ".tsv", delimiter="\t")
    data = pd.read_csv(input_dir + subset + ".tsv", delimiter="\t", header=None)
    data.columns = ["0", "labels", "2", "sentence"]

batch_size = 10 # batch size
number_of_batches = ceil(len(data['Text']) / batch_size)
DUMP_SIZE = 100 # number of batches to be dumped
number_of_files = ceil(number_of_batches / DUMP_SIZE)

sentences = list(map(len, map(lambda x: re.sub('\w', ' ', x).split(" "), data['Text'])))
print("Average amount of words in example:", np.mean(sentences))
print("Max. amount of words in example:", np.max(sentences))
print("Min. amount of words in example:", np.min(sentences))

MAX_LEN = max_tokens_amount

tokenizer = pickle.load(open('/home/amshtareva/tokenizer_obj','rb'))
#tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)

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

data['tokenizer_length'] = get_token_length(data['Text'].values)

ntokens_array = data['tokenizer_length'].values
pickle.dump(ntokens_array, open('/home/amshtareva/ntokens_train.obj', 'wb'))

print("Tokens read")

batched_sentences = np.array_split(data['Text'].values, number_of_batches)
adj_matricies = []
adj_filenames = []
number_of_batches = ceil(len(data['Text']) / batch_size)
number_of_files = ceil(number_of_batches / DUMP_SIZE)
number_of_batches_single = ceil(len(data['Text']) / batch_size)
single_set = ceil(number_of_batches_single / DUMP_SIZE)

assert number_of_batches == len(batched_sentences) # sanity check
assert torch.cuda.is_available() == True

dim = 1
lower_bound = 1e-3

mod = 2000
number_of_batches_single = ceil(mod / batch_size)
single_set = ceil(number_of_batches_single / DUMP_SIZE)

features_array = pickle.load(open('/home/amshtareva/features.obj', 'rb'))
component = ceil(len(features_array)/2)
iterv = number_of_batches-component*number_of_batches_single

device='cuda'
#model = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
model = pickle.load(open('/home/amshtareva/model_obj','rb'))
model = nn.DataParallel(model)
model = model.to(device)

print("Entering loop.")

while iterv > 0:

#for component in range(4):
    for i in range(min(number_of_batches_single, iterv)): 
        attention_w = grab_attention_weights(model, tokenizer, batched_sentences[i+component*number_of_batches_single], max_tokens_amount, device)
        # sample X layer X head X n_token X n_token
        adj_matricies.append(attention_w)
        if (i+1) % DUMP_SIZE == 0: # dumping
            print(f'Saving: shape {adj_matricies[0].shape}')
            adj_matricies = np.concatenate(adj_matricies, axis=1)
            print("Concatenated")
            adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
            # Carefully with boundaries
            filename = r_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
            print(f"Saving weights to : {filename}")
            adj_filenames.append(filename)
            np.save(filename, adj_matricies)
            adj_matricies = []
            
    if len(adj_matricies):
        print("Alert!")
        filename = r_file + "_part" + str(ceil(i/DUMP_SIZE)+component*single_set) + "of" + str(number_of_files) + '.npy'
        print(f'Saving: shape {adj_matricies[0].shape}')
        adj_matricies = np.concatenate(adj_matricies, axis=1)
        print("Concatenated")
        adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
        print(f"Saving weights to : {filename}")
        np.save(filename, adj_matricies)
        adj_matricies = []

    print("Results saved.")
    
    # Run computations in completely isolated runtime assuming it helps with memory problem.
    p = subprocess.Popen(["python3", "-u", "ripser_caller.py"], stdout = sys.stdout, stderr = sys.stderr)
    p.wait()

    adj_matricies = []
    iterv = iterv - number_of_batches_single
    component = component + 1

features_array = pickle.load(open('/home/amshtareva/features.obj', 'rb'))
features = np.concatenate(features_array, axis=2)
features.shape

np.save(ripser_file, features)
