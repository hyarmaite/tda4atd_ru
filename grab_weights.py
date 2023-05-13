import os
import re
import numpy as np
import math
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import euclidean, cosine


from tqdm.notebook import tqdm
from transformers import BertTokenizer, BertModel

def grab_attention_weights(model, tokenizer, sentences, MAX_LEN, device='cuda:0'):
    inputs = tokenizer.batch_encode_plus([text_preprocessing(s) for s in sentences],
                                       return_tensors='pt',
                                       add_special_tokens=True,
                                       max_length=MAX_LEN,             # Max length to truncate/pad
                                       pad_to_max_length=True,         # Pad sentence to max length)
                                       truncation=True
                                      )
    input_ids = inputs['input_ids'].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    attention = model(input_ids, attention_mask, token_type_ids)['attentions']
    # layer X sample X head X n_token X n_token
    attention = np.asarray([layer.cpu().detach().numpy() for layer in attention], dtype=np.float16)

    return attention

def grab_embeddings(model, tokenizer, sentences, MAX_LEN, device='cuda:0'):
    inputs = tokenizer.batch_encode_plus([text_preprocessing(s) for s in sentences],
                                       return_tensors='pt',
                                       add_special_tokens=True,
                                       max_length=MAX_LEN,             # Max length to truncate/pad
                                       pad_to_max_length=True,         # Pad sentence to max length)
                                       truncation=True
                                      )
    input_ids = inputs['input_ids'].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    outputs = model(input_ids, attention_mask, token_type_ids)

    result_embeddings = outputs[1][12].cpu().detach().numpy()
    res_emb_euclid = np.asarray([pairwise_distances(item, metric="euclidean") for item in result_embeddings], dtype=np.float16)
    res_emb_spher = np.asarray([np.arccos(1-pairwise_distances(item, metric="cosine")) for item in result_embeddings], dtype=np.float16)

    start_embeddings = outputs[1][0].cpu().detach().numpy()
    start_emb_euclid = np.asarray([pairwise_distances(item, metric="euclidean") for item in start_embeddings], dtype=np.float16)
    start_emb_spher = np.asarray([np.arccos(1-pairwise_distances(item, metric="cosine")) for item in start_embeddings], dtype=np.float16)

    return np.expand_dims(start_emb_euclid, axis=1), np.expand_dims(start_emb_spher, axis=1), np.expand_dims(res_emb_euclid, axis=1), np.expand_dims(res_emb_spher, axis=1)


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
