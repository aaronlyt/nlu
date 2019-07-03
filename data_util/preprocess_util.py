"""
split_document read_document_seqs have problem
"""
import os
import sys
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
from data_util.make_features import extract_raw_data


def to_train_id(seqs, v):
    results = []
    for seq in seqs:
        results.append([v[ch] for ch in seq])
    return results


def read_embedding(emb_path, vocab):
    """
    read embedding
    """
    emb_matrix = [None] * len(vocab)
    emb_dimension = None
    unk_embedding = None
    hit_count = 0
    with open(emb_path, 'r') as reader:
        for line in reader:
            line_vals = line.split()
            if len(line_vals) == 2:
                emb_dimension = line_vals[1]
            else:
                word = line_vals[0]
                embedding = [float(val) for val in line_vals[1:]]
                emb_dimension = len(embedding)
                if word in vocab:
                    emb_matrix[vocab[word]] = embedding
                    hit_count += 1
                if word == 'UNK':
                    unk_embedding = embedding
    non_hit_count = 0
    for idx, emb in enumerate(emb_matrix):
        if not emb:
            emb_matrix[idx] = unk_embedding
            non_hit_count += 1
    emb_matrix[0] = [0] * emb_dimension
    print("---word in embedding and in vocab hit count---", hit_count)
    print("---word not in embedding and intialize count---", \
        len(vocab) - hit_count, non_hit_count)
    return emb_matrix, emb_dimension


def one_hot(data, entity_num):
    """
    """
    new_data = []
    for (batch_x, batch_y) in data:
        batch_onehot_y = []
        for y in batch_y:
            one_hot_y = np.zeros((len(y), entity_num))
            one_hot_y[np.arange(len(y)), y] = 1
            batch_onehot_y.append(one_hot_y)
        batch_onehot_y = np.array(batch_onehot_y)
        assert(np.equal(np.argmax(batch_onehot_y, -1), batch_y).all())
        new_data.append((batch_x, batch_onehot_y))
    return new_data


if __name__ == '__main__':
    vocab = json.load(open('../dataset/char2id', 'r'))
    l_vocab = json.load(open('../dataset/label2id', 'r'))

    datasetpath = '../dataset/emrexample.train'
    max_sent_len = 96
    