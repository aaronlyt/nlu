"""
split_document read_document_seqs have problem
"""
import os
import sys
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def to_train_id(seqs, v):
    results = []
    for seq in seqs:
        results.append([v[ch] for ch in seq])
    return results


def read_data_split(dataset_path, vocab, l_vocab, \
    max_sent_len, data_process, min_sent_len=None):
    """
    """
    dataset_txts = []
    dataset_labels = []
    txt_seqs, label_seqs = read_documents_seq(dataset_path)
    for idx, txt_seq in enumerate(txt_seqs):
        if data_process == 'simple':
            split_txts, split_labels = split_document_simple(\
                txt_seq, label_seqs[idx], max_sent_len)
        elif data_process == 'splitters_split':
            split_txts, split_labels = split_document_splitters(\
                txt_seq, label_seqs[idx], max_sent_len, \
                    min_sent_len)
        split_txts = to_train_id(split_txts, vocab)
        split_labels = to_train_id(split_labels, l_vocab)
        dataset_txts.extend(split_txts)
        dataset_labels.extend(split_labels)
    print('---split sentence count---', len(dataset_txts))    
    return list(zip(dataset_txts, dataset_labels))


def split_document_simple(txt_seqs, label_seqs, max_sent_len):
    """
    not consider the dot, the context
    @param txt_seqs: list of char
    @param max_sent_len: max sentence length
    return:
        return list of list of pair(word, label)
    """
    total_len = len(txt_seqs)
    if total_len <= max_sent_len:
        return [txt_seqs], [label_seqs]

    sent_num = total_len // max_sent_len
    if (total_len - sent_num * max_sent_len) > 0:
        sent_num += 1

    split_txt_seqs = []
    split_label_seqs = []
    for sent_idx in range(sent_num):
        start_idx = sent_idx * max_sent_len
        end_idx = (sent_idx + 1) * max_sent_len
        split_txt_seqs.append(txt_seqs[start_idx : end_idx])
        split_label_seqs.append(label_seqs[start_idx : end_idx])
    
    assert(len(txt_seqs) == sum([len(seq) for seq in split_txt_seqs]))
    return split_txt_seqs, split_label_seqs


def split_document_splitters(txt_seqs, label_seqs, max_sent_len, \
    min_sent_len=None):
    """
    txt_seqs: list of char
    real_max_sen may (max_sent_len + 2 * min_sent_len)
    split document with spiltters, and no more then max_sent_len
    """
    total_len = len(txt_seqs)
    if total_len <= max_sent_len:
        return [txt_seqs], [label_seqs]
    
    txt_seqs_splitter = []
    label_seqs_splitter = []
    splitters = ['。', ',', '，', ';', '；']
    idx = 0
    while idx < len(txt_seqs):
        start_idx = idx
        # choose max_sent_len words, idx bigger 1
        idx += max_sent_len
        if idx >= len(txt_seqs) or txt_seqs[idx - 1] in splitters:
             txt_seqs_splitter.append(txt_seqs[start_idx : idx])
             label_seqs_splitter.append(label_seqs[start_idx : idx])
             continue
        # backtracking, two cases: no splitters, and have splitters
        idx -= 1
        while idx > start_idx and txt_seqs[idx] not in splitters:
            idx -= 1
        # no splitters
        if idx == start_idx:
            # TODO may take a look at the right
            txt_seqs_splitter.append(txt_seqs[start_idx : (start_idx + max_sent_len)])
            label_seqs_splitter.append(label_seqs[start_idx : (start_idx + max_sent_len)])
            idx += (max_sent_len - 1)
        else:
            txt_seqs_splitter.append(txt_seqs[start_idx : (idx + 1)])
            label_seqs_splitter.append(label_seqs[start_idx : (idx + 1)])
        idx += 1
    
    #post process
    if min_sent_len:
        idx = 0
        while idx < len(txt_seqs_splitter):
            seq = txt_seqs_splitter[idx]
            labels = label_seqs_splitter[idx]
            if len(seq) < min_sent_len and idx ==0:
                txt_seqs_splitter[idx + 1] = seq + txt_seqs_splitter[idx + 1]
                txt_seqs_splitter.remove(txt_seqs_splitter[idx])
                label_seqs_splitter[idx + 1] = labels + label_seqs_splitter[idx + 1]
                label_seqs_splitter.remove(label_seqs_splitter[idx])
            elif len(seq) < min_sent_len and idx !=0:
                txt_seqs_splitter[idx - 1] = seq + txt_seqs_splitter[idx - 1]
                txt_seqs_splitter.remove(txt_seqs_splitter[idx])
                label_seqs_splitter[idx - 1] = labels + label_seqs_splitter[idx - 1]
                label_seqs_splitter.remove(label_seqs_splitter[idx])
            else:
                idx += 1

    assert(len(txt_seqs) == sum([len(seq) for seq in txt_seqs_splitter]))
    assert(len(label_seqs) == sum([len(seq) for seq in label_seqs_splitter]))
    assert(np.array([len(l_seq) == len(txt_seq) for l_seq, txt_seq in \
        zip(label_seqs_splitter, txt_seqs_splitter)]).all())
    return txt_seqs_splitter, label_seqs_splitter


def read_documents_seq(data_path):
    """
    word label sequence
    """
    txt_seqs = []
    label_seqs = []

    with open(data_path, 'r') as reader:
        txts = []
        labels = []
        for line in reader:
            if not line.strip():
                # end with empty line
                if len(txts) != 0:
                    txt_seqs.append(txts)
                    label_seqs.append(labels)
                    txts = []
                    labels = []
                continue
            line_str = line.strip().split()
            txts.append(line_str[0])
            label = line_str[1]
            labels.append(label)
        # not end with empty line
        if len(txts) > 0:
            txt_seqs.append(txts)
            label_seqs.append(labels)
    print('----sequence length---', len(txt_seqs))
    return txt_seqs, label_seqs


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

    #dataset = \
    #    read_data_split(datasetpath, vocab, l_vocab, max_sent_len)

    txt_seqs, label_seqs = read_documents_seq(datasetpath)
    sentence_count = 0
    new_txt_seqs = []
    end_with_splitters_count = 0
    splitters = ['。', ',', '，', ';', '；']
    for idx, txt_seq in enumerate(txt_seqs):
        #txt_seqs_splitter, label_seqs_splitter = \
        #    split_document_simple(txt_seq, label_seqs[idx], max_sent_len)
        txt_seqs_splitter, label_seqs_splitter = \
            split_document_splitters(txt_seq, label_seqs[idx], max_sent_len, 32)
        end_with_splitters_count += sum([1 for seq in txt_seqs_splitter \
            if seq[-1] in splitters])
        new_txt_seqs.extend(txt_seqs_splitter)
        sentence_count += len(txt_seqs_splitter)
    print('---total sentence count----', sentence_count)
    print('----total sentence end with splitters---', end_with_splitters_count)
    len_arr = np.array(sorted([len(seq) for seq in new_txt_seqs]))
    print('------sentence splitter summary------')
    print('---min max---', len_arr.min(), len_arr.max())
    print('---mean---', np.mean(len_arr))
    for max_len in [50, 100, 150]:
        print('---length less then %d is---: %d' \
            %(max_len, len([val for val in len_arr if val < max_len])))
