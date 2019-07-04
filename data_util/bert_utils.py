"""
"""
import os
import sys
import codecs
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras_bert import Tokenizer

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
from data_util.preprocess_util import *
from data_util.make_features import extract_raw_data


def read_bert_token(vocab_path):
    """
    """
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


def process_bert_format(tokenizer, text, max_sent_len):
    """
    text: char list
    need write for you self, the invoke have problem
    """
    # problem solution
    #indices, segments = tokenizer.encode(first=text, \
    #    max_len=max_sent_len)
    
    text = ['[CLS]'] + text + ['[SEP]']
    indices = [tokenizer.get(token, tokenizer.get('[UNK]')) \
        for token in text]
    indices = indices + (max_sent_len - len(indices)) * [0]
    segments = [0] * max_sent_len
    return indices, segments


def trans_2labelid(vocab, labels, max_sent_len):
    """
    """
    labels = [vocab[label] for label in labels]
    labels = [0] + labels
    labels += [0] * (max_sent_len - len(labels))
    return labels


def read_bert_data(dataset_path, token_dict, ner_vocab, \
    domain_vocab, intent_vocab, max_sent_len):
    """
    """
    #tokenizer = Tokenizer(token_dict)
    dataset_indices = []
    dataset_segments = []
    dataset_ner_labels = []
    dataset_domain_labels = []
    dataset_intent_labels = []
    txt_seqs, domain_labels, intent_labels, slots_ners = \
        extract_raw_data(dataset_path)
    for idx, txt_seq in enumerate(txt_seqs):
        assert(len(txt_seq) < (max_sent_len - 2))
        txt_seq = list(txt_seq)    
        indices, segments = process_bert_format(token_dict, \
            txt_seq, max_sent_len)
        id2token_d = dict([(id, token) for token, id in token_dict.items()])
        dataset_indices.append(indices)
        dataset_segments.append(segments)
        dataset_ner_labels.append(trans_2labelid(ner_vocab, \
            slots_ners[idx], max_sent_len))
        dataset_domain_labels.append(domain_vocab[domain_labels[idx]])
        try:
            dataset_intent_labels.append(intent_vocab[intent_labels[idx]])
        except KeyError:
            print('---', intent_labels[idx])
            dataset_intent_labels.append(intent_vocab['NaN'])
        
        assert(len(dataset_ner_labels[idx]) == len(indices))
        
    print('---split sentence count---', len(dataset_indices))    
    dataset_indices = np.array(dataset_indices)
    dataset_segments = np.array(dataset_segments)
    dataset_ner_labels = np.array(dataset_ner_labels)
    domain_labels = np.array(dataset_domain_labels)
    intent_labels = np.array(dataset_intent_labels)
    return dataset_indices, dataset_segments, dataset_ner_labels, \
        domain_labels, intent_labels


def build_bert_data_array(dataset_path, token_dict, ner_vocab, \
    domain_vocab, intent_vocab, params, phrase='train'):
    """
    """
    max_sent_len = params['max_sent_len']
    dataset_indices, dataset_segments, dataset_ner_labels, \
        dataset_domain_labels, dataset_intent_labels = \
        read_bert_data(dataset_path, token_dict, \
            ner_vocab, domain_vocab, intent_vocab, max_sent_len)

    def map_fn(seq, label):
        #return (seq, tf.expand_dims(label, -1))
        return (seq, keras.backend.one_hot(label, \
            params['num_entities']))

    print('----dataset shape----', dataset_indices.shape, \
        dataset_ner_labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices(\
        ({
            'Input-Token': dataset_indices, 
            'Input-Segment': dataset_segments
            }, \
                {
                    'ner_output': dataset_ner_labels,
                    'domain_output': dataset_domain_labels,
                    'intent_output': dataset_intent_labels
                }))
    #dataset = dataset.shuffle(dataset_indices.shape[0])
    dataset = dataset.batch(params['batch_size'])
    if phrase == 'train':
        dataset = dataset.repeat(params['ori_epochs'])
    #dataset = dataset.map(map_fn)

    return dataset, dataset_indices.shape[0]


if __name__ == '__main__':
    params = {
        'max_sent_len': 30,
        'bert_path': '../dataset/chinese_L-12_H-768_A-12', 
        'num_entities': 0,
        'batch_size': 64,
        'buffer': 3714,
        'ori_epochs': 2
    }
    l_vocab = json.load(open('../dataset/label2id', 'r'))
    params['num_entities'] = len(l_vocab)

    vocab_path = os.path.join(params['bert_path'], 'vocab.txt')
    vocab = read_bert_token(vocab_path)

    with codecs.open('../dataset/label2id', 'r', 'utf-8') as fp:
        label2id = json.load(fp)

    ner_vocab, domain_vocab, intent_vocab = label2id['ner_label2id'], \
        label2id['domain_label2id'], label2id['intent_label2id']
    print(domain_vocab)
    print(intent_vocab)
    dataset_path = '../dataset/train_s.json'
    dataset, _ = build_bert_data_array(dataset_path, vocab, ner_vocab, \
        domain_vocab, intent_vocab, params)


    for val in dataset.take(1):
        print(val[0]['Input-Token'])
        print(val[1])
    