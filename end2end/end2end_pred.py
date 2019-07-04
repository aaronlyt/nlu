# -*- coding: utf-8 -*-
import sys
import os
import json
import codecs
import xgboost as xgb
import sklearn_crfsuite
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from end2end_model import jointModel
from data_util.bert_utils import read_bert_token, process_bert_format
from data_util.preprocess_util import *
from third_party.seqeval.metrics import get_entities

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_KERAS'] = '1'


params = {
        'use_bert': True,
        'bert_path': '../dataset/chinese_L-12_H-768_A-12', 
        'data_process': 'simple_split',
        'data_fit_format': 'array',
        'max_sent_len': 30,
        'ori_epochs': 100,
        'batch_size': 64,
        'use_emb_drop': False,
        'use_lstm': False, 
        'emb_drop_rate': 0.1,
        'num_layer': 0,
        'num_lstm_cell': [150],
        'rec_drops': [0.1],
        'lr': 1e-5
    }
 
def load_info():
    """
    """
    
    vocab_path = os.path.join(params['bert_path'], 'vocab.txt')
    vocab = read_bert_token(vocab_path)

    with codecs.open('../dataset/label2id', 'r', 'utf-8') as fp:
        label2id = json.load(fp)
    ner_vocab, domain_vocab, intent_vocab = label2id['ner_label2id'], \
        label2id['domain_label2id'], label2id['intent_label2id']
    ner_id2label = dict([(id, label) for label, id in ner_vocab.items()])
    domain_id2label = dict([(id, label) for label, id in domain_vocab.items()])
    intent_id2label = dict([(id, label) for label, id in intent_vocab.items()])

    params['entities_num'] = len(ner_vocab)
    params['domain_num'] = len(domain_vocab)
    params['intent_num'] = len(intent_vocab)
    params['id2label'] = ner_id2label
    params['pad_value'] = 0

    optimizer = keras.optimizers.Adam(params['lr'])
    model, loss, metrics = jointModel(params)
    model.compile(optimizer, loss=loss, metrics=metrics)

    model.load_weights('../results/bert.cpk')
    
    return model, vocab, ner_id2label, domain_id2label, intent_id2label


def bs_pred(text_dict, model, vocab, ner_id2label, \
    domain_id2label, intent_id2label):
    """
    """
    txt_seq = list(text_dict['text'])
    indices, segments = process_bert_format(vocab, \
            txt_seq, params['max_sent_len'])
    dataset_indices = np.array([indices])
    dataset_segments = np.array([segments])
    print(dataset_indices)
    dataset = tf.data.Dataset.from_tensor_slices(\
        {
            'Input-Token': dataset_indices, 
            'Input-Segment': dataset_segments
            })
    dataset = dataset.batch(1)
    preds = model.predict(dataset)
    domain_labels = preds[0].argmax(-1)
    intent_labels = preds[1].argmax(-1)
    ner_labels = preds[2].argmax(-1)

    text_dict['domain'] = domain_id2label[domain_labels[0]]
    text_dict['intent'] = intent_id2label[intent_labels[0]]
    ner_predict = [ner_id2label[l_id] for l_id in ner_labels[0]]
    # remove padding
    ner_predict = ner_predict[1: (len(txt_seq) + 1)]
    entities = get_entities(ner_predict)
    slot = {}
    print(entities)
    for entity_tuple in entities:
        entity, start_idx, end_idx = entity_tuple
        slot[entity] = text_dict['text'][start_idx: (end_idx + 1)]
    text_dict['slots'] = slot
    
    print(text_dict)


if __name__ == '__main__':
    import json
    model, vocab, ner_id2label, domain_id2label, intent_id2label = load_info()

    data_path = '../dataset/dev_s.json'
    output_path = '../dataset/dev_output.json'
    #data_path = sys.argv[1]
    #output_path = sys.argv[2]

    with codecs.open(data_path, 'r', encoding='utf-8') as fp:
        dev_dct = json.load(fp)

    rguess_dct = []
    for dev_data in dev_dct:
        text_dic = {"text": dev_data['text']}
        rguess_dct.append(bs_pred(text_dic, model, vocab, \
            ner_id2label, domain_id2label, intent_id2label))
        sys.exit(0)

    with codecs.open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(rguess_dct, fp, ensure_ascii=False)