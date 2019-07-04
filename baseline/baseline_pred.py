# -*- coding: utf-8 -*-
import sys
import os
import json
import codecs
import xgboost as xgb
import sklearn_crfsuite
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from data_util.crf_features import *
from third_party.seqeval.metrics import *

def load_info():
    """
    """
    # load model info
    domain_cls = xgb.Booster()
    domain_cls.load_model('../dataset/models/domain_cls.bin')
    intent_cls = xgb.Booster()
    intent_cls.load_model('../dataset/models/intent_cls.bin')

    crf_ner = pickle.load(\
        open('../dataset/models/bs_crf.model', 'rb'))

    # load basic infomation
    ds_info = json.load(open('../dataset/ds_info.json', 'r'))
    ds_info['vectorizer'] = pickle.load(\
        open('../dataset/vectorizer.pkl', 'rb'))
    ds_info['domain_l_dict'] = dict([(idx, l) \
        for l, idx in ds_info['domain_l_dict'].items()])
    ds_info['intent_l_dict'] = dict([(idx, l) \
        for l, idx in ds_info['intent_l_dict'].items()])
    ds_info['domain_cls'] = domain_cls
    ds_info['intent_cls'] = intent_cls
    ds_info['ner_crf'] = crf_ner
    return ds_info


def bs_pred(text_dict, ds_info_dict):
    """
    """
    
    features = ds_info_dict['vectorizer'].transform([text_dict['text']])
    features = xgb.DMatrix(features)
    domain_l = ds_info_dict['domain_cls'].predict(features)[0]
    text_dict['domain'] = ds_info_dict['domain_l_dict'][domain_l]
    intent_l = ds_info_dict['intent_cls'].predict(features)[0]
    text_dict['intent'] = ds_info_dict['intent_l_dict'][intent_l]
    slot = {}
    text_tuple = [[ch] for ch in text_dict['text']]
    ner_feautres = sent2features(text_tuple)
    ner_predict = ds_info_dict['ner_crf'].predict([ner_feautres])[0]
    entities = get_entities(ner_predict)
    for entity_tuple in entities:
        entity, start_idx, end_idx = entity_tuple
        slot[entity] = text_dict['text'][start_idx: (end_idx + 1)]
    text_dict['slots'] = slot

    return text_dict


if __name__ == '__main__':
    import json
    ds_info_dict = load_info()

    data_path = '../dataset/test_s.json'
    output_path = '../dataset/output.json'

    data_path = sys.argv[1]
    output_path = sys.argv[2]

    with codecs.open(data_path, 'r', encoding='utf-8') as fp:
        dev_dct = json.load(fp)

    rguess_dct = []
    for dev_data in dev_dct:
        text_dic = {"text": dev_data['text']}
        rguess_dct.append(bs_pred(text_dic, ds_info_dict))
    
    with codecs.open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(rguess_dct, fp, ensure_ascii=False)