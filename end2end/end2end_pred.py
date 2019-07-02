# -*- coding: utf-8 -*-
import sys
import os
import json
import codecs
import xgboost as xgb
import sklearn_crfsuite
import pickle
import numpy as np

from crf_features import *
from seqeval import *

print(sys.version)

def load_info():
    """
    """
    pass


def bs_pred(text_dict, ds_info_dict):
    """
    """
    pass


if __name__ == '__main__':
    import json
    ds_info_dict = load_info()

    #data_path = '../dataset/test_s.json'
    #output_path = '../dataset/output.json'

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