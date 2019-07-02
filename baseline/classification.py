import os
import sys
import json
import pickle
import numpy as np
import scipy
import sklearn
import xgboost as xgb
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from make_features import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from data_util.crf_features import extract_crf_feature
from third_party.seqeval import *


def train_domain_cls(dataset):
    domain_train = xgb.DMatrix(dataset['train']['feats'], \
        dataset['train']['y_domain'])
    domain_dev = xgb.DMatrix(dataset['dev']['feats'], \
        dataset['dev']['y_domain'])
    domain_test = xgb.DMatrix(dataset['test']['feats'], \
        dataset['test']['y_domain'])

    param = {'max_depth': 6, 'eta': 0.3, 'silent': 1, \
        'objective': 'multi:softmax', 'colsample_bytree':0.9, \
            'lambda': 5, 'subsample': 0.8, 'num_parallel_tree': 1, \
                'max_delta_step': 0}
    param['nthread'] = 4
    param['num_class'] = len(dataset['domain_l_dict'])
    param['eval_metric'] = ['mlogloss', 'merror']
    
    evallist = [(domain_train, 'train'), (domain_dev, 'eval'), \
        (domain_test, 'test')]
    num_round = 200
    bst = xgb.train(param, domain_train, num_round, \
        evallist)
    bst.save_model('../dataset/models/domain_cls.bin')
    return bst


def train_intent_cls(dataset):
    """
    """
    domain_train = xgb.DMatrix(dataset['train']['feats'], \
        dataset['train']['y_intent'])
    domain_dev = xgb.DMatrix(dataset['dev']['feats'], \
        dataset['dev']['y_intent'])
    domain_test = xgb.DMatrix(dataset['test']['feats'], \
        dataset['test']['y_intent'])

    param = {'max_depth': 6, 'eta': 0.3, 'silent': 1, \
        'objective': 'multi:softmax', 'colsample_bytree':0.9, \
            'lambda': 5, 'subsample': 0.8, 'num_parallel_tree': 1}
    param['nthread'] = 4
    param['num_class'] = len(dataset['intent_l_dict'])
    param['eval_metric'] = ['mlogloss', 'merror']
    
    evallist = [(domain_train, 'train'), (domain_dev, 'eval'), \
        (domain_test, 'test')]
    num_round = 100
    bst = xgb.train(param, domain_train, num_round, \
        evallist)

    bst.save_model('../dataset/models/intent_cls.bin')
    return bst


def  train_ner(dataset):
    """
    """
    train_x, train_y = extract_crf_feature(\
        dataset['train']['raw_txt'], \
            dataset['train']['ner_labels'])
    dev_x, dev_y = extract_crf_feature(\
        dataset['dev']['raw_txt'], \
            dataset['dev']['ner_labels'])
    test_x, test_y = extract_crf_feature(\
        dataset['test']['raw_txt'], \
            dataset['test']['ner_labels'])
    
    crf = sklearn_crfsuite.CRF(algorithm='l2sgd', \
        all_possible_transitions=True)
    #crf.fit(train_x, train_y)
    print(crf.get_params().keys())
    # {'max_iterations': 1000, 'calibration_eta': 0.1, 'c2': 0.1} 0.66
    params_space = {
        'max_iterations': [100, 200, 500, 1000], 
        'c2': [0, 0.1, 0.5, 1],
        'calibration_eta':[0.001, 0.1, 0.2]
        }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted')
    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(train_x, train_y)

    crf = rs.best_estimator_
    print('---best prameters---', rs.best_params_)
    pickle.dump(crf, open('../dataset/models/bs_crf.model', 'wb'))

    def eval_crf(y_true, y_pred):
        """
        """
        classification_report(y_true, y_pred)
        f1score = f1_score(y_true, y_pred)
        print('----', f1score)

    dev_pred = crf.predict(dev_x)
    test_pred = crf.predict(test_x)

    labels = list(crf.classes_)
    labels.remove('O')

    eval_crf(dev_y, dev_pred)
    eval_crf(test_y, test_pred)

    return True, 'Ok'


if __name__ == '__main__':
    train_path = '../dataset/train_s.json'
    dev_path = '../dataset/dev_s.json'
    test_path = '../dataset/test_s.json'
    dataset = build_data_feats(train_path, dev_path, test_path)

    #train_domain_cls(dataset)
    #train_intent_cls(dataset)
    train_ner(dataset)
    """
    pickle.dump(dataset['vectorizer'], \
        open('../dataset/vectorizer.pkl', 'wb'))
    dataset_info = '../dataset/ds_info.json'
    json.dump({'domain_l_dict': dataset['domain_l_dict'], \
            'intent_l_dict': dataset['intent_l_dict']}, \
                open(dataset_info, 'w'))
    """