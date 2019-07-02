import sys
import os
import collections
import json
import codecs
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

def extract_raw_data(data_path):
    """
    """
    with codecs.open(data_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    
    texts = [example['text'] for example in data]
    domain_labels = [example['domain'] for example in data]
    intent_labels = [example['intent'] for example in data]

    slots_ners = []
    for example in data:
        text = example['text']
        ner = ['O'] * len(text)
        slots = example['slots']
        for key, val in slots.items():
            start_idx = text.find(val)
            end_idx = start_idx + len(val) - 1
            if len(val) == 1:
                ner[start_idx] = 'S-' + key
            else:
                ner[start_idx] = 'B-' + key
                ner[end_idx] = 'E-' + key
                for idx in range(start_idx + 1, end_idx):
                     ner[idx] = 'I-' + key
        slots_ners.append(ner)
    # TODO 验证这个函数正确输出
    return texts, domain_labels, intent_labels, slots_ners


def self_tokenizer(text):
    return list(text)


def extract_tfidf_feats(train_txt, dev_text, test_text):
    """
    into the matrix form
    """
    
    whole_text = train_txt + dev_text + test_text
    # tfidf vectorizer
    vectorizer = TfidfVectorizer(tokenizer=self_tokenizer, \
        analyzer='word', ngram_range=(1, 2), \
            min_df=1)
    vectorizer.fit(whole_text)

    x_tra_matrix = vectorizer.transform(train_txt)
    x_dev_matrix = vectorizer.transform(dev_text)
    x_test_matrix = vectorizer.transform(test_text)

    return vectorizer, x_tra_matrix, x_dev_matrix, x_test_matrix


def to_label_id(dom_labels, intent_labels, \
    domain_dict, intent_dict, key, \
        domain_num, intent_num):
    """
    """
    domain_enc = OneHotEncoder(sparse=False, \
        categories=[range(domain_num)])
    intent_enc = OneHotEncoder(sparse=False, \
        categories=[range(intent_num)])

    y_domain = np.array([domain_dict[dom_l] for dom_l in dom_labels])
    y_intent = np.array([intent_dict[intent_l] for intent_l in intent_labels])

    y_domain_onehot = domain_enc.fit_transform(y_domain.reshape(-1, 1))
    y_intent_onehot = intent_enc.fit_transform(y_intent.reshape(-1, 1))
    return {key: {'y_domain': y_domain, 'y_intent': y_intent, \
        'y_domain_onehot': y_domain_onehot, \
            'y_intent_onehot': y_intent_onehot}}
    

def build_data_feats(train_path, dev_path, test_path):
    """
    """
    # get the raw text
    train_texts, train_domain_labels, \
        train_intent_labels, train_slots_ners = \
            extract_raw_data(train_path)
    dev_texts, dev_domain_labels, \
        dev_intent_labels, dev_slots_ners = \
            extract_raw_data(dev_path)
    test_texts, test_domain_labels, \
        test_intent_labels, test_slots_ners = \
            extract_raw_data(test_path)

    # the whole labels
    domain_labels = set(train_domain_labels + dev_domain_labels \
        + test_domain_labels)
    intent_labels = set(train_intent_labels + dev_intent_labels \
        + test_intent_labels)
    domain_labels_dict = dict(zip(list(domain_labels), \
        range(len(domain_labels))))
    intent_labels_dict = dict(zip(list(intent_labels), \
        range(len(intent_labels))))
    
    print('----label dict summary----', \
        len(domain_labels_dict), len(intent_labels_dict))

    dataset_dict = collections.defaultdict(dict)
    dataset_dict.update(to_label_id(train_domain_labels, \
        train_intent_labels, domain_labels_dict, \
            intent_labels_dict, 'train', \
                len(domain_labels_dict), len(intent_labels_dict)))
    dataset_dict.update(to_label_id(dev_domain_labels, \
        dev_intent_labels, domain_labels_dict, \
            intent_labels_dict, 'dev', \
                len(domain_labels_dict), len(intent_labels_dict)))
    dataset_dict.update(to_label_id(test_domain_labels, \
        test_intent_labels, domain_labels_dict, \
            intent_labels_dict, 'test', \
                len(domain_labels_dict), len(intent_labels_dict)))
    
    # the feats matrix
    vectorizer, train_feats, dev_feats, test_feats, = \
        extract_tfidf_feats(train_texts, dev_texts, test_texts)
    print('----feats shape----', train_feats.shape, \
        dev_feats.shape, test_feats.shape)
    dataset_dict['train'].update({'feats': train_feats})
    dataset_dict['dev'].update({'feats': dev_feats})
    dataset_dict['test'].update({'feats': test_feats})

    # dataset level data
    dataset_dict['domain_l_dict'] = domain_labels_dict
    dataset_dict['intent_l_dict'] = intent_labels_dict
    dataset_dict['vectorizer'] = vectorizer

    # ner data
    dataset_dict['train']['raw_txt'] = train_texts
    dataset_dict['train']['ner_labels'] = train_slots_ners
    dataset_dict['dev']['raw_txt'] = dev_texts
    dataset_dict['dev']['ner_labels'] = dev_slots_ners
    dataset_dict['test']['raw_txt'] = test_texts
    dataset_dict['test']['ner_labels'] = test_slots_ners

    return dataset_dict


if __name__ == '__main__':
    texts, domain_labels, intent_labels, slots_ners = \
        extract_raw_data('../dataset/train.json')
    print(slots_ners)
    train_path = '../dataset/train_s.json'
    dev_path = '../dataset/dev_s.json'
    test_path = '../dataset/test_s.json'
    dataset = build_data_feats(train_path, dev_path, test_path)

    