import os
import sys
import json
import codecs
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from third_party.seqeval import get_entities
from make_features import extract_raw_data


if __name__ == '__main__':

    data_paths = ['../dataset/train_s.json', 
    '../dataset/dev_s.json']

    label2id = {}
    label2id['PADL'] = 0
    
    domain_label2id = {}
    domain_label2id['PADL'] = 0

    intent_label2id = {}
    intent_label2id['PADL'] = 0

    char2id = {}
    char2id['PAD'] = 0
    char2id['UNK'] = 1
    # TODO, not include the UNK
    summary = {}
    summary['sent_count'] = 0
    summary['sent_len'] = []
    summary['wlabel_count'] = {}
    summary['max_sent_len'] = 0
    label_seqs = []
    for data_path in data_paths:
        txt_seqs, domain_labels, intent_labels, slots_ners = \
            extract_raw_data(data_path)
        seq_len = 0
        for idx, line in enumerate(txt_seqs):
            summary['sent_count'] += 1
            summary['sent_len'].append(len(line))
            for char in line:
                if char not in char2id:
                    char2id[char] = len(char2id)
            for label in slots_ners[idx]:
                if label not in label2id:
                    label2id[label] = len(label2id)

            label = domain_labels[idx]
            if label not in domain_label2id:
                domain_label2id[label] = len(domain_label2id)
            
            label = intent_labels[idx]
            if label not in intent_label2id:
                intent_label2id[label] = len(intent_label2id)

            label_seqs.extend(slots_ners[idx])
            summary['max_sent_len'] = max(summary['max_sent_len'], \
                len(line))
            
    w_entities = [val[0] for val in get_entities(label_seqs)]
    for entity in w_entities:
        if entity not in summary['wlabel_count']:
            summary['wlabel_count'][entity] = 1
        else:
            summary['wlabel_count'][entity] += 1
    
    summary['wlabel_count_l'] = sorted(summary['wlabel_count'].items(), \
        key=lambda x: x[1], reverse=True)

    label_dict = {'ner_label2id': label2id, \
        'domain_label2id': domain_label2id, \
            'intent_label2id': intent_label2id}
    
    label2id_path = '../dataset/label2id'
    char2id_path = '../dataset/char2id'
    summary_path = '../dataset/data_summary'
    
    with codecs.open(label2id_path, 'w', 'utf-8') as fp:
        json.dump(label_dict, fp, ensure_ascii=False)
    
    with codecs.open(char2id_path, 'w', 'utf-8') as fp:
        json.dump(char2id, fp, ensure_ascii=False)
    
    json.dump(summary, open(summary_path, 'w'))