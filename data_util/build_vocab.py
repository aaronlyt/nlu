import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from third_party.seqeval.metrics import get_entities


if __name__ == '__main__':

    data_paths = ['../dataset/emrexample.train', 
    '../dataset/emrexample.dev', '../dataset/emrexample.test']

    label2id = {}
    label2id['PADL'] = 0
    char2id = {}
    char2id['PAD'] = 0
    char2id['UNK'] = 1
    # TODO, not include the UNK
    summary = {}
    summary['sent_count'] = 0
    summary['sent_len'] = []
    summary['wlabel_count'] = {}
    
    label_seq = []
    for data_path in data_paths:
        with open(data_path, 'r') as reader:
            seq_len = 0
            for line in reader:
                if not line.strip():
                    summary['sent_count'] += 1
                    summary['sent_len'].append(seq_len)
                    seq_len = 0
                    continue
                seq_len += 1
                line_str = line.strip().split()
                char, label = line_str[0], line_str[1]

                if char not in char2id:
                    char2id[char] = len(char2id)
                
                if label not in label2id:
                    label2id[label] = len(label2id)

                label_seq.append(label)
                
    w_entities = [val[0] for val in get_entities(label_seq)]
    for entity in w_entities:
        if entity not in summary['wlabel_count']:
            summary['wlabel_count'][entity] = 1
        else:
            summary['wlabel_count'][entity] += 1
    
    summary['wlabel_count_l'] = sorted(summary['wlabel_count'].items(), \
        key=lambda x: x[1], reverse=True)

    label2id_path = '../dataset/label2id'
    char2id_path = '../dataset/char2id'
    summary_path = '../dataset/data_summary'
    

    json.dump(label2id, open(label2id_path, 'w'))
    json.dump(char2id, open(char2id_path, 'w'))
    json.dump(summary, open(summary_path, 'w'))