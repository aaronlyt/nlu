
import os
import sys
import math
import codecs
import random
import json

if __name__ == '__main__':
    with open('../dataset/train.json', 'rb') as reader:
        data = json.load(reader)
    dataset_len = len(data)
    
    random.shuffle(data)

    train_ratio = math.floor(0.9 * dataset_len)
    dev_ratio = math.floor(0.1 * dataset_len)
    
    train_data = data[:train_ratio]
    dev_data = data[train_ratio:]


    with codecs.open('../dataset/train_s.json', 'w', 'utf-8') as writer:
        json.dump(train_data, writer, ensure_ascii=False)
    
    with codecs.open('../dataset/dev_s.json', 'wb', encoding='utf-8') as writer:
        json.dump(dev_data, writer, ensure_ascii=False)