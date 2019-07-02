import os
import sys
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from .preprocess_util import read_data_split


class DataLoader(object):
    """
    """
    def __init__(self, dsset_path, vocab, l_vocab, \
        data_process='origin', max_sent_len=256, min_sent_len=None):
        """
        """
        self.vocab = vocab
        self.l_vocab = l_vocab
        if data_process == 'origin':
            self.dsset = self.read_dsset(dsset_path)[:100]
        elif data_process == 'simple_split':
            self.dsset = read_data_split(dsset_path, vocab, \
                l_vocab, max_sent_len, data_process)
        elif data_process == 'splitters_split':
            self.dsset = read_data_split(dsset_path, vocab, \
                l_vocab, max_sent_len, data_process, min_sent_len)

        self.ds_len = len(self.dsset)

    def read_dsset(self, dsset_path):
        """
        """
        txt_seqs = []
        label_seqs = []

        with open(dsset_path, 'r') as reader:
            txts = []
            labels = []
            for line in reader:
                if not line.strip():
                    txt_seqs.append(txts)
                    label_seqs.append(labels)
                    txts = []
                    labels = []
                    continue
                line_str = line.strip().split()
                txts.append(self.vocab[line_str[0]])
                label = line_str[1]
                labels.append(self.l_vocab[label])

        return list(zip(txt_seqs, label_seqs))

    def padding(self, dset, batch_size, padding_type='batch'):
        """
        padding on the whole dataset, can also on the batch
        """
        new_dset = []
        new_dset_seqs = []
        new_dset_labels = []
        
        batch_num = (len(dset) // batch_size) \
            if len(dset) % batch_size ==0 else (len(dset) // batch_size + 1)
        if padding_type == 'whole':
            max_seqs_len = max([len(seq) for seq, label in dset])
        for batch_idx in range(batch_num):
            start_idx = (batch_idx) * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch_data = dset[start_idx:end_idx]
            if padding_type == 'batch':
                max_seqs_len = max([len(seq) for seq, label in batch_data])
            x = []
            y = []
            for seq, label in batch_data:
                x.append(seq + [0] * (max_seqs_len - len(seq)))
                y.append(label + [0] * (max_seqs_len - len(label)))
            # padding on the batch
            if padding_type == 'batch':
                new_dset.append((np.array(x), np.array(y)))
            else:
                # padding on the whole dataset
                new_dset_seqs.extend(x)
                new_dset_labels.extend(y)

        return new_dset, new_dset_seqs, new_dset_labels 

    def generator(self):
        for example in self.dsset:
            yield example


def build_data_generator(dataset_loader, params):
    """
    """
    def map_fn(seq, label):
        #return (seq, tf.expand_dims(label, -1))
        return (seq, keras.backend.one_hot(label, \
            params['num_entities']))

    def element_length_fn(example_x, example_y):
        """
        """
        return tf.shape(example_x)
        
    shapes = ([params['max_sent_len'], ], \
        [params['max_sent_len'], ])
    data = tf.data.Dataset.from_generator(dataset_loader.generator, \
        (tf.int32, tf.int32))
    
    defaults = (0, 0)
    data = data.shuffle(params['buffer'])
    ori_data = data.padded_batch(params['batch_size'], shapes, defaults)
    
    # new added, bucket
    bucket_data = data.apply(tf.data.experimental.bucket_by_sequence_length(\
        element_length_func=element_length_fn, \
            bucket_boundaries=[212, 280, 400], \
            bucket_batch_sizes=[params['batch_size']] * 4, \
                padded_shapes=shapes))

    bucket_data = bucket_data.map(map_fn)
    ori_data = ori_data.map(map_fn)

    ori_data = ori_data.repeat(params['ori_epochs'])
    bucket_data = bucket_data.repeat(params['bucket_epochs'])

    return ori_data, bucket_data


def build_data_array(dataset_loader, params):
    """
    """
    def map_fn(seq, label):
        return (seq, keras.backend.one_hot(label, \
            params['num_entities']))

    _, pad_dset_seqs, pad_dset_labels = dataset_loader.padding(\
        dataset_loader.dsset, params['batch_size'], 'whole')
    pad_dset_seqs, pad_dset_labels = np.array(pad_dset_seqs), \
            np.array(pad_dset_labels)
    print('----dataset shape----', pad_dset_seqs.shape, \
        pad_dset_labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices(\
        (pad_dset_seqs, pad_dset_labels))
    dataset = dataset.batch(params['batch_size'])
    dataset = dataset.map(map_fn)
    dataset = dataset.shuffle(params['buffer'])
    dataset = dataset.cache()
    dataset = dataset.repeat(params['ori_epochs'])
    return dataset 
