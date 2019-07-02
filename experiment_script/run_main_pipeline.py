"""
"""

import os
import sys
import json
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras as keras
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from model import *
from data_util import *

from third_party.seqeval.callbacks import F1Metrics

os.environ['CUDA_VISIBLE_DEVICES'] = ''

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

if not os.path.exists('../results'):
    os.makedirs('../results')

tf.compat.v1.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('../results/runlstm.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('run_lstm').handlers = handlers

logdir = '../results/lstm_logs_0.005'
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()


vocab = json.load(open('../dataset/char2id', 'r'))
l_vocab = json.load(open('../dataset/label2id', 'r'))
id2label = dict([(id, label) for label, id in l_vocab.items()])

def param_step_decay(initial_lrate, drop, epochs_drop):
    def step_decay(epoch):
        lrate = initial_lrate * math.pow(drop, \
            math.floor((1 + epoch)/ epochs_drop))
        tf.summary.scalar('learning rate', data=lrate, step=epoch)
        return lrate
    return step_decay

def build_data_generator(dataset_loader, params):
    """
    """
    def map_fn(seq, label):
        return (seq, tf.expand_dims(label, -1))
    
    def element_length_fn(example_x, example_y):
        """
        """
        return tf.shape(example_x)
        
    shapes = ([None], [None])
    data = tf.data.Dataset.from_generator(dataset_loader.generator, \
        (tf.int32, tf.int32), shapes)
    
    defaults = (0, 0)
    ori_data = data.padded_batch(params['batch_size'], shapes, defaults)
    ori_data = ori_data.shuffle(params['buffer'])

    # new added, bucket
    bucket_data = data.apply(tf.data.experimental.bucket_by_sequence_length(\
        element_length_func=element_length_fn, \
            bucket_boundaries=[212, 280, 400], \
            bucket_batch_sizes=[params['batch_size']] * 4, \
                padded_shapes=shapes))

    bucket_data = bucket_data.map(map_fn)
    ori_data = ori_data.map(map_fn)

    return ori_data, bucket_data


def build_pipeline_2(params, ori_train_generator, bucket_train_generator, \
    validation_data, steps_per_epoch):
    """
    """
    #tf.enable_eager_execution()    
    for value in ori_train_generator.take(1):
        print(value[0].shape)

    ori_data = ori_train_generator.repeat(params['ori_epochs'])
    bucket_data = ori_train_generator.repeat(params['bucket_epochs'])
    
    #model = build_test_model(params)
    model = lstm_model(params)

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        keras.callbacks.EarlyStopping(patience=2, monitor='loss'),
        # Write TensorBoard logs to `./logs` directory
        keras.callbacks.TensorBoard(log_dir=logdir),
        keras.callbacks.ModelCheckpoint('../results/lstm_test.cpk', \
            save_weights_only=True, verbose=1),
        keras.callbacks.LearningRateScheduler(\
            param_step_decay(0.01, 0.5, 7)),
        F1Metrics(id2label, validation_data=validation_data)
        ]
    model.fit(bucket_data, epochs=params['bucket_epochs'], \
        steps_per_epoch=steps_per_epoch, \
            callbacks=callbacks)
    
    callbacks[3] = keras.callbacks.LearningRateScheduler(\
            param_step_decay(0.001, 0.8, 10))

    model.fit(ori_data, epochs=params['ori_epochs'], \
        steps_per_epoch=steps_per_epoch, \
            callbacks=callbacks)

    
if __name__ == '__main__':    
    params = {
        'embedding_path': '../dataset/word_embedding_char', 
        'ori_epochs': 51,
        'bucket_epochs': 15,
        'batch_size': 64,
        'buffer': 15000,
        'use_pre_emb': True,
        'use_emb_drop': True,
        'emb_drop': 0.3,
        'emb_dim': 200,
        'num_layer': 1,
        'num_lstm_cell': [150],
        'dropout': [0.5],
        'lr': 0.01
    }
    embedding, emb_dimension = read_embedding(\
        params['embedding_path'], vocab)
    params['pre_emb'] = np.array(embedding)
    params['emb_dim'] = emb_dimension
    params['vocab_len'] = len(vocab)
    params['label_num'] = len(l_vocab)

    dsset_path = '../dataset/emrexample.train'
    dev_path = '../dataset/emrexample.dev'
    test_path = '../dataset/emrexample.text'
    dataset_loader = DataLoader(dsset_path, vocab, l_vocab)
    dev_dataset_loader = DataLoader(dev_path, vocab, l_vocab)

    params['buffer'] = len(dataset_loader.dsset)

    ori_generator, bucket_generator = build_data_generator(dataset_loader, params)
    dev_genrator = build_data_generator(dev_dataset_loader, params)
    validation_data = dev_dataset_loader.padding(dev_dataset_loader.dsset, \
        params['batch_size'])
    
    steps_per_epoch = int(len(dataset_loader.dsset)/params['batch_size']) + 1

    params['id2label'] = id2label
    params['pad_value'] = 0

    build_pipeline_2(params, ori_generator, bucket_generator, \
        validation_data, steps_per_epoch)
    