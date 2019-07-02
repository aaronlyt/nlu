"""
"""

import os
import sys
import json
import numpy as np
import math
import logging
import datetime
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python import debug as tf_debug

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from end2end_model import jointModel
from data_util import read_bert_token, build_bert_data_array
from third_party.seqeval.callbacks import F1Metrics

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_KERAS'] = '1'


if not os.path.exists('../results'):
    os.makedirs('../results')

tf.compat.v1.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('../results/bert.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('run_bert').handlers = handlers

logdir = '../results/bert_logs_2'
file_writer = tf.summary.create_file_writer(logdir + "/metrics_crf")
file_writer.set_as_default()


l_vocab = json.load(open('../dataset/label2id', 'r'))
id2label = dict([(id, label) for label, id in l_vocab.items()])


def param_step_decay(initial_lrate, drop, epochs_drop):
    def step_decay(epoch):
        lrate = initial_lrate * math.pow(drop, \
            math.floor((1 + epoch)/ epochs_drop))
        tf.summary.scalar('learning rate', data=lrate, step=epoch)
        return lrate
    return step_decay



def build_model_pipeline(params, train_dataset, \
    validation_data, steps_per_epoch):
    """
    """
    optimizer = keras.optimizers.Adam(params['lr'])
    model, loss, metrics = jointModel(params)
    model.compile(optimizer, loss=loss, metrics=metrics)

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        keras.callbacks.EarlyStopping(patience=5, monitor='loss'),
        # Write TensorBoard logs to `./logs` directory
        keras.callbacks.TensorBoard(log_dir=logdir),
        keras.callbacks.ModelCheckpoint('../results/bert.cpk', \
            verbose=1, save_best_only=True, save_weights_only=True, \
                monitor='loss'),
        #keras.callbacks.LearningRateScheduler(\
        #    param_step_decay(0.001, 0.8, 8)),
        F1Metrics(id2label, validation_data=validation_data)
        ]
    model.load_weights('../results/bert.cpk')
    model.fit(train_dataset, epochs=params['ori_epochs'], \
        steps_per_epoch=steps_per_epoch, \
            callbacks=callbacks)
    

if __name__ == '__main__':    
    params = {
        'use_bert': True,
        'bert_path': '../dataset/chinese_L-12_H-768_A-12', 
        'data_process': 'simple_split',
        'data_fit_format': 'array',
        'max_sent_len': 128,
        'ori_epochs': 100,
        'batch_size': 64,
        'use_emb_drop': False,
        'emb_drop_rate': 0.1,
        'num_layer': 0,
        'num_lstm_cell': [150],
        'rec_drops': [0.1],
        'lr': 1e-5
    }
    params['num_entities'] = len(l_vocab)
    params['id2label'] = id2label
    params['pad_value'] = 0

    dsset_path = '../dataset/emrexample.train'
    dev_path = '../dataset/emrexample.dev'
    test_path = '../dataset/emrexample.text'

    vocab_path = os.path.join(params['bert_path'], 'vocab.txt')
    vocab = read_bert_token(vocab_path)

    train_dataset, train_len = build_bert_data_array(dsset_path, \
        vocab, l_vocab, params)
    dev_dataset, _  = build_bert_data_array(dev_path, vocab, \
        l_vocab, params, 'dev')

    steps_per_epoch = int(train_len/params['batch_size']) + 1

    build_model_pipeline(params, train_dataset, \
        dev_dataset, steps_per_epoch)
    
