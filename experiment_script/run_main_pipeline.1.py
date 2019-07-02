"""
"""

import os
import sys
import json
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python import debug as tf_debug

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from model import *
from data_util import *

from third_party.seqeval.callbacks import F1Metrics

os.environ['CUDA_VISIBLE_DEVICES'] = ''

if not os.path.exists('../results'):
    os.makedirs('../results')

model_path = '../results/lstmcrf_test_2.cpk'

logdir = '../results/blstmcrf_logs_2'
file_writer = tf.summary.create_file_writer(logdir + "/metrics_crf")
file_writer.set_as_default()


vocab = json.load(open('../dataset/char2id', 'r'))
l_vocab = json.load(open('../dataset/label2id', 'r'))
id2label = dict([(id, label) for label, id in l_vocab.items()])
id2char = dict([(id, label) for label, id in vocab.items()])

def param_step_decay(initial_lrate, drop, epochs_drop, staircase=True):
    def step_decay(epoch):
        if staircase:
            lrate = initial_lrate * math.pow(drop, \
                math.floor((1 + epoch)/ epochs_drop))
        else:
            lrate = initial_lrate * math.pow(drop, \
                (1 + epoch) / epochs_drop)
        tf.summary.scalar('learning rate', data=lrate, step=epoch)
        return lrate
    return step_decay



def build_pipeline_2(params, ori_train_data, bucket_train_data, \
    validation_data, steps_per_epoch):
    """
    """
    optimizer = keras.optimizers.Adam(params['lr'])
    model, loss, acc = LstmCrf(params)
    model.compile(optimizer, loss=loss, metrics=[acc])

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        keras.callbacks.EarlyStopping(patience=2, monitor='loss'),
        # Write TensorBoard logs to `./logs` directory
        keras.callbacks.TensorBoard(log_dir=logdir),
        keras.callbacks.ModelCheckpoint(model_path, \
            save_weights_only=True, verbose=1, save_best_only='True', \
                monitor='loss'),
        keras.callbacks.LearningRateScheduler(\
            param_step_decay(0.01, 0.8, 8, False)),
        F1Metrics(id2label, validation_data=validation_data)
        ]
    
    """
    model.fit(bucket_train_data, epochs=params['bucket_epochs'], \
        steps_per_epoch=steps_per_epoch, \
            callbacks=callbacks)
    """
    #model.load_weights(model_path)

    callbacks[3] = keras.callbacks.LearningRateScheduler(\
            param_step_decay(0.001, 0.8, 6, False))

    model.fit(ori_train_data, epochs=params['ori_epochs'], \
        steps_per_epoch=steps_per_epoch, \
            callbacks=callbacks)
    
    # for finetune with small learning rate
    callbacks[3] = keras.callbacks.LearningRateScheduler(\
            param_step_decay(1e-5, 0.8, 100, False))

    model.fit(ori_train_data, epochs=10, \
        steps_per_epoch=steps_per_epoch, \
            callbacks=callbacks)


def predict(test_path, params, max_sent_len, min_sent_len):
    """
    """
    test_dataset_loader = DataLoader(test_path, vocab, \
        l_vocab, params['data_process'], max_sent_len, min_sent_len)
    test_dataset, _,  _ = test_dataset_loader.padding(\
        test_dataset_loader.dsset, \
        params['batch_size'], 'batch')
    test_dataset = one_hot(test_dataset, params['num_entities'])
    
    optimizer = keras.optimizers.Adam(params['lr'])
    model, loss, acc = LstmCrf(params)
    model.compile(optimizer, loss=loss, metrics=[acc])
    model.load_weights(model_path)

    f1metrics = F1Metrics(params['id2label'])
    f1metrics.model = model
    y_true = []
    y_pred = []
    x_data = []
    for X, y in test_dataset:
        y_true_batch, y_pred_batch = f1metrics.predict(X, y)
        #print(y_pred_batch)
        y_true.extend(y_true_batch)
        y_pred.extend(y_pred_batch)
        x_data.extend(X)
    
    erro_info_stats = {}
    erro_info_stats_from = {}
    erro_info_stats_to = {}
    total_count = 0
    with open('../results/pred_true.output', 'w') as writer:
        for x_seq, label_seq, pred_seq in list(zip(x_data, y_true, y_pred)):
            for ch, label, pred in list(zip(x_seq, label_seq, pred_seq)):
                writer.write('%s\t%s\t%s\t%d\n' % \
                        (id2char[ch], label, pred, label == pred))
                if label != pred:
                    total_count += 1
                    error_inf = '%s2%s' %(label, pred)
                    if error_inf not in erro_info_stats:
                        erro_info_stats[error_inf] = 1
                    else:
                        erro_info_stats[error_inf] += 1
                    if label not in erro_info_stats_from:
                        erro_info_stats_from[label] = 1
                    else:
                        erro_info_stats_from[label] += 1
                    
                    if pred not in erro_info_stats_to:
                        erro_info_stats_to[pred] = 1
                    else:
                        erro_info_stats_to[pred] += 1

    sorted_inf = sorted(erro_info_stats.items(), key=lambda x: x[1], reverse=True)
    sorted_inf_f = sorted(erro_info_stats_from.items(), key=lambda x: x[1], reverse=True)
    sorted_inf_t = sorted(erro_info_stats_to.items(), key=lambda x: x[1], reverse=True)

    with open('../results/error_info.output', 'w') as writer:
        for (inf, count) in sorted_inf:
            writer.write('%s\t%d\t%f\n' %(inf, count, count * 1.0 / total_count))
    
    with open('../results/error_info_f.output', 'w') as writer:
        for (inf, count) in sorted_inf_f:
            writer.write('%s\t%d\t%f\n' %(inf, count, count * 1.0 / total_count))

    with open('../results/error_info_t.output', 'w') as writer:
        for (inf, count) in sorted_inf_t:
            writer.write('%s\t%d\t%f\n' %(inf, count, count * 1.0 / total_count))

    score = f1metrics.score(y_true, y_pred)



if __name__ == '__main__':    
    params = {
        'phrase': 'predict',
        'data_process': 'splitters_split',
        'splitters_max_len': 96, 
        'splitters_min_len':32,
        'data_fit_format': 'array',
        'max_sent_len': 128,
        'embedding_path': '../dataset/word_embedding_char', 
        'use_bert': False,
        'use_crf': True,
        'ori_epochs': 120,
        'bucket_epochs': 11,
        'batch_size': 128,
        'buffer': 15000,
        'use_pre_emb': True,
        'use_emb_drop': True,
        'emb_drop_rate': 0.1,
        'emb_dim': 200,
        'use_emb_transform':False,
        'num_layer': 2,
        'num_lstm_cell': [200, 150],
        'rec_drops': [0.1, 0.3],
        'lr': 0.01
    }
    
    embedding, emb_dimension = read_embedding(\
        params['embedding_path'], vocab)

    params['pre_emb'] = np.array(embedding)
    params['emb_dim'] = emb_dimension
    params['vocab_len'] = len(vocab)
    params['num_entities'] = len(l_vocab)
    params['id2label'] = id2label
    params['pad_value'] = 0

    dsset_path = '../dataset/emrexample.train'
    dev_path = '../dataset/emrexample.dev'
    test_path = '../dataset/emrexample.test'

    max_sent_len = min_sent_len = None
    if params['data_process'] == 'simple':
        max_sent_len = params['max_sent_len']
    elif params['data_process'] == 'splitters_split':
        max_sent_len = params['splitters_max_len']
        min_sent_len = params['splitters_min_len']
    
    if params['phrase'] == 'train':
        dataset_loader = DataLoader(dsset_path, vocab, \
            l_vocab, params['data_process'], max_sent_len, min_sent_len)
        dev_dataset_loader = DataLoader(dev_path, vocab, \
            l_vocab, params['data_process'], max_sent_len, min_sent_len)

        params['buffer'] = dataset_loader.ds_len
        steps_per_epoch = int(dataset_loader.ds_len/params['batch_size']) + 1

        bucket_dataset = None
        if params['data_fit_format'] == 'generator':
            ori_dataset, bucket_dataset = build_data_generator(dataset_loader, params)
        elif params['data_fit_format'] == 'array':
            ori_dataset = build_data_array(dataset_loader, params)

        dev_dataset, _,  _ = dev_dataset_loader.padding(\
            dev_dataset_loader.dsset, \
            params['batch_size'], 'batch')
        dev_dataset = one_hot(dev_dataset, params['num_entities'])
        
        build_pipeline_2(params, ori_dataset, bucket_dataset, \
            dev_dataset, steps_per_epoch)
    elif params['phrase'] == 'predict':
        predict(test_path, params, max_sent_len, min_sent_len)
