"""
"""
import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

os.environ['TF_KERAS'] = '1'

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from keras_bert import load_trained_model_from_checkpoint


def get_bert_model(params):
    """
    """
    SEQ_LEN = params['max_sent_len']
    pretrained_path = params['bert_path']
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    
    model = load_trained_model_from_checkpoint(\
        config_path, checkpoint_path, training=False,\
            trainable=True, seq_len=SEQ_LEN)

    return model.inputs, model.outputs[0]


def jointModel(params):
    if params['use_bert']:
        inputs, x = get_bert_model(params)
        # now for test, no need, but in realty, should remove the CL
        #output = keras.layers.Lambda(lambda x: x[:, 1:-1, :])(output)
    else:
        inputs = tf.keras.Input(shape=(None, ))
        init_emb = None
        if params['use_pre_emb']:
            init_emb = tf.keras.initializers.Constant(params['pre_emb'])
        
        x = layers.Embedding(\
                params['vocab_len'], params['emb_dim'], \
                    embeddings_initializer=init_emb, zero_mask=True)(inputs)
    if params['use_emb_drop']:
        x = layers.Dropout(params['emb_drop_rate'])(x)

    if params['use_lstm']:
        for layer_idx in range(params['lstm_num']):
            x = layers.LSTM(\
                units=params['lstm_cells'][layer_idx], \
                    recurrent_dropout=params['rec_drops'][layer_idx], \
                        return_sequences=True)(x)
    regularizer = None
    if params['regularizer'] == 'l1':
        regularizer = tf.keras.regularizers.l1
    elif params['regularizer'] == 'l2':
        regularizer = tf.keras.regularizers.l2

    ner_pred = layers.TimeDistributed(\
        layers.Dense(params['entities_num'], \
        kernel_regularizer=regularizer(params['reg_w'][0]), \
            activation='softmax'), name='ner_output')(x)
    # batch_size * seq_len * embedding_sz
    x_shape = x.get_shape()
    cls_x = layers.Lambda(lambda t: t[:, 0, :])(x)
    intent_pred = layers.Dense(params['intent_num'], \
        kernel_regularizer=regularizer(params['reg_w'][1]), \
        activation='softmax', name='intent_output')(cls_x)
    domain_pred = layers.Dense(params['domain_num'], \
        kernel_regularizer=regularizer(params['reg_w'][2]), \
        activation='softmax', name='domain_output')(cls_x)

    model = tf.keras.Model(inputs=inputs, \
        outputs=[domain_pred, intent_pred, ner_pred])

    # key same as the output layer name, also same as input dict key
    # https://www.tensorflow.org/beta/guide/keras/training_and_evaluation#passing_data_to_multi-input_multi-output_models
    losses = {
        'domain_output': 'sparse_categorical_crossentropy',
        'intent_output': 'sparse_categorical_crossentropy',
        'ner_output': 'sparse_categorical_crossentropy'
        }
    metrics = {
        'domain_output': ['accuracy'],
        'intent_output': ['accuracy'],
        'ner_output': ['accuracy']
        }
    
    return model, losses, metrics 
