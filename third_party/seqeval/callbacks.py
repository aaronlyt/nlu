import numpy as np
import sys
import sklearn
import sklearn.metrics as sk_metrics
from tensorflow.keras.callbacks import Callback
from .metrics import f1_score, classification_report


class F1Metrics(Callback):

    def __init__(self, id2label, pad_value=0, \
        validation_data=None, digits=4):
        """
        Args:
            id2label (dict): id to label mapping.
            (e.g. {1: 'B-LOC', 2: 'I-LOC'})
            pad_value (int): padding value.
            digits (int or None): number of digits in printed classification report
              (use None to print only F1 score without a report).
        """
        super(F1Metrics, self).__init__()
        self.id2label = id2label
        self.pad_value = pad_value
        self.validation_data = validation_data
        self.digits = digits
        self.is_fit = validation_data is None

    def convert_idx_to_name(self, y, array_indexes):
        """Convert label index to name.
        Args:
            y (np.ndarray): label index 2d array.
            array_indexes (list): list of valid index arrays for each row.
        Returns:
            y: label name list.
        """
        y = [[self.id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

    def predict(self, X, y):
        """Predict sequences.
        Args:
            X (np.ndarray): input data.
            y (np.ndarray): tags.
        Returns:
            y_true: true sequences.
            y_pred: predicted sequences.
        """
        y_pred = self.model.predict_on_batch(X)
        assert(len(y_pred) == 3)
       
        domain_true = y['domain_output'].numpy()
        intent_true = y['intent_output'].numpy()
        domain_pred = y_pred[0].argmax(-1)
        intent_pred = y_pred[1].argmax(-1)
        y_pred = y_pred[2]
        y = y['ner_output'].numpy()
        # reduce dimension.
        if len(np.array(y).shape) == 3:
            y_true = np.argmax(y, -1)
        else:
            y_true = y
        y_pred = np.argmax(y_pred, -1)

        non_pad_indexes = [list(np.nonzero(np.array(y_true_row) != self.pad_value)[0]) \
            for y_true_row in list(y_true)]
            
        y_true = self.convert_idx_to_name(y_true, non_pad_indexes)
        y_pred = self.convert_idx_to_name(y_pred, non_pad_indexes)



        return y_true, y_pred, domain_true, \
            domain_pred, intent_true, intent_pred

    def score(self, y_true, y_pred, y_true_domain, \
        y_pred_domain, y_true_intent, y_pred_intent):
        """Calculate f1 score.
        Args:
            y_true (list): true sequences.
            y_pred (list): predicted sequences.
        Returns:
            score: f1 score.
        """
        score = f1_score(y_true, y_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        domain_f1_score = sk_metrics.f1_score(\
            y_true_domain, y_pred_domain, average='micro')
        intent_f1_score = sk_metrics.f1_score(\
            y_true_intent, y_pred_intent, average='micro')
        print(' - domain f1: {:04.2f}'.format(domain_f1_score * 100))
        print(' - intent f1: {:04.2f}'.format(intent_f1_score * 100))
        if self.digits:
            print(classification_report(y_true, y_pred, digits=self.digits))
        return score, domain_f1_score, intent_f1_score

    def on_epoch_end(self, epoch, logs={}):
        if self.is_fit:
            self.on_epoch_end_fit(epoch, logs)
        else:
            self.on_epoch_end_fit_generator(epoch, logs)

    def on_epoch_end_fit(self, epoch, logs={}):
        X = self.validation_data[0]
        y = self.validation_data[1]
        y_true, y_pred = self.predict(X, y)
        score = self.score(y_true, y_pred)
        logs['f1'] = score

    def on_epoch_end_fit_generator(self, epoch, logs={}):
        y_true = []
        y_pred = []
        y_true_domain = []
        y_pred_domain = []
        y_true_intent = []
        y_pred_intent = []
        for X, y in self.validation_data:
            y_true_batch, y_pred_batch, domain_true_batch, \
                domain_pred_batch, intent_true_batch, \
                    intent_pred_batch = self.predict(X, y)
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
            y_true_domain.extend(domain_true_batch)
            y_pred_domain.extend(domain_pred_batch)
            y_true_intent.extend(intent_true_batch)
            y_pred_intent.extend(intent_pred_batch)
        score, domain_score, intent_score = self.score(y_true, y_pred, y_true_domain, \
        y_pred_domain, y_true_intent, y_pred_intent)
        logs['f1'] = score
        logs['domain_f1'] = domain_score
        logs['intent_f1'] = intent_score
        