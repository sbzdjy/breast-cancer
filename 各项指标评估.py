from sklearn import metrics

from data_import import class_names, val_ds
from model_train import model
from 混淆矩阵 import val_label, val_pre


def test_accuracy_report(model):
    print(metrics.classification_report(val_label, val_pre, target_names=class_names))
    score = model.evaluate(val_ds, verbose=0)
    print('Loss function: %s, accuracy:' % score[0], score[1])


test_accuracy_report(model)

