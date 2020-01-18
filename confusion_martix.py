import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def test(method_name, pipeline, X_test, Y_test):
    accuracy = accuracy_score(pipeline.predict(X_test), Y_test)
    print("score: {0}".format(accuracy))
    cm = confusion_matrix(Y_test, pipeline.predict(X_test))
    print("Wyniki dla {0}".format(method_name))
    print("False positive: {0}/{1}".format(cm[0, 1], len(Y_test) - np.sum(Y_test)))
    print("False negative: {0}/{1}".format(cm[1, 0], np.sum(Y_test)))
    print()
    return cm[0, 1]
