from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
import numpy as np

from data_reader import read_data

data = read_data()
X_train, X_test, Y_train, Y_test = train_test_split(data[:, :30], data[:, 30:], random_state=0)
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

X_train, Y_train = SMOTE().fit_resample(X_train, Y_train)

pipeline = make_pipeline(StandardScaler(), MLPClassifier())
pipeline.fit(X_train, Y_train)
print("score: {0}".format(accuracy_score(pipeline.predict(X_test), Y_test)))
cm = confusion_matrix(Y_test, pipeline.predict(X_test))
print("False positive:{0}/{1}".format(cm[0, 1], cm[0, 0] + cm[0, 1]))
print("False negative:{0}/{1}".format(cm[1, 0], cm[1, 1] + cm[1, 0]))
