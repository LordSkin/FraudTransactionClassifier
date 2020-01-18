import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import cluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from confusion_martix import test
from data_reader import read_data

data = read_data()
X_train, X_test, Y_train, Y_test = train_test_split(data[:, :30], data[:, 30:], random_state=0)
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

X_train, Y_train = SMOTE().fit_resample(X_train, Y_train)

results = []

neural_network = make_pipeline(cluster.FeatureAgglomeration(n_clusters=16), StandardScaler(), MLPClassifier())
neural_network.fit(X_train, Y_train)
score = test("Neural network", neural_network, X_test, Y_test)
results.append(score)

random_forest = make_pipeline(StandardScaler(), RandomForestClassifier(n_jobs=-1))
random_forest.fit(X_train, Y_train)
score = test("Random forest", random_forest, X_test, Y_test)
results.append(score)

# bayes = make_pipeline(StandardScaler(), GaussianNB())
# bayes.fit(X_train, Y_train)
# score = test("naive bayes", bayes, X_test, Y_test)
# results.append(score)
#
# QDA = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis())
# QDA.fit(X_train, Y_train)
# score = test("QDA", QDA, X_test, Y_test)
# results.append(score)

logistic_regression = make_pipeline(StandardScaler(), LogisticRegression())
logistic_regression.fit(X_train, Y_train)
score = test("Logistic regression", logistic_regression, X_test, Y_test)
results.append(score)

svc = make_pipeline(StandardScaler(), SVC())
svc.fit(X_train, Y_train)
score = test("SVC", svc, X_test, Y_test)
results.append(score)

methods = ["Neural network", "Random forest", "Logistic Regression", "SVC"]

plt.bar(np.arange(len(methods)), results, align='center', alpha=0.5)
plt.xticks(np.arange(len(methods)), methods)
plt.ylabel('False negative')
plt.title('Method')

plt.show()

# Create grid search, and pass in all defined values
param_grid = [{'n_estimators': [1, 10, 100, 200]}]

gridsearch = GridSearchCV(random_forest, param_grid=param_grid)
gridsearch.get_params()
# the verbose parameter above will give output updates as the calculations are complete.
best_model = gridsearch.fit(X_train, Y_train)

print('Best n_estimators:', best_model.best_estimator_.get_params(['n_estimators']))
best_model.score()
