import json
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import joblib
from sklearn.metrics import accuracy_score


################################################################################################################


def test_get_browsers():
    X_test = pd.read_csv('data/prepared/X_test.csv')
    y_test = pd.read_csv('data/prepared/y_test.csv')

    clf = joblib.load('model/model.joblib')
    prediction = clf.predict(X_test)

    # check accuracy
    accuracy = accuracy_score(y_test, prediction)
    assert accuracy > 0.1
