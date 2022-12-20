import json
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import joblib
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    X_test = pd.read_csv('data/prepared/X_test.csv')
    y_test = pd.read_csv('data/prepared/y_test.csv')

    clf = joblib.load('model/model.joblib')
    prediction = clf.predict(X_test)

    # check accuracy
    accuracy_score = accuracy_score(y_test, prediction)
    print(accuracy_score)

    json.dump(
        obj={
            'accuracy_score': accuracy_score
        },
        fp=open('metrics/predict.json', 'w')
    )
