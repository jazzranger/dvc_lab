import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

import joblib

if __name__ == '__main__':
    X_train = pd.read_csv('data/prepared/X_train.csv')
    y_train = pd.read_csv('data/prepared/y_train.csv')

    classifier = RandomForestClassifier(n_estimators = 200, max_depth=2, random_state=0)
    # Applying classifier on training data
    classifier = classifier.fit(X_train, y_train)
    joblib.dump(classifier, 'model/model.joblib')
