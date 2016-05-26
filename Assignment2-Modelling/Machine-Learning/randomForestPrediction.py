# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:22:39 2016

@author: matgo
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import matplotlib.pyplot as plt


from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd

def randomForestPrediction(train_dfs, targetLabels, fold_cv):    
    
    scoresRandFor = [0.0]
    n_estimators = 0
    
    test = []     
    
    for i in range(1,15):
        randFor, instances_train, instances_test, target_train, target_test, scoresRandForTmp = testScore(train_dfs, targetLabels, fold_cv, i)
        if sum(scoresRandForTmp)/len(scoresRandForTmp)>sum(scoresRandFor)/len(scoresRandFor) :
           scoresRandFor=scoresRandForTmp
           n_estimators=i
           test.append(sum(scoresRandForTmp)/len(scoresRandForTmp))
           print(sum(scoresRandForTmp)/len(scoresRandForTmp))

    
    print(len(test))    
    
    plt.plot(range(1,9),test)
    plt.show()
    print(n_estimators)    
    
    predictions = randFor.predict(instances_test)
    
    
    return randFor, instances_train, target_train, target_test, predictions
    
def testScore(train_dfs, targetLabels, fold_cv, n_estimators):
    randFor = RandomForestClassifier(n_estimators)
    randFor.fit(train_dfs, targetLabels)
    
    instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels,
                                                                                                   test_size=0.2,
                                                                                                   random_state=0)
    
    scoresRandFor = cross_validation.cross_val_score(randFor, instances_train, target_train, cv=fold_cv)
    
    return randFor, instances_test, instances_train, target_train, target_test, scoresRandFor

fold_cv = 5

# Reading the dataset from a local file
# ---------------------------------------------
censusData = pd.read_csv("../data/bank/bank-additional-full-2.csv", index_col=False, na_values=['N/A'], nrows=45211)

# Extract Target Feature
targetLabels = censusData['y']
# Extract Numeric Descriptive Features
numeric_features = ["age", "duration", "campaign",
                    "pdays", "previous", "emp.var.rate",
                    "euribor3m", "nr.employed"]
numeric_dfs = censusData[numeric_features]
# Extract Categorical Descriptive Features
cat_dfs = censusData.drop(numeric_features + ['y'], axis=1)

# There's no missing value
# transpose into array of dictionaries (one dict per instance) of feature:level pairs
cat_dfs = cat_dfs.T.to_dict().values()
# convert to numeric encoding
vectorizer = DictVectorizer(sparse=False)
vec_cat_dfs = vectorizer.fit_transform(cat_dfs)

# Merge Categorical and Numeric Descriptive Features
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))

randomForestPrediction(train_dfs, targetLabels, fold_cv)