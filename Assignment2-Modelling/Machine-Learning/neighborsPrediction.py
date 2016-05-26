# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:21:27 2016

@author: matgo
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import matplotlib.pyplot as plt


from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd

def neighborsPrediction(train_dfs, targetLabels, fold_cv):
    
    scoresNeighbor = [0.0]
    n_neighbors = 0
    
    test = []    
    
    for i in range(1,15):
        neighbor, instances_train, instances_test, target_train, target_test, scoresNeighborTmp = testScore(train_dfs, targetLabels, fold_cv, i)
        if sum(scoresNeighborTmp)/len(scoresNeighborTmp)>sum(scoresNeighbor)/len(scoresNeighbor) :
           scoresNeighbor=scoresNeighborTmp
           n_neighbors=i
           test.append(sum(scoresNeighborTmp)/len(scoresNeighborTmp))
           print(sum(scoresNeighborTmp)/len(scoresNeighborTmp))

    print(len(test))    
    
    plt.plot(range(1,9),test)
    plt.show()
    print(n_neighbors)
    
    predictions = neighbor.predict(instances_test)
    
    return neighbor, instances_train, target_train, target_test, predictions
    
def testScore(train_dfs, targetLabels, fold_cv, n_neighbors):
    neighbor = KNeighborsClassifier(n_neighbors)
    neighbor.fit(train_dfs, targetLabels)
    
    instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels,
                                                                                                   test_size=0.2,
                                                                                                   random_state=0)
    
    scoresNeighbor = cross_validation.cross_val_score(neighbor, instances_train, target_train, cv=fold_cv)
    
    return neighbor, instances_train, instances_test, target_train, target_test, scoresNeighbor
    
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

neighborsPrediction(train_dfs, targetLabels, fold_cv)