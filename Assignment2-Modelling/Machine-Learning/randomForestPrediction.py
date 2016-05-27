# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:22:39 2016

@author: matgo
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def randomForestPrediction(train_dfs, targetLabels, fold_cv):
    scoresRandFor = [0.0]
    n_estimators = 0

    for i in range(1, 10):
        randFor, instances_train, instances_test, target_train, target_test, scoresRandForTmp = testScore(train_dfs,
                                                                                                          targetLabels,
                                                                                                          fold_cv, i)
        if sum(scoresRandForTmp) / len(scoresRandForTmp) > sum(scoresRandFor) / len(scoresRandFor):
            scoresRandFor = scoresRandForTmp
            n_estimators = i
            # print(sum(scoresRandForTmp)/len(scoresRandForTmp))

    randFor = RandomForestClassifier(n_estimators)
    randFor.fit(train_dfs, targetLabels)
    instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs,
                                                                                                   targetLabels,
                                                                                                   test_size=0.4,
                                                                                                   random_state=0)
    predictions = randFor.predict(instances_test)
    return randFor, instances_train, target_train, target_test, predictions, scoresRandFor


def testScore(train_dfs, targetLabels, fold_cv, n_estimators):
    randFor = RandomForestClassifier(n_estimators)
    randFor.fit(train_dfs, targetLabels)

    instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs,
                                                                                                   targetLabels,
                                                                                                   test_size=0.4,
                                                                                                   random_state=0)

    scoresRandFor = cross_validation.cross_val_score(randFor, instances_train, target_train, cv=fold_cv)

    return randFor, instances_test, instances_train, target_train, target_test, scoresRandFor
