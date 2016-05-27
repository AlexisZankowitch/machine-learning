# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:21:27 2016

@author: matgo
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import matplotlib.pyplot as plt


def neighborsPrediction(train_dfs, targetLabels, fold_cv):
    scoresNeighbor = [0.0]
    n_neighbors = 0

    for i in range(1, 10):
        neighbor, instances_train, instances_test, target_train, target_test, scoresNeighborTmp = testScore(train_dfs,
                                                                                                            targetLabels,
                                                                                                            fold_cv,
                                                                                                            i * 2)
        if sum(scoresNeighborTmp) / len(scoresNeighborTmp) > sum(scoresNeighbor) / len(scoresNeighbor):
            scoresNeighbor = scoresNeighborTmp
            n_neighbors = i * 2
            # print(sum(scoresNeighborTmp)/len(scoresNeighborTmp))

    neighbor = KNeighborsClassifier(n_neighbors)
    neighbor.fit(train_dfs, targetLabels)

    instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs,
                                                                                                   targetLabels,
                                                                                                   test_size=0.4,
                                                                                                   random_state=0)

    predictions = neighbor.predict(instances_test)
    print("Generate random forest with: {0} neighbors".format(str(n_neighbors)))
    return neighbor, instances_train, target_train, target_test, predictions, scoresNeighbor


def testScore(train_dfs, targetLabels, fold_cv, n_neighbors):
    neighbor = KNeighborsClassifier(n_neighbors)
    neighbor.fit(train_dfs, targetLabels)

    instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs,
                                                                                                   targetLabels,
                                                                                                   test_size=0.4,
                                                                                                   random_state=0)

    scoresNeighbor = cross_validation.cross_val_score(neighbor, instances_train, target_train, cv=fold_cv)

    return neighbor, instances_train, instances_test, target_train, target_test, scoresNeighbor
