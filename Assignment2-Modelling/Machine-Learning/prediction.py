# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:21:57 2016

@author: matgo
"""

from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd

import treePrediction
import neighborsPrediction
import randomForestPrediction

fold_cv = 5


def doPredictions(train_dfs, targetLabels):
    treeVar, instances_train_tree, target_train_tree, target_test_tree, predictionsTree = treePrediction.treePrediction(
        train_dfs, targetLabels)
    neighbor, instances_train_neighbor, target_train_neighbor, target_test_neighbor, predictionsNeighbor = neighborsPrediction.neighborsPrediction(
        train_dfs, targetLabels, fold_cv)
    randFor, instances_train_randFor, target_train_randFor, target_test_randFor, predictionsRandFor = randomForestPrediction.randomForestPrediction(
        train_dfs, targetLabels, fold_cv)

    printPredictions(treeVar, instances_train_tree, target_train_tree, target_test_tree, predictionsTree,
                     neighbor, instances_train_neighbor, target_train_neighbor, target_test_neighbor,
                     predictionsNeighbor,
                     randFor, instances_train_randFor, target_train_randFor, target_test_randFor, predictionsRandFor)


def printPredictions(treeVar, instances_train_tree, target_train_tree, target_test_tree, predictionsTree,
                     neighbor, instances_train_neighbor, target_train_neighbor, target_test_neighbor,
                     predictionsNeighbor,
                     randFor, instances_train_randFor, target_train_randFor, target_test_randFor, predictionsRandFor):
    # Output the accuracy score of the model on the test set
    print("AccuracyTree= " + str(accuracy_score(target_test_tree, predictionsTree, normalize=True)))

    print("AccuracyNeighbor= " + str(accuracy_score(target_test_neighbor, predictionsNeighbor, normalize=True)))

    print("AccuracyRandFor= " + str(accuracy_score(target_test_randFor, predictionsRandFor, normalize=True)))

    # Output the confusion matrix on the test set
    confusionMatrix = confusion_matrix(target_test_tree, predictionsTree)
    print(confusionMatrix)
    print("\n\n")

    # Draw the confusion matrix
    import matplotlib.pyplot as plt

    # Show confusion matrix in a separate window
    plt.matshow(confusionMatrix)
    # plt.plot(confusionMatrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # --------------------------------------------
    # Cross-validation to Compare to Models
    # --------------------------------------------
    # run a 5 fold cross validation on this model using the full census data
    scoresTree = cross_validation.cross_val_score(treeVar, instances_train_tree, target_train_tree, cv=fold_cv)
    # the cross validaton function returns an accuracy score for each fold
    print("treeDesision:")
    print("Score by fold: " + str(scoresTree))
    # we can output the mean accuracy score and standard deviation as follows:
    print("Accuracy: %0.4f (+/- %0.2f)" % (scoresTree.mean(), scoresTree.std() * 2))
    print("\n\n")

    scoresNeighbor = cross_validation.cross_val_score(neighbor, instances_train_neighbor, target_train_neighbor,
                                                      cv=fold_cv)
    # the cross validaton function returns an accuracy score for each fold
    print("neighbor:")
    print("Score by fold: " + str(scoresNeighbor))
    # we can output the mean accuracy score and standard deviation as follows:
    print("Accuracy: %0.4f (+/- %0.2f)" % (scoresNeighbor.mean(), scoresNeighbor.std() * 2))
    print("\n\n")

    scoresRandFor = cross_validation.cross_val_score(randFor, instances_train_randFor, target_train_randFor, cv=fold_cv)
    # the cross validaton function returns an accuracy score for each fold
    print("randFor:")
    print("Score by fold: " + str(scoresRandFor))
    # we can output the mean accuracy score and standard deviation as follows:
    print("Accuracy: %0.4f (+/- %0.2f)" % (scoresRandFor.mean(), scoresRandFor.std() * 2))
    print("\n\n")


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

doPredictions(train_dfs, targetLabels)
