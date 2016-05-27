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
import matplotlib.pyplot as plt
import treePrediction
import neighborsPrediction
import randomForestPrediction

fold_cv = 5
models_order = ['tree decision', 'neighbors', 'random forest']
accuracies = []
c_m_all = []


# todo display parameters which have been chosen for each models

def doPredictions(train_dfs, targetLabels, fold_cv):
    tree_model, instances_train_tree, target_train_tree, target_test_tree, predictions_tree = treePrediction.treePrediction(
        train_dfs, targetLabels)
    neighbor, instances_train_neighbor, target_train_neighbor, target_test_neighbor, predictions_neighbor, scoresNeighbor = neighborsPrediction.neighborsPrediction(
        train_dfs, targetLabels, fold_cv)
    randFor, instances_train_randFor, target_train_randFor, target_test_randFor, predictionsRandFor, scoresRandFor = randomForestPrediction.randomForestPrediction(
        train_dfs, targetLabels, fold_cv)

    # confusion matrix and accuracy for each model
    c_m_tree, accuracy_tree = final_model_decision(target_test_tree, predictions_tree)
    accuracies.append(accuracy_tree)
    c_m_all.append(c_m_tree)
    print("Confusion matrix tree")
    print(c_m_tree)
    print("Accuracy= " + str(accuracy_tree))
    c_m_neighbor, accuracy_neighbor = final_model_decision(target_test_neighbor, predictions_neighbor)
    accuracies.append(accuracy_neighbor)
    c_m_all.append(c_m_neighbor)
    print("Confusion matrix neighbor")
    print(c_m_neighbor)
    print("Accuracy= " + str(accuracy_neighbor))
    c_m_rand_for, accuracy_rand_for = final_model_decision(target_test_randFor, predictionsRandFor)
    accuracies.append(accuracy_rand_for)
    c_m_all.append(c_m_rand_for)
    print("Confusion matrix random forest")
    print(c_m_rand_for)
    print("Accuracy= " + str(accuracy_rand_for))

    # final test to determine which model is the best
    index_accuracy = accuracies.index(max([accuracy_tree, accuracy_neighbor, accuracy_rand_for]))

    # display
    print("The best model is: " + models_order[index_accuracy] + " its accuracy is: " + str(accuracies[index_accuracy]))
    plt.matshow(c_m_all[index_accuracy])
    # plt.plot(confusionMatrix)
    plt.title('Confusion matrix' + models_order[index_accuracy])
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def final_model_decision(target_test, predictions):
    # Output the accuracy score of the model on the test set
    model_accuracy = accuracy_score(target_test, predictions, normalize=True)
    # Output the confusion matrix on the test set
    c_m = confusion_matrix(target_test, predictions)
    return c_m, model_accuracy


# Reading the dataset from a local file
# ---------------------------------------------
censusData = pd.read_csv("../data/bank/bank-additional-full-2.csv", index_col=False, na_values=['N/A'], nrows=45211)

# Extract Target Feature
targetLabels = censusData['y']
# Extract Numeric Descriptive Features
numeric_features = ["age", "campaign",
                    "pdays", "previous", "emp.var.rate",
                    "euribor3m", "nr.employed"]
numeric_dfs = censusData[numeric_features]
# Extract Categorical Descriptive Features
cat_dfs = censusData.drop(numeric_features + ['y', "duration"], axis=1)

# There's no missing value
# transpose into array of dictionaries (one dict per instance) of feature:level pairs
cat_dfs = cat_dfs.T.to_dict().values()
# convert to numeric encoding
vectorizer = DictVectorizer(sparse=False)
vec_cat_dfs = vectorizer.fit_transform(cat_dfs)

# Merge Categorical and Numeric Descriptive Features
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))

doPredictions(train_dfs, targetLabels, fold_cv)
