# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:20:27 2016

@author: matgo
"""
from sklearn import tree
from sklearn import cross_validation

def treePrediction(train_dfs, targetLabels):
    # --------------------------------------------
    # Hold-out Test Set + Confusion Matrix
    # --------------------------------------------
    # define a decision tree model using entropy based information gain
    treeVar = tree.DecisionTreeClassifier(criterion='entropy')
    # Split the data: 60% training : 40% test set
    instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels,
                                                                                                   test_size=0.2,
                                                                                                   random_state=0)
    # fit the model using just the test set
    treeVar.fit(instances_train, target_train)
    # Use the model to make predictions for the test set queries
    predictions = treeVar.predict(instances_test)
    
    return treeVar, instances_train, target_train, target_test, predictions