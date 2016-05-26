# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:20:27 2016

@author: matgo
"""
from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import utilities


def treePrediction(train_dfs, targetLabels):
    # Split the data: 10%-90% training : 90%-10% test set
    instances_train = []
    instances_test = []
    target_train = []
    target_test = []
    for x in range(1, 10):
        i_train, i_test, t_train, t_test = cross_validation.train_test_split(
            train_dfs,
            targetLabels,
            test_size=x / 10,
            random_state=0)
        instances_train.append(i_train)
        instances_test.append(i_test)
        target_train.append(t_train)
        target_test.append(t_test)

    # --------------------------------------------
    # Cross-validation to Compare to Models
    # --------------------------------------------
    # , [], [], [], [], [], [], [], [], []
    nb_cv = 5
    # decision tree
    dec_tree_model_2 = tree.DecisionTreeClassifier(criterion='entropy')
    dec_tree_model_3 = tree.DecisionTreeClassifier(criterion='gini')

    plt.subplot(211)
    scores_entropy = utilities.scores_calculation(instances_train, dec_tree_model_2, target_train, nb_cv)
    mean_entropy = utilities.mean_calculation(scores_entropy, instances_train)
    plt.plot(range(1, len(mean_entropy) + 1), mean_entropy)
    plt.title('Entropy accuracy')

    plt.subplot(212)
    # for a comparison we will do the same experiment using a decision tree that uses the Gini impurity metric
    scores_gini = utilities.scores_calculation(scores_entropy, instances_train, dec_tree_model_3, target_train, nb_cv)
    mean_gini = utilities.mean_calculation(scores_gini, instances_train)
    plt.plot(range(1, len(mean_gini) + 1), mean_gini)
    plt.title('Gini accuracy')

    # must efficient rate for training set and tree method
    m_ent = max(mean_entropy)
    m_gin = max(mean_gini)
    if m_ent > m_gin:
        tr_set_rate = mean_entropy.index(m_ent) + 1
        tree_str = "entropy"
    else:
        tr_set_rate = mean_gini.index(m_gin) + 1
        tree_str = "gini"

    # --------------------------------------------
    # Test
    # --------------------------------------------
    # retrain with most efficient rate
    instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(
        train_dfs,
        targetLabels,
        test_size=1 - tr_set_rate * 0.1,
        random_state=0)
    print("generate decision tree: " + tree_str + " training set: " + str(tr_set_rate * 10) + "%")

    # tree generation
    tree_final = tree.DecisionTreeClassifier(criterion=tree_str)
    # fit the model using just the test set
    tree_final.fit(instances_train, target_train)
    # Use the model to make predictions for the test set queries
    predictions = tree_final.predict(instances_test)
    # Output the accuracy score of the model on the test set
    print("Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))
    # Output the confusion matrix on the test set
    confusionMatrix = confusion_matrix(target_test, predictions)
    print(confusionMatrix)
    print("\n\n")

    # Draw the confusion matrix
    # Show confusion matrix in a separate window
    plt.matshow(confusionMatrix)
    # plt.plot(confusionMatrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return tree_final, instances_train, target_train, target_test, predictions
