# coding: utf-8


from pandas import DataFrame
from sklearn import preprocessing
from sklearn import tree
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

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

# --------------------------------------------
# Hold-out Test Set + Confusion Matrix
# --------------------------------------------
# define a decision tree model using entropy based information gain
decTreeModel2 = tree.DecisionTreeClassifier(criterion='entropy')
# Split the data: 60% training : 40% test set
instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels,
                                                                                               test_size=0.2,
                                                                                               random_state=0)
# fit the model using just the test set
decTreeModel2.fit(instances_train, target_train)
# Use the model to make predictions for the test set queries
predictions = decTreeModel2.predict(instances_test)
# Output the accuracy score of the model on the test set
print("Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))
# Output the confusion matrix on the test set
confusionMatrix = confusion_matrix(target_test, predictions)
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
scores = cross_validation.cross_val_score(decTreeModel2, instances_train, target_train, cv=fold_cv)
# the cross validaton function returns an accuracy score for each fold
print("Entropy based Model:")
print("Score by fold: " + str(scores))
# we can output the mean accuracy score and standard deviation as follows:
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("\n\n")
# for a comparison we will do the same experiment using a decision tree that uses the Gini impurity metric
decTreeModel3 = tree.DecisionTreeClassifier(criterion='gini')
scores = cross_validation.cross_val_score(decTreeModel3, instances_train, target_train, cv=fold_cv)
print("Gini based Model:")
print("Score by fold: " + str(scores))
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
