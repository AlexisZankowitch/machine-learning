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
import prediction






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

prediction.doPredictions(train_dfs, targetLabels)

