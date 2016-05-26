# coding: utf-8


from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean_calculation(scores):
    mean_table = []
    for i in range(0, len(instances_train)):
        m = 0
        for j in range(0, len(scores)):
            m += scores[j][i]
        m /= len(scores)
        mean_table.append(m)
    return mean_table


# Reading the dataset from a local file
# ---------------------------------------------
features_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                  'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome',
                  'y']
censusData = pd.read_csv("../data/bank/bank-full-2.csv", index_col=False, na_values=['N/A'], nrows=45211,
                         usecols=features_names)

# Extract Target Feature
targetLabels = censusData['y']
# Extract Numeric Descriptive Features
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
numeric_dfs = censusData[numeric_features]
# Extract Categorical Descriptive Features
cat_dfs = censusData.drop(numeric_features + ['y'], axis=1)

# No missing values
# transpose into array of dictionaries (one dict per instance) of feature:level pairs
cat_dfs = cat_dfs.T.to_dict().values()
# convert to numeric encoding
vectorizer = DictVectorizer(sparse=False)
vec_cat_dfs = vectorizer.fit_transform(cat_dfs)

# Merge Categorical and Numeric Descriptive Features
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))

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
scores_entropy = [[], [], [], [], [], [], [], [], [], [], [], []]
scores_gini = [[], [], [], [], [], [], [], [], [], [], [], []]
# decision tree
decTreeModel2 = tree.DecisionTreeClassifier(criterion='entropy')
decTreeModel3 = tree.DecisionTreeClassifier(criterion='gini')

plt.subplot(211)
for y in range(0, len(scores_entropy)):
    for x in range(0, len(instances_train)):
        # run a 5 fold cross validation on this model using the full census data
        scores_x_entropy = cross_validation.cross_val_score(decTreeModel2, instances_train[x], target_train[x],
                                                            cv=nb_cv)
        scores_entropy[y].append(scores_x_entropy.mean())
        # Show entropy accuracy
mean_entropy = mean_calculation(scores_entropy)
plt.plot(range(1, len(mean_entropy) + 1), mean_entropy)
plt.title('Entropy accuracy')

plt.subplot(212)
# for a comparison we will do the same experiment using a decision tree that uses the Gini impurity metric
for y in range(0, len(scores_gini)):
    for x in range(0, len(instances_train)):
        scores_x_gini = cross_validation.cross_val_score(decTreeModel3, instances_train[x], target_train[x], cv=nb_cv)
        scores_gini[y].append(scores_x_gini.mean())
        # Show gini accuracy
mean_gini = mean_calculation(scores_gini)
plt.plot(range(1, len(mean_gini) + 1), mean_gini)
plt.title('Gini accuracy')

# must efficient rate for training set and tree method
m_ent = max(mean_entropy)
m_gin = max(mean_gini)
if m_ent > m_gin:
    tr_set_rate = mean_entropy.index(m_ent)+1
    tree_str = "entropy"
else:
    tr_set_rate = mean_gini.index(m_gin)+1
    tree_str = "gini"


# --------------------------------------------
# Test
# --------------------------------------------
# retrain with most efficient rate
instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(
    train_dfs,
    targetLabels,
    test_size=1-tr_set_rate*0.1,
    random_state=0)
print("generate decision tree: " + tree_str + " training set: " + str(tr_set_rate*10) + "%")


# tree generation
decTreeModel3 = tree.DecisionTreeClassifier(criterion=tree_str)
# fit the model using just the test set
decTreeModel3.fit(instances_train, target_train)
# Use the model to make predictions for the test set queries
predictions = decTreeModel3.predict(instances_test)
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
