import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go

# data
columnHeadings = [
    ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
     'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']]
data = pd.read_csv('../data/bank/bank-full-2.csv', index_col=False,
                   na_values=['N/A'], nrows=45211)
numeric_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
print(data)
numeric_dfs = data[numeric_features]
cat_dfs = data.drop(numeric_features)

# CONTINUOUS FEATURES
numeric_dfs = numeric_dfs.describe().transpose()
# add columns
numeric_dfs['Miss %'] = ""
numeric_dfs['Card'] = ""

for i in range(0, len(numeric_features)):
    # Miss % calculation
    myList = data[numeric_features[i]].tolist()
    numeric_dfs.iat[i, 8] = myList.count('N/A')
    # Cardinality calculation
    numeric_dfs.iat[i, 9] = len(data[numeric_features[i]].unique())

df = pd.DataFrame(numeric_dfs, columns=['count', 'Miss %', 'Card', 'min', '25%', 'mean', '50%', '75%', 'max', 'std'])
df.columns = ['Count', 'Miss %', 'Card', 'Min', '1st Qrt', 'Mean', 'Median', '3rd Qrt', 'Max', 'Std']
# storage
df.to_csv('../data/bank/gouhier-zankowitch-DQR-ContinuousFeatures.csv')

# CATEGORICAL FEATURES
cat_dfs = cat_dfs.describe().transpose()
categoricalFeaturesList = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                           'native-country']

cat_dfs['Mode %'] = None
cat_dfs['Mode2'] = None
cat_dfs['Mode2 freq'] = None
cat_dfs['Mode2 %'] = None
cat_dfs['Miss'] = None

for i in range(0, len(categoricalFeaturesList)):
    cat_dfs.iat[i, 4] = round(data[categoricalFeaturesList[i]].value_counts()[0] / cat_dfs.iat[i, 0] * 100, 1)
    cat_dfs.iat[i, 5] = data[categoricalFeaturesList[i]].value_counts().keys()[1]
    cat_dfs.iat[i, 6] = data[categoricalFeaturesList[i]].value_counts()[1]
    cat_dfs.iat[i, 7] = round(data[categoricalFeaturesList[i]].value_counts()[1] / cat_dfs.iat[i, 0] * 100, 1)
    myList = data[categoricalFeaturesList[i]].tolist()
    cat_dfs.iat[i, 8] = round(myList.count('N/A') / cat_dfs.iat[i, 0] * 100, 1)

df = pd.DataFrame(cat_dfs,
                  columns=['count', 'Miss', 'unique', 'top', 'freq', 'Mode %', 'Mode2', 'Mode2 freq', 'Mode2 %'])
df.columns = ['Count', 'Miss %', 'Card', 'Mode', 'Mode freq', 'Mode %', '2nd Mode', '2nd Mode Freq', '2nd Mode %']
# storage
df.to_csv('../data/bank/gouhier-zankowitch-DQR-CategoricalFeatures.csv')

# FIGURES GENERATION

for col in data.columns:
    # if the column is continuous and that they have a cardinality higher than 10, histogram !
    if data[col].dtypes == 'int64' and data[col].value_counts().__len__() >= 10:
        data = [
            go.Histogram(
                x=data[col]
            )
        ]
    # Else, it's just bar plot
    else:
        data = [
            go.Bar(
                x=data[col].value_counts().keys(),
                y=data[col].value_counts().values
            )
        ]
    plot_url = py.plot(data, filename="../data/figures/bank/" + col + ".html")
