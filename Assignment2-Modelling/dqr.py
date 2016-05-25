import pandas as pd

data = pd.read_csv('../data/bank/bank-full.csv').select_dtypes(include=[object])
dData = data.describe().transpose()
dData = dData.drop('y')
dData = data.describe().transpose()

# CONTINUOUS FEATURES
continuousFeaturesList = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# add columns
dData['Miss %'] = ""
dData['Card'] = ""

for i in range(0, len(continuousFeaturesList)):
    # Miss % calculation
    myList = data[continuousFeaturesList[i]].tolist()
    dData.iat[i, 8] = myList.count('N/A')
    # Cardinality calculation
    dData.iat[i, 9] = len(data[continuousFeaturesList[i]].unique())

df = pd.DataFrame(dData, columns=['count', 'Miss %', 'Card', 'min', '25%', 'mean', '50%', '75%', 'max', 'std'])
df.columns = ['Count', 'Miss %', 'Card', 'Min', '1st Qrt', 'Mean', 'Median', '3rd Qrt', 'Max', 'Std']
# storage
df.to_csv('../data/bank/gouhier-zankowitch-DQR-ContinuousFeatures.csv')

# CATEGORICAL FEATURES
categoricalFeaturesList = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                           'native-country']
dData['Mode %'] = None
dData['Mode2'] = None
dData['Mode2 freq'] = None
dData['Mode2 %'] = None
dData['Miss'] = None

for i in range(0, len(categoricalFeaturesList)):
    dData.iat[i, 4] = round(data[categoricalFeaturesList[i]].value_counts()[0] / dData.iat[i, 0] * 100, 1)
    dData.iat[i, 5] = data[categoricalFeaturesList[i]].value_counts().keys()[1]
    dData.iat[i, 6] = data[categoricalFeaturesList[i]].value_counts()[1]
    dData.iat[i, 7] = round(data[categoricalFeaturesList[i]].value_counts()[1] / dData.iat[i, 0] * 100, 1)
    myList = data[categoricalFeaturesList[i]].tolist()
    dData.iat[i, 8] = round(myList.count(' ?') / dData.iat[i, 0] * 100, 1)

df = pd.DataFrame(dData, columns=['count', 'Miss', 'unique', 'top', 'freq', 'Mode %', 'Mode2', 'Mode2 freq', 'Mode2 %'])
df.columns = ['Count', 'Miss %', 'Card', 'Mode', 'Mode freq', 'Mode %', '2nd Mode', '2nd Mode Freq', '2nd Mode %']
# storage
df.to_csv('../data/boucher-zankowitch-DQR-CategoricalFeatures.csv')
