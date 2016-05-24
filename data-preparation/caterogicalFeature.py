import pandas as pd

data = pd.read_csv('../data/DataSet.csv').select_dtypes(include=[object])
dData = data.describe().transpose()
dData = dData.drop('id')
dData = dData.drop('target')

categoricalFeaturesList = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                           'native-country']
dData['Mode %'] = None
dData['Mode2'] = None
dData['Mode2 freq'] = None
dData['Mode2 %'] = None
dData['Miss'] = None

for i in range(0, len(categoricalFeaturesList)):
    dData.iat[i, 4] = data[categoricalFeaturesList[i]].value_counts()[0] / dData.iat[i, 0] * 100
    dData.iat[i, 5] = data[categoricalFeaturesList[i]].value_counts().keys()[1]
    dData.iat[i, 6] = data[categoricalFeaturesList[i]].value_counts().keys()[1]
    dData.iat[i, 7] = data[categoricalFeaturesList[i]].value_counts()[1] / dData.iat[i, 0] * 100
    myList = data[categoricalFeaturesList[i]].tolist()
    dData.iat[i, 8] = myList.count(' ?') / dData.iat[i, 0] * 100

df = pd.DataFrame(dData, columns=['count', 'Miss', 'unique', 'top', 'freq', 'Mode %', 'Mode2', 'Mode2 freq', 'Mode2 %'])
df.columns = ['Count', 'Miss %', 'Card', 'Mode', 'Mode freq', 'Mode %', '2nd Mode', '2nd Mode Freq', '2nd Mode %']
#storage
df.to_csv('../data/boucher-zankowitch-DQR-CategoricalFeatures.csv')
