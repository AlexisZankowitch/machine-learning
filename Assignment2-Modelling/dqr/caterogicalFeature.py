import pandas as pd

data = pd.read_csv('../data/bank-additional-full-2.csv').select_dtypes(include=[object])
dData = data.describe().transpose()
dData = dData.drop('y')

categoricalFeaturesList = ['job', 'marital', 'education', 'default',
                           'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
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
    dData.iat[i, 8] = round(myList.count('N/A') / dData.iat[i, 0] * 100, 1)

df = pd.DataFrame(dData, columns=['count', 'Miss', 'unique', 'top', 'freq', 'Mode %', 'Mode2', 'Mode2 freq', 'Mode2 %'])
df.columns = ['Count', 'Miss %', 'Card', 'Mode', 'Mode freq', 'Mode %', '2nd Mode', '2nd Mode Freq', '2nd Mode %']
#storage
df.to_csv('../data/gouhier-zankowitch-DQR-CategoricalFeatures.csv')
