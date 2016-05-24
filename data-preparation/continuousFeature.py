import pandas as pd

data = pd.read_csv('../data/DataSet.csv')
continuousFeaturesList = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
dData = data.describe().transpose()
dData['Miss %'] = ""
dData['Card'] = ""

for i in range(0, len(continuousFeaturesList)):
    # Miss % calculation
    myList = data[continuousFeaturesList[i]].tolist()
    dData.iat[i, 8] = myList.count('?')
    # Cardinality calculation
    dData.iat[i, 9] = len(data[continuousFeaturesList[i]].unique())

df = pd.DataFrame(dData, columns=['count', 'Miss %', 'Card', 'min', '25%', 'mean', '50%', '75%', 'max', 'std'])
df.columns = ['Count', 'Miss %', 'Card', 'Min', '1st Qrt', 'Mean', 'Median', '3rd Qrt', 'Max', 'Std']
#storage
df.to_csv('../data/boucher-zankowitch-DQR-ContinuousFeatures.csv')
