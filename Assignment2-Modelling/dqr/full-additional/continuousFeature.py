import pandas as pd

data = pd.read_csv('../../data/bank/bank-additional-full-2.csv')
continuousFeaturesList = ["age", "duration", "campaign",
                          "pdays", "previous", "emp.var.rate",
                          "euribor3m", "nr.employed"]
dData = data.describe().transpose()
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
#storage
df.to_csv('../../data/dqr/gouhier-zankowitch-DQR-ContinuousFeatures-full-additional.csv')

