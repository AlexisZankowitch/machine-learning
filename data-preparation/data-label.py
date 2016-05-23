import pandas as pd
import pprint as pp

data = pd.read_csv('../data/DataSet.csv')

#print(data)

cat_rows = ['workclass', 'education', 'marital-status',	'occupation', 'relationship', 'race', 'sex', 'native-country']
info_cat = dict()

for row in cat_rows:
    current = data[row]
    current_list = current.tolist()
    info_cat[row] = dict()
    info_cat[row]['Name'] = row
    info_cat[row]['Count'] = len(current)
    info_cat[row]['Missing'] = current_list.count(' ?')/info_cat[row]['Count']*100
    info_cat[row]['Card'] = len(current.unique())
    info_cat[row]['Mode'] = current.value_counts().keys()[0]
    info_cat[row]['Mode_Count'] = current.value_counts()[0]
    info_cat[row]['Mode_%'] = current.value_counts()[0]/info_cat[row]['Count']*100
    info_cat[row]['Mode2'] = current.value_counts().keys()[1]
    info_cat[row]['Mode2_Count'] = current.value_counts()[1]
    info_cat[row]['Mode2_%'] = current.value_counts()[1]/info_cat[row]['Count']*100

pp.pprint(info_cat)
