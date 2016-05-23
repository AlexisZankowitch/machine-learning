import pandas as pd
import numpy

import plotly.offline as py
import plotly.graph_objs as go


input_file = "./data/DataSet.csv"
output_cont_file = "./data/baronnet-regnault-DQR-ContinuousFeatures.csv"
output_cat_file = "./data/baronnet-regnault-DQR-CategorialFeatures.csv"

df = pd.read_csv(input_file)

dcont = df.describe().transpose();
dcont['Card'] = 0
dcont['Miss'] = 0
miss  = dcont['Miss']
dcont.drop(labels=['Miss'], axis=1,inplace = True)
dcont.insert(1,'Miss', miss)

card  = dcont['Card']
dcont.drop(labels=['Card'], axis=1,inplace = True)
dcont.insert(2,'Card', card)

std = dcont['std']
dcont.drop(labels=['std'], axis=1,inplace = True)
dcont.insert(9,'std', std)

mean = dcont['mean']
dcont.drop(labels=['mean'], axis=1,inplace = True)
dcont.insert(5,'mean', mean)

dcat = df.describe(include=['O']).transpose()
dcat['Mode %'] = 0
dcat['2nd Mode'] = 0
dcat['2nd Mode freq'] = 0
dcat['2nd Mode %'] = 0

dcat['Miss'] = 0
miss  = dcat['Miss']
dcat.drop(labels=['Miss'], axis=1,inplace = True)
dcat.insert(1,'Miss', miss)



for col in df.columns:
    counts = df[col].value_counts(sort=1)

    card = counts.index.size

    try:
        val = counts.loc[' ?']
        card -= 1
    except Exception:
        val = 0

    print(counts)

    if df[col].dtype != numpy.int64:
        dcat.ix[col, 'Miss']=val/df[col].count()
        dcat.ix[col, 'Mode %'] = counts.iloc[0]/df[col].count()
        dcat.ix[col, 'unique'] = card

        try:
            dcat.ix[col, '2nd Mode'] = counts.index[1]
            dcat.ix[col, '2nd Mode freq'] = counts.iloc[1]
            dcat.ix[col, '2nd Mode %'] = counts.iloc[1]/df[col].count()
        except Exception:
            pass
    else:
        dcont.ix[col, 'Card'] = card
        dcont.ix[col, 'Miss']=val

#rename categories
dcont.columns = ['Count', 'Miss %', 'Card','Min','1st Qrt','Mean','Median','3rd Qrt' , 'Max', 'Std Dev']
dcat.columns = ['Count', 'Miss %', 'Card', 'Mode', 'Mode Freq', 'Mode %', '2nd Mode', '2nd Mode freq', '2nd Mode %']

#create output files
dcont.to_csv(output_cont_file)
dcat.to_csv(output_cat_file)

for col in df.columns:
    #if the column is continuous and that they have a cardinality higher than 10, histogram !
    if df[col].dtypes == 'int64' and df[col].value_counts().__len__() >= 10:
        data = [
            go.Scatter(
                y=df[col].value_counts().keys(),
                x=df[col].value_counts().values
            )
        ]
    #Else, it's just bar plot
    else:
        data = [
            go.Bar(
                x=df[col].value_counts().keys(),
                y=df[col].value_counts().values
            )
        ]
    plot_url = py.plot(data, filename="./data/html/"+col+".html")
