import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go

data = pd.read_csv('../data/DataSet.csv')

for col in data.columns:
    #if the column is continuous and that they have a cardinality higher than 10, histogram !
    if data[col].dtypes == 'int64' and data[col].value_counts().__len__() >= 10:
        data = [
            go.Scatter(
                y=data[col].value_counts().keys(),
                x=data[col].value_counts().values
            )
        ]
    #Else, it's just bar plot
    else:
        data = [
            go.Bar(
                x=data[col].value_counts().keys(),
                y=data[col].value_counts().values
            )
        ]
    plot_url = py.plot(data, filename="../data/figures/"+col+".html")