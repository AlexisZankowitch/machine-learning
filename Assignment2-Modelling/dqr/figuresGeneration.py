import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go

dataCsv = pd.read_csv('../data/bank-full-2.csv')

for col in dataCsv.columns:
    #if the column is continuous and that they have a cardinality higher than 10, histogram !
    if dataCsv[col].dtypes == 'int64' and dataCsv[col].value_counts().__len__() >= 10:
        data = [
            go.Histogram(
                x=dataCsv[col]
            )
        ]
    #Else, it's just bar plot
    else:
        data = [
            go.Bar(
                x=dataCsv[col].value_counts().keys(),
                y=dataCsv[col].value_counts().values
            )
        ]
    plot_url = py.plot(data, filename="../data/plots/"+col+".html")
