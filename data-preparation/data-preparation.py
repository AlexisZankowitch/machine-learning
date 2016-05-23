import pandas as pd

data = pd.read_csv('../data/DataSet.csv')

continuousFeaturesList = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

continuousFeature = dict()
continuousFeatureKey = ['Count', '% Miss', 'Card', 'min', '1st Qrt', 'Mean', 'Median', '3rd Qrt', 'Max', 'Std Dev']

# for key in continuousFeaturesList: