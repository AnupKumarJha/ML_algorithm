import pandas as pd
import numpy as np
dataset=pd.read_csv("Data.csv")
print(dataset)
x=dataset.iloc[:,0:3].values
y=dataset.iloc[:,3].values
print(x)
print(y)
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues=missingvalues.fit(x[:,1:3])
x[:, 1:3]=missingvalues.transform(x[:, 1:3])
print(x,y)
#Encoding catogorical data
