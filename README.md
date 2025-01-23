# python
pip install pandas statsmodels
import pandas as pd
import statsmodels.api as sm
data = pd.read_excel('C:/Users/rihab/OneDrive/Documents/Tutorial R/Data.xlsx')
X = Data[['B1, 'C1', 'E1']] 
y = data['A1']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
