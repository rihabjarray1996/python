# python
# Title: Drivers of CO₂ Emissions Model: the determinants of CO₂ emissions: GDP per capita, energy consumption, urbanization rate, and industrial production. Useful for policy-relevant environmental research.
# Description: To model the determinants of CO₂ emissions in EU between 1960-2023: GDP per capita, energy consumption, urbanization rate, and industrial production can be incorporated into a multivariate regression and econometric framework. This kind of model could provide insights into how each factor contributes to CO₂ emissions and is widely applicable in policy-relevant environmental research. 
# All data is taken from World Bank database
# Load your data into a DataFrame
pip install pandas statsmodels
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf #not sure which to use
data = pd.read_excel('C:/Users/rihab/OneDrive/Documents/Tutorial R/Data.xlsx')
X = Data[['B1, 'C1', 'E1']] 
y = data['A1']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# 1. Define the formula for OLS regression including an interaction term
# 'energy_consumption * industrial_production' interaction term
formula = 'CO2 emissions ~ energy_consumption * industrial_production'

# Fit the OLS model
model = smf.ols(formula=formula, data=data).fit()

# Print the summary of the regression
print(model.summary())

 # panel data analysis
from linearmodels.panel import PanelOLS, RandomEffects

# Load panel data into a DataFrame
df = data

# Set the index
df = df.set_index(['GDP per capita', 'year'])

# 2. Fixed effects model
fe_model = PanelOLS.from_formula('CO2 emissions ~ 1 + energy_consumption + industrial_production + EntityEffects', df)
fe_results = fe_model.fit()
print(fe_results.summary)

# 3. Random effects model
re_model = RandomEffects.from_formula('CO emissions' ~ 1 + energy_consumption + industrial_production', df)
re_results = re_model.fit()
print(re_results.summary)

 # 4. Machine learning analysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load data into a DataFrame
df = data
# Define features and target variable
X = df[['energy_consumption', 'industrial_production']]
y = df['CO2 emissions']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print(f'Random Forest MSE: {mean_squared_error(y_test, rf_predictions)}')

# 6. Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
print(f'Gradient Boosting MSE: {mean_squared_error(y_test, gb_predictions)}')

# 7. R-squared, Adjusted R-squared, AIC/BIC
import statsmodels.formula.api as smf

# Load your data into a DataFrame
df = data
# Define the formula for OLS regression
formula = 'CO2 emissions' ~ independent_variable1 + independent_variable2'

# Fit the OLS model
model = smf.ols(formula=formula, data=df).fit()

# Print the summary of the regression
print(model.summary())

# 8. Extracting specific metrics
r_squared = model.rsquared
adj_r_squared = model.rsquared_adj
aic = model.aic
bic = model.bic

print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adj_r_squared}")
print(f"AIC: {aic}")
print(f"BIC: {bic}")

# 9. P-values and Confidence Intervals
# Print the p-values of the coefficients
print(model.pvalues)

# Print the 95% confidence intervals for the coefficients
print(model.conf_int())

# Multicollinearity (Variance Inflation Factor, VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Calculate VIF for each feature
X = df[['GDP per capita', 'industrial production']]
X['intercept'] = 1  # Add intercept for VIF calculation #Don't know which to choose

vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

# 10. Heteroskedasticity (Breusch-Pagan Test)
from statsmodels.stats.diagnostic import het_breuschpagan

# Perform Breusch-Pagan test
bp_test = het_breuschpagan(model.resid, model.model.exog)
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
print(dict(zip(labels, bp_test)))

## 3. Endogeneity (Durbin-Wu-Hausman Test, Instrumental Variables)
from linearmodels.iv import IV2SLS

## Define the instruments and endogenous variables
instruments = df[['instrument_variable']]
endog = df[['endogenous_variable']]
exog = df[['independent_variable1', 'independent_variable2']]
dependent = df['dependent_variable']

## Fit the IV model
iv_model = IV2SLS(dependent, exog, endog=endog, instruments=instruments).fit()
print(iv_model.summary)

## Perform Hausman test
from linearmodels.iv import hausman

hausman_test = hausman(iv_model, model)
print(hausman_test)


# 11. Policy Insights
•	Identify key drivers of emissions for targeted interventions.
•	Inform carbon taxes, urban planning, and industrial policies.
•	Analyze thresholds (e.g., energy intensity levels) for sustainable transitions.


