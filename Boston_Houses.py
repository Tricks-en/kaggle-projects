import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load kaggle dataset from cache into pandas
BH_data = pd.read_csv("C:/Users/trick/.cache/kagglehub/datasets/vikrishnan/boston-house-prices/versions/1/housing.csv", delimiter=",")
BH_data = BH_data.astype(float)
# check data for missing data, errors, etc.
print(BH_data.info())
print(BH_data.corr())

#setup variables for sklearn multi-regression (using only correlations of r > .4)
X = BH_data[['INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT']]
y = BH_data[['MEDV']]

Multi_reg = linear_model.LinearRegression()
Multi_reg.fit(X, y)

# linear regression over scatter plot for visualization
def lr_scatter(x, y, xlabel, ylabel):
  reg = linear_model.LinearRegression()
  reg.fit(x, y)
  y_pred = reg.predict(x)
  plt.scatter(x, y)
  plt.plot(x, y_pred, color='red')
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.show()

# visualize single var regressions over home value
lr_scatter(BH_data[['INDUS']], y, "proportion of non-retail business acres per town", "Home Value $/thousands")
lr_scatter(BH_data[['NOX']], y, "nitric oxides concentration (pp10m)", "Home Value $/thousands")
lr_scatter(BH_data[['RM']], y, "# of rooms/dwelling", "Home Value $/thousands")
lr_scatter(BH_data[['TAX']], y, "property-tax rate per $10k", "Home Value $/thousands")
lr_scatter(BH_data[['PTRATIO']], y, "pupil-teacher ratio by town", "Home Value $/thousands")
lr_scatter(BH_data[['LSTAT']], y, "% lower status of the population", "Home Value $/thousands")

X_test = [[12.5,	0.524,	5.889, 311,	15.2,	15.71]]
multi_y_pred = Multi_reg.predict(X_test)
# actual_MEDV = "21.7"

print(multi_y_pred)
