# Code modified from    https://data36.com/linear-regression-in-python-numpy-polyfit/
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

df = pd.read_csv("../test/clean_house.csv")
x = df.sqft_living
y = df.price

# Plot points
plt.scatter(x, y)
plt.show()

# Apply linear regression model (polynomial of degree 1) to data and show theta1 and theta0
degree = 1
model = Polynomial.fit(x, y, degree).convert()

# Code modified from https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
X = df[['sqft_living', 'bedrooms', 'floors', 'view', 'yr_built']]
Y = df['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
mlr = LinearRegression()
mlr.fit(X_train, Y_train)
coeff_df = pd.DataFrame(mlr.coef_, mlr.feature_names_in_, columns=['Coefficient'])
print("Multivariate:")
print(coeff_df, '\n')
Y_pred = mlr.predict(X_test)
compare = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
print(compare, '\n')

print('Mean absolute error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean squared error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root mean squared error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print(f"R^2 Value: {mlr.score(X, Y)} \n")

if degree == 1:
    print("Theta0 and Theta1", model.coef)
else:
    print("Thetas . . .", model.coef)

# Show the r2 score (goodness of fit)
print("Goodness of fit (r2 score):", r2_score(y, polyval(x, model.coef)))

# Plot the points now with the regression line
x_lin_reg = list(range(0, df['sqft_living'].max()+1))
y_lin_reg = polyval(x_lin_reg, model.coef)
plt.scatter(x, y)
plt.plot(x_lin_reg, y_lin_reg, c='r')
plt.show(block=True)
