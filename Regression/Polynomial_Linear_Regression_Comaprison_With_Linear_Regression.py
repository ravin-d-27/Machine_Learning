import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("D:/Machine_Learning_Datasets_and_more/My_ML_Codes/Part1/Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#First we are Creating the Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X,y)

#Now we are creating a polynomial linear regression model
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly,y)

#Visualising the results of Linear Regression Model
plt.scatter(X,y, color='red')
y_pred = linear.predict(X)
plt.plot(X,y_pred,color='blue')
plt.title("Linear Regression")
plt.xlabel("Job Position")
plt.ylabel("Salary")
plt.show()

#visualising the Polynomial Linear Regression
plt.scatter(X,y, color='red')
plt.plot(X,poly_model.predict(poly.fit_transform(X)),color='blue')
plt.title("Polynomial Linear Regression")
plt.xlabel("Job Position")
plt.ylabel("Salary")
plt.show()

# Visualising the polynomial Linear Regression with even more smooth curve:
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,poly_model.predict(poly.fit_transform(X_grid)),color='blue')
plt.title("Polynomial Linear Regression (With Smooth Curve)")
plt.xlabel("Job Position")
plt.ylabel("Salary")
plt.show()

# Predicting a random result using Linear Regression Model
print("Predicted Result using Linear Regression Model is:")
print(linear.predict([[6.5]]))

# Predicting a random result using Polynomial Linear Regression Model
pred = poly.fit_transform([[6.5]])
print("Predicted Result using Polynomial Linear Regression Model is:")
print(poly_model.predict(pred))