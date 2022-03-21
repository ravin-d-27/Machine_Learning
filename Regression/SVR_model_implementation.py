# Now we are gonna implement SVR (Support Vector Regression) in Python for the same dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
scy = StandardScaler()
X = scx.fit_transform(X)
y = scy.fit_transform(y)

# Now we are gonna train the SVR Model

from sklearn.svm import SVR
model = SVR(kernel = 'rbf') # rbf means radial basis function kernel, which is popularly used
model.fit(X,y.ravel()) # We have to fit only the Values which are feature scaled
print("The model is ready to predict")

x = scy.inverse_transform([model.predict(scx.transform([[6.5]]))])
print(x)

#Visualising the Graph of SVR
plt.scatter(scx.inverse_transform(X),scy.inverse_transform(y),color='red')
plt.plot(scx.inverse_transform(X),scy.inverse_transform([model.predict(scx.transform(scx.inverse_transform(X)))]).reshape(10,1))
plt.xlabel("Level")
plt.ylabel("Salaries")
plt.title("Salary Prediction Using SVR")
plt.show()
