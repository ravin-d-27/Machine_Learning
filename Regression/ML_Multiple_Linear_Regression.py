import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("D:/Machine_Learning_Datasets_and_more/My_ML_Codes/Part1/Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print("Created Independent and Dependent Variables")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("Encoded the categorical variables")

#Spliting the dataset into training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train)
print(y_train)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
print("The Training of model is completed")

print()
print("The Predicted Values are")
predicted = model.predict(X_test)
for i in range(len(predicted)):
    print(round(predicted[i],2),y_test[i])

