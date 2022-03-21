import pandas as pd
import numpy as np
import matplotlib as plt

dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Splitting into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# Now, gonna start feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Now importing the knn algorithm
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
model.fit(X_train, y_train)
print('The model is ready to predict')

print("Will the person buy the SUV? : ",model.predict(sc.transform([[30,87000]])))

# Predicting the test results

y_pred = model.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# To check the accuracy of this model
from sklearn.metrics import accuracy_score
print("Accuracy: ",accuracy_score(y_test,y_pred)*100,'%')