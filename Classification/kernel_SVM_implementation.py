import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)

# Splitting into training and test dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# Feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Now implementing the model

from sklearn.svm import SVC
model = SVC(kernel='rbf',random_state=0)
model.fit(X_train,y_train.ravel())
print("The model is ready to predict")

print("Does the person bought the SUV? : ",model.predict(sc.transform([[30,87000]])))

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

predict_data = [[55,1,3,123,200,0,1,180,1,0.4,2,0,1]]
sc_pred_data = sc.fit_transform(predict_data)
predicted_value = model.predict(sc_pred_data)

if predicted_value[0] == 1:
    print("Yes, the patient has Heart Disease")
else:
    print("No, the patient doesn't have Heart Diesase")

