import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Placement_Data_Full_class.csv")
#print(dataset.head())

# Data Preprocessing

dataset.replace({'gender':{'M':1,'F':0},
'ssc_b':{'Central':1,'Others':0},
'hsc_b':{'Central':1,'Others':0},
'hsc_s':{'Science':2,'Commerce':1,'Arts':0},
'degree_t':{'Comm&Mgmt':1,'Sci&Tech':2,'Others':0},
'workex':{'No':0,'Yes':1},
'specialisation':{'Mkt&Fin':0,'Mkt&HR':1},
'status':{'Placed':1,'Not Placed':0},
},inplace=True)

print(dataset.head())
print('\n\n')
# Selecting Dependent and Independent Variables

X = dataset.iloc[:,1:13].values
y = dataset.iloc[:,-2].values
y = y.reshape(len(y),1)

# Spliting the dataset into training and testing dataset

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Implementing ANN using TensorFLow

import tensorflow as tf

ann = tf.keras.Sequential()
ann.add(tf.keras.layers.Dense(units=10, activation='relu')) # Input layer
ann.add(tf.keras.layers.Dense(units=10, activation='relu')) # Hidden Layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # Output Layer

ann.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy']) # Compiling the ANN
ann.fit(X_train,y_train,batch_size=32,epochs=100) # Training the ANN

print("\nArtificial Neural Networks is Ready to Predict\n")

y_pred = ann.predict(X_test) # Predicting results
y_pred2 = []
answers = []
for i in y_pred:
    for j in i:
        if j>0.5:
            y_pred2.append(1)
        else:
            y_pred2.append(0)
print(y_pred2)

for i in range(len(y_test)):
    print(y_pred2[i],y_test[i],y_pred2[i]==y_test[i])

    



