import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

# Data Preprocessing
transactions  = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(20)])

# Training the Apriori Model
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence=0.2,min_lift=3,min_length=2,max_length=2)
print("Done with the training of apriori")
results = list(rules)

# Visualising the results

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the results non sorted
print(resultsinDataFrame)
print()
print("Sorted by Lift")
## Displaying the results sorted by descending lifts
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))
