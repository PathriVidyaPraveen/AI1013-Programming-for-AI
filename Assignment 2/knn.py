import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random
from collections import Counter

# The abalone dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
dataset = pd.read_csv(url, header=None)


# Data preprocessing and cleaning
column_names = ["Sex","Length","Diameter","Height","Whole weight","Shucked weight" , "Viscera weight","Shell weight","Rings"]
dataset.columns = column_names
dataset = dataset.drop('Sex',axis=1)

# Training and testing data
X = dataset.drop('Rings',axis=1).values
y = dataset['Rings'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=333)

# Implementing the k nearest meighbours algorithm
def custom_mode(lst):
    freq = Counter(lst)
    max_freq = max(freq.values())
    modes = [key for key, count in freq.items() if count == max_freq]
    return random.choice(modes)
def find_age(measures,k):
    # measures is a numpy array expected
    n_rows = X_train.shape[0]
    distances = []
    for idx in range(n_rows):
        measures_train = X_train[idx].tolist()
        measures_train = np.array(measures_train).astype(float)
        distance = (np.linalg.norm(measures_train-measures))**2
        distances.append((distance,idx))
    distances_sorted = sorted(distances)
    age_values_neighboring = []
    for index in range(k):
        index_neighbor = distances_sorted[index][1]
        age_values_neighboring.append(y_train[index_neighbor])
    return custom_mode(age_values_neighboring)

def find_mse(k):
    num_rows = X_test.shape[0]
    y_pred = []
    for idx in range(num_rows):
        measures = X_test[idx].tolist()
        measures = np.array(measures)
        y_pred.append(find_age(measures,k))
    mse = mean_squared_error(y_pred , y_test)
    return mse


# Tuning k to achieve optimal performance
k_values = [k for k in range(1,51)]
mse_values = []
for k in k_values:
    mse = find_mse(k)
    mse_values.append(mse)

k_values = np.array(k_values)
mse_values = np.array(mse_values)
optimal_mse = np.min(mse_values)
optimal_k = np.argmin(mse_values)+1

print(f"Optimal value of k : {optimal_k}")
print(f"Optimal value of MSE : {optimal_mse}")

plt.plot(k_values,mse_values,color='blue',label='MSE vs k')
plt.xlabel('K value')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs k')
plt.plot()


    









