import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Part 1 : Computation of average search delays , rel;ative entropy distance per neuron and l1 distance per neuron

search_times_db = pd.read_csv("02_data_visual_neuroscience_searchtimes.csv")
firing_rates_db = pd.read_csv("02_data_visual_neuroscience_firingrates.csv")

# AVERAGE SEARCH DELAYS
search_times_db_new = search_times_db.loc[3:]
search_times_db_new = search_times_db_new.astype(float)
average_search_delays = []
for i in range(0,18,2):
    left_image = search_times_db_new.iloc[:,i]
    right_image = search_times_db_new.iloc[:,i+1]
    left_image = left_image.to_numpy()
    right_image = right_image.to_numpy()
    sum_pairs = left_image + right_image
    
    sum_pairs = sum_pairs - 656
    average = sum_pairs.mean()
    average_search_delays.append(average)

# print(average_search_delays)

# relative entropy distances and l1 distances
firing_rates_db_new = firing_rates_db.loc[3:]
firing_rates_db_new = firing_rates_db_new.astype(float)
relative_entropy_distances = []
L_1_distances = []
epsilon = 1e-8
for i in range(0,18,2):
    left_image = firing_rates_db_new.iloc[:,i]
    right_image = firing_rates_db_new.iloc[:,i+1]
    mask = ~(left_image.isna() | right_image.isna())  # Keep only rows where both are not NaN

    left_image = left_image[mask].to_numpy()
    right_image = right_image[mask].to_numpy()
    n = len(left_image)
    relative_entropy_distance = 0
    l1 = 0
    for j in range(n):
        left = left_image[j]+epsilon
        right = right_image[j]+epsilon
        term = (left)*(np.log(left/right))
        relative_entropy_distance += term-left+right
        l1 += np.abs(left-right)
    relative_entropy_distance /= n
    l1 = l1/n
    relative_entropy_distances.append(relative_entropy_distance)
    L_1_distances.append(l1)

print(relative_entropy_distances)
print(L_1_distances)


# 2. Fitting a straight line passing through the origin.
inverse_average_search_delays = [1/x for x in average_search_delays]
model1 = LinearRegression()
model2 = LinearRegression()
X1 = np.array(relative_entropy_distances).reshape(-1,1)
X2 = np.array(L_1_distances).reshape(-1,1)
y = inverse_average_search_delays
model1.fit(X1,y)
model2.fit(X2,y)
y1_pred = model1.predict(X1)
y2_pred = model2.predict(X2)

fig,axs = plt.subplots(1,2,figsize=(12,6))
axs1 = axs[0]
axs2 = axs[1]
axs1.scatter(X1, y, color='blue', label='Data points')  # Scatter plot of original data
axs1.plot(X1, y1_pred, color='red', label='Fitted line')  # Plot the fitted line
axs1.set_xlabel('X')
axs1.set_ylabel('y')
axs1.set_title('Linear Regression for Entropy')

axs2.scatter(X2, y, color='blue', label='Data points')  # Scatter plot of original data
axs2.plot(X2, y2_pred, color='red', label='Fitted line')  # Plot the fitted line
axs2.set_xlabel('X')
axs2.set_ylabel('y')
axs2.set_title('Linear Regression for L1 distances')
plt.show()

mse1 = mean_squared_error(y,y1_pred)
mse2 = mean_squared_error(y,y2_pred)

print(f"Error for Entropy : {mse1} and Error for L1 : {mse2}")


# 3. Fitting a Gamma distribution to the search delays.

