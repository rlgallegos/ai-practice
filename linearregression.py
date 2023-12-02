import math, copy
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read data into lists
with open('data.txt', 'r') as f:
    data_lines = f.read().split('\n')
    
    x_data = []
    y_data = []

    # x data -> per 1k square feet (not truncated)
    # y data -> per 100k dollars

    for line in data_lines:
        values = line.split(',')
        x_data.append(int(values[0]) / 1000)
        y_data.append(int(values[2]) // 1000)

    x_data = np.array(x_data)
    print(x_data)
    y_data = np.array(y_data)
    print(y_data)

    # Scaling idea to help with numerical instability, currently unimplemented
    # scaler = StandardScaler()
    # # Fit the scaler to the training data
    # scaler.fit(x_data)

# Initialize variables
num_of_iterations = 10000
w_init, b_init = 1000, 100
alpha = 0.01


# Function to figure out weight adjustments for the data set with the current weights (inner fn)
def calculate_gradient(x, y, w, b, m):
    dj_db = 0
    dj_dw = 0

    for i in range(m):
        # Current prediciton with next training sample (x[i])
        f_wb = w * x[i] + b

        # Weights for the current prediction
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]

        # Simulatenously update so w's change doesn't impact b
        dj_dw += dj_dw_i / m
        dj_db += dj_db_i / m

    # divide by m to make more manageable
    # dj_dw /= m
    # dj_db /= m
    
    return dj_dw, dj_db


def compute_cost(x, y, w, b, m):
    cost = 0

    # In a loop over each data set
    for i in range(m):
        # calculate the predicted output (y-hat) with these weights
        f_wb = w * x[i] + b
        # Squared Error Cost function
        cost += (f_wb - y[i]) ** 2
    
    # Modify to complete formula
    total_cost = round(1 / (2 * m) * cost, 2)

    return total_cost


# Overall Function 
def gradient_descent(x_data, y_data, w_in, b_in, alpha, num_of_iterations):
    m = x_data.shape[0]

    w, b = w_in, b_in
    intial_cost = compute_cost(x_data, y_data, w, b, m)
    print('Initial Cost: ', intial_cost)

    for i in range(num_of_iterations):

        dj_dw, dj_db = calculate_gradient(x_data, y_data, w, b, m)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
    ending_cost = compute_cost(x_data, y_data, w, b, m)
    print('Ending Cost: ', ending_cost)

    return w, b

w_final, b_final = gradient_descent(x_data, y_data, w_init, b_init, alpha, num_of_iterations)

