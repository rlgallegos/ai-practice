import numpy as np
import copy

# Format Data ---
# Gradient Descent
# Compute Cost
# Compute Gradient

def format_data(path):
    with open(path, 'r') as f:
        text = f.read()
        example_rows = text.split('\n')
        examples = [row.split(',') for row in example_rows]
        y_col = [example[3] for example in examples]
        x_mat = [[example[0], example[1], example[2]] for example in examples]

        print('y examples length (i)', len(y_col))
        print('X Matrix examples length (i)', len(x_mat))
        print('quantiy of features (j)', len(x_mat[0]))

        y_train = np.array(y_col)
        X_train = np.array(x_mat)

        return X_train, y_train


def calculate_cost(X, y, w, b):
    m, n = X.shape

    cost = 0
    for i in range(n):
        f_wb_i += np.dot(X[i], w) + b
        cost += (f_wb_i - y[i]) ** 2

    return cost

def compute_gradient(X, y, w, b):
    m, n = X.shape

    dj_db = 0
    dj_dw = 0

    for i in range(m):
        err = np.dot(X[i], w)
        # adjust b
        dj_db += err
        # loop thru the different features
        for j in range(n):
            dj_dw += err * X[i, j]
        
        dj_db /= m
        dj_dw /= m

    return dj_dw, dj_db


def gradient_descent(w_in, b_in, X, y, alpha, num_iterations):
    print(num_iterations)

    # Don't alter global
    w = copy.deepcopy(w_in)
    b = b_in

    init_cost = calculate_cost(X, y, w, b)
    print('Initial Cost: ', init_cost)

    for _ in range(num_iterations):

        # Get gradient
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        
        # Update weights
        w -= alpha * dj_dw
        b -= alpha * dj_db

    final_cost = calculate_cost(X, y, w, b)
    print('Final Cost: ', final_cost)








X_train, y_train = format_data('houses.txt')

w_init = [0 for _ in range(X_train.shape[1])]
b_init = 0
num_iterations = 1000
alpha = .001


value = gradient_descent(w_init, b_init, X_train, y_train, alpha, num_iterations)