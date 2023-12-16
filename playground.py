import numpy as np
from sklearn import datasets


def sigmoid(z):
    val = 1 / (1 + np.exp(-z))
    return val

def z_score(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    data_norm = (data - mu) / sigma
    return data_norm

def load_data():
    np.set_printoptions(precision=8, suppress=True)

    # Load Dataset
    iris_dataset = datasets.load_iris()

    # Pull Data
    iris_data = iris_dataset.data
    iris_target = iris_dataset.target

    # Shuffle Data
    permutation_indices = np.random.permutation(len(iris_data))
    iris_data_shuffled = iris_data[permutation_indices]
    iris_target_shuffled = iris_target[permutation_indices]

    # Z-Score Features
    X_normalized = z_score(iris_data_shuffled)
    
    # Split Training and Testing Sets
    X_train, X_test = np.split(X_normalized, [int(0.8 * len(X_normalized))], axis=0)
    y_train, y_test = np.split(iris_target_shuffled, [int(0.8 * len(iris_target_shuffled))], axis=0)

    return X_train, X_test, y_train, y_test

def compute_cost(X, y, w, b, lambda_ = 1):
    m, n = X.shape

    # Cost
    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)

        cost_i = (-y[i] * np.log(f_wb_i)) - ( (1 - y[i]) * np.log(1 - f_wb_i))
        cost += cost_i
    cost /= m

    # Regularization Term
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j] ** 2
    reg_cost *= (lambda_ / (2 * m))

    return cost + reg_cost

def compute_gradient(X, y, w, b, lambda_ = 1):
    m, n = X.shape
    dj_db = 0.0
    dj_dw = np.zeros(n)

    for i in range(m):
        # Calculate z, sigmoid, then calculate error
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)
        err_i = f_wb_i - y[i]

        dj_db += err_i
        for j in range(n):
            # Add in error per feature
            dj_dw[j] += err_i * X[i, j]

    dj_dw /= m
    dj_db /= m

    # Loop Adding Regularization Term
    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]

    return dj_dw, dj_db




def gradient_descent(X, y, alpha, num_of_iterations):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    initial_cost = compute_cost(X, y, w, b)
    print('Initial Cost:', initial_cost)
    

    for _ in range(num_of_iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        # Update weights agree
        w -= alpha * dj_dw
        # Update bias
        b -= alpha * dj_db




    final_cost = compute_cost(X, y, w, b)
    print('Final Cost:', final_cost)

X_train, X_test, y_train, y_test = load_data()

