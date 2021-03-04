import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import math


def flatten_net(params):
    ret = []
    for key in params.keys():
        if key.startswith('Conv'):
            conv_layer = params[key]
            for i in conv_layer["kernel size"]:
                ret.append(i)
            for i in conv_layer["stride"]:
                ret.append(i)
    return ret


def remove_outliers(X, y):
    # remove X and y outliers
    # remove coords that exceed 3Q+1.5IQR or 1Q-1.5IQR
    x_q1 = np.quantile(X, .25)
    x_q3 = np.quantile(X, .75)
    x_iqr = x_q3 - x_q1
    x_upper = x_q3 + 1.5 * x_iqr
    x_lower = x_q1 - 1.5 * x_iqr
    y_q1 = np.quantile(y, .25)
    y_q3 = np.quantile(y, .75)
    y_iqr = y_q3 - y_q1
    y_upper = y_q3 + 1.5 * y_iqr
    y_lower = y_q1 - 1.5 * y_iqr
    coordinates_list = []
    for i in range(len(X)):
        coordinates_list.append((X[i], y[i]))
    filtered_coordinates_list = []
    for x_val, y_val in coordinates_list:
        if x_lower <= x_val <= x_upper and y_lower <= y_val <= y_upper:
            filtered_coordinates_list.append((x_val, y_val))
    X.clear()
    y.clear()
    for x_val, y_val in filtered_coordinates_list:
        X.append(x_val)
        y.append(y_val)
    return X, y


def compress_vertical_lines(X, y):
    buckets = []
    for i in range(len(X)):
        if len(buckets) == 0:
            buckets.append([(X[i], y[i])])
        else:
            new_bucket = True
            for bucket in buckets:
                if bucket[0][0] == X[i]:
                    new_bucket = False
                    bucket.append((X[i], y[i]))
                    break
            if new_bucket:
                buckets.append([(X[i], y[i])])

    coordinates_list = []
    for bucket in buckets:
        y_values = []
        for coordinates in bucket:
            y_values.append(coordinates[1])
        median = np.quantile(y_values, .5)
        for coordinates in bucket:
            if coordinates[1] == median:
                coordinates_list.append(coordinates)
                break
    X.clear()
    y.clear()
    for x_val, y_val in coordinates_list:
        X.append(x_val)
        y.append(y_val)
    return X, y


def sort(X, y):
    new_X, new_y = [], []
    while len(X) > 0:
        min_val = min(X)
        index = X.index(min_val)
        X.remove(min_val)
        y_val = y.pop(index)
        new_X.append(min_val)
        new_y.append(y_val)
    return new_X, new_y


def get_data(dataset):
    results_path = './results_' + dataset + '/'
    X = []
    y = []
    max_len = 0
    for filename in os.listdir(results_path):
        with open(results_path + filename) as json_file:
            params = json.load(json_file)
            fn = flatten_net(params)
            max_len = max(len(fn), max_len)
            l = params['num layers']
            w = params['num params']
            ad = params['accuracy difference']
            X.append(fn)
            y.append(round(ad, 2))
    """
    X, y = remove_outliers(X, y)
    X, y = compress_vertical_lines(X, y)
    X, y = sort(X, y)
    """
    for fn in X:
        while len(fn) < max_len:
            fn.append(0)

    print(dataset)
    # print("Number of nets: " + str(len(y)))
    # print('X: ' + str(X))
    # print('y: ' + str(y))
    return X, y


def run(dataset, degree):
    X, y = get_data(dataset)

    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    """
    plt.scatter(X, y)
    y_pred = model.predict(X_poly)
    plt.plot(X_poly[:, 1], y_pred, color='red', linewidth=3)
    plt.plot([0, max_x], [0, 0], color='green', linewidth=3)
    plt.grid()
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    title = dataset
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Accuracy Difference')
    plt.savefig(dataset + '_regression.png', bbox_inches='tight')
    plt.show()
    """

    y_pred = model.predict(X_poly)
    print('R2: ' + str(round(r2_score(y, y_pred), 2)))


for dataset in ['MNIST', 'CIFAR', 'SVHN']:
    run(dataset, 2)
