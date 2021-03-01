import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def num_conv_layers(params):
    ret = 0
    for key in params.keys():
        if key.startswith('Conv'):
            ret += 1
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
    for i in range(len(X) - 1, -1, -1):
        x_val = X[i]
        y_val = y[i]
        if x_val < x_lower or x_val > x_upper or y_val < y_lower or y_val > y_upper:
            X.pop(i)
            y.pop(i)
    return X, y


def simplify_clusters(X, y):
    data = set()
    for i in range(len(X)):
        data.add((X[i], y[i]))
    new_X, new_y = [], []
    while len(data) > 0:
        first = data.pop()
        temp = {first}  # only 1 point from temp will be added to new_x, new_y
        for point in data:
            if point[0] == first[0]:
                temp.add(point)
        # find median y coordinate from temp points
        y_values = [point[1] for point in temp]
        median = np.quantile(y_values, .5)
        for point in temp:
            if point[1] == median:
                new_X.append(point[0])
                new_y.append(point[1])
                break
    return new_X, new_y


dataset = 'CIFAR'  # options are [MNIST, CIFAR, SVHN]
results_path = './results_' + dataset + '/'
X = []
y = []
for filename in os.listdir(results_path):
    with open(results_path + filename) as json_file:
        params = json.load(json_file)
        cl = num_conv_layers(params)
        ad = params['accuracy difference']
        X.append(cl)
        y.append(ad)
X, y = remove_outliers(X, y)
X, y = simplify_clusters(X, y)
min_x = 0
max_x = max(X)
min_y = min(y)
max_y = max(y)
print("Number of nets: " + str(len(y)))
print('X: ' + str(X))
print('y: ' + str(y))
X, y = np.asarray(X), np.asarray(y)
X = X.reshape(-1, 1)

degree = 2
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

'''
# Use Hummingbird to convert the model to PyTorch
model = convert(model, 'pytorch')

# Run predictions on CPU
model.predict(X_poly)

# Run predictions on GPU
model.to('cuda')
model.predict(X_poly)

# Save the model
model.save('hb_model')

# Load the model back
model = hummingbird.ml.load('hb_model')
'''
plt.scatter(X, y)
X_plot = np.linspace(min_x, max_x, len(y)).reshape(-1, 1)
X_plot_poly = poly.fit_transform(X_plot)
y_pred = model.predict(X_plot_poly)
plt.plot(X_plot_poly[:, 1], y_pred, color='red', linewidth=3)
plt.grid()
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
title = dataset + ': Degree = {}, R2: {}'.format(degree, r2_score(y, y_pred))
plt.title(title)
plt.xlabel('Number of Conv Layers')
plt.ylabel('Accuracy Difference')
plt.savefig(dataset + '_regression.png', bbox_inches='tight')
plt.show()
