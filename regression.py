import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def flatten_net(params):
    ret = []
    for layer in params['layers']:
        if len(layer.keys()) > 1:
            if "kernel size" in layer.keys():
                prod = 1
                if isinstance(layer["kernel size"], list):
                    for i in layer["kernel size"]:
                        prod *= i
                else:
                    prod *= layer["kernel size"]
                ret.append(prod)
            if "stride" in layer.keys():
                prod = 1
                if isinstance(layer["stride"], list):
                    for i in layer["stride"]:
                        prod *= i
                else:
                    prod *= layer["stride"]
                ret.append(prod)
    return ret


def get_data():
    N, C, H, W = 0, 0, 0, 0
    X = []
    y = []
    max_len = 0
    for dataset in ['MNIST', 'CIFAR', 'SVHN']:
        results_path = './results_' + dataset + '/'
        for filename in os.listdir(results_path):
            if dataset == 'MNIST':
                N = 60000
                C = 1
                H = 28
                W = 28
            elif dataset == 'CIFAR':
                N = 50000
                C = 3
                H = 32
                W = 32
            elif dataset == 'SVHN':
                N = 73257
                C = 3
                H = 32
                W = 32
            with open(results_path + filename) as json_file:
                params = json.load(json_file)
                x = [N * C * H * W]
                x.extend(flatten_net(params))
                max_len = max(len(x), max_len)
                ad = params['accuracy difference']
                X.append(x)
                y.append(ad)

    for x in X:
        while len(x) < max_len:
            x.append(0)

    return X, y


X, y = get_data()

X = PolynomialFeatures(2).fit_transform(X)
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
print('R2: ' + str(round(r2_score(y, y_pred), 2)))
