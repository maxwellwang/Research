import os
import json
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


def flatten_net(params):
    ret = []
    for layer in params['layers']:
        if layer['name'].startswith('Conv'):
            ret.append(layer['kernel size'][0])
        elif layer['name'].startswith('MaxPool'):
            ret.append(layer['kernel size'])
    return ret


def get_data(dataset):
    # n, c, h, w = 0, 0, 0, 0
    a_X = []
    a_y = []
    max_len = 0
    # datasets = ['MNIST', 'CIFAR', 'SVHN', 'FMNIST', 'GTSRB']
    # for dataset in datasets:
    results_path = './results_' + dataset + '/'
    for filename in os.listdir(results_path):
        '''
        if dataset == 'MNIST':
            n = 60000
            c = 1
            h = 28
            w = 28
        elif dataset == 'CIFAR':
            n = 50000
            c = 3
            h = 32
            w = 32
        elif dataset == 'SVHN':
            n = 73257
            c = 3
            h = 32
            w = 32
        elif dataset == 'FMNIST':
            n = 60000
            c = 1
            h = 28
            w = 28
        elif dataset == 'GTSRB':
            n = 39209
            c = 3
            h = 32
            w = 32
        '''
        with open(results_path + filename) as json_file:
            params = json.load(json_file)
            # numl = params['num layers']
            # nump = params['num params']
            # x = [c * h * w]
            # x.extend(flatten_net(params))
            x = flatten_net(params)
            max_len = max(max_len, len(x))
            ad = round(params['accuracy difference'], 2)
            a_X.append(x)
            a_y.append(ad)
    for x in a_X:
        while len(x) < max_len:
            x.append(0)
    return a_X, a_y


for dataset in ['MNIST', 'CIFAR', 'SVHN', 'FMNIST', 'GTSRB']:
    X, y = get_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
    regr = MLPRegressor(random_state=3, max_iter=10000).fit(X_train, y_train)
    print(dataset, 'Train R2:', round(regr.score(X_train, y_train), 2))
    print(dataset, 'Test R2:', round(regr.score(X_test, y_test), 2))
    # plt.scatter(regr.predict(X_test), y_test)
    # plt.xlabel('Predictions')
    # plt.ylabel('True Values')
    # plt.title(dataset)
    # plt.savefig(dataset + '.png')
    # plt.show()
