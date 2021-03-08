import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import sys
from models.densenet import custom_densenet
from models.resnet import custom_resnet
from models.shufflenetv2 import custom_shufflenetv2
from torchsummary import summary
from utils import progress_bar
import os
import os.path
import json
import math
from models import *

# set constants
if len(sys.argv) < 2:
    print('Error: please enter dataset argument')
    exit(1)
DATASET = sys.argv[1]  # options are [MNIST, CIFAR, SVHN]
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 100
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
LEARNING_RATE = .1
MOMENTUM = .9
EPOCHS = 100  # stop training if num epochs exceeds this
TRAIN_ACCURACY_QUOTA = 99  # stop training if train accuracy exceeds this


def prep_loaders():
    train_set, test_set = None, None
    if DATASET == 'MNIST':
        train_set = torchvision.datasets.MNIST(root='.', train=True, download=False, transform=transform_mnist)
        test_set = torchvision.datasets.MNIST(root='.', train=False, download=False, transform=transform_mnist)
    elif DATASET == 'CIFAR':
        train_set = torchvision.datasets.CIFAR10(root='.', train=True, download=False, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='.', train=False, download=False, transform=transform_test)
    elif DATASET == 'SVHN':
        train_set = torchvision.datasets.SVHN(root='.', split='train', download=False, transform=transform_train)
        test_set = torchvision.datasets.SVHN(root='.', split='test', download=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train(net, device, train_loader, criterion, optimizer, epoch, model_name):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        assert not math.isnan(train_loss / (batch_idx + 1))

        progress_bar(batch_idx, len(train_loader),
                     (model_name + ' ' + DATASET + ': Train ' + str(epoch)) + ' | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return round(100. * correct / total, 3)


def test(net, device, test_loader, criterion, epoch, model_name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            assert not math.isnan(test_loss / (batch_idx + 1))

            progress_bar(batch_idx, len(test_loader),
                         (model_name + ' ' + DATASET + ':  Test ' + str(
                             epoch)) + ' | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return round(100. * correct / total, 3)


def handle_result(result_path, result, net, saved_model_path):
    if os.path.isfile(result_path):
        with open(result_path) as json_file:
            saved_result = json.load(json_file)
        if abs(result['accuracy difference']) < abs(saved_result['accuracy difference']):
            print('Improvement detected, overwriting result and saving new model...')
            with open(result_path, 'w') as outfile:
                json.dump(result, outfile)
            torch.save(net.module.state_dict(), saved_model_path)
        else:
            print('No improvement detected')
    else:
        print('Writing first result and saving first model...')
        with open(result_path, 'w') as outfile:
            json.dump(result, outfile)
        torch.save(net.state_dict(), saved_model_path)
    print('Result: ' + str(result))


def run(net, model_name):
    saved_model_path = './saved_models_' + DATASET + '/{}.pt'.format(model_name)
    result_path = './results_' + DATASET + '/{}.json'.format(model_name)

    # Prepare result model name and function params
    function_params = {}
    if net.function_params:
        function_params = net.function_params  # doing this first before net is moved to device
    result = {'model': model_name, 'function params': function_params}

    # Prepare data and network
    print('Running {} on {}...'.format(model_name, DATASET))
    num_epochs = 0
    if os.path.isfile(result_path):
        with open(result_path) as json_file:
            saved_result = json.load(json_file)
            num_epochs = saved_result['num epochs']
            if saved_result['train accuracy'] >= TRAIN_ACCURACY_QUOTA:
                print('Train accuracy exceeded {}%, no need to train'.format(TRAIN_ACCURACY_QUOTA))
                return
            if saved_result['num epochs'] >= EPOCHS:
                print('Num epochs exceeded {}, no need to train...'.format(EPOCHS))
                return
    train_loader, test_loader = prep_loaders()
    if os.path.isfile(saved_model_path):
        print('Loading saved model...')
        net.load_state_dict(torch.load(saved_model_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print('Error: CUDA not working')
        exit(1)
    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

    # Run network and organize output
    train_accuracy, test_accuracy = 0, 0
    for epoch in range(1 + num_epochs, EPOCHS + 1 + num_epochs):
        num_epochs += 1
        train_accuracy = train(net, device, train_loader, criterion, optimizer, epoch, model_name)
        test_accuracy = test(net, device, test_loader, criterion, epoch, model_name)
        if train_accuracy >= TRAIN_ACCURACY_QUOTA:
            print('Train accuracy exceeded {}%, stopping training...'.format(TRAIN_ACCURACY_QUOTA))
            break
        if num_epochs >= EPOCHS:
            print('Num epochs exceeded {}, stopping training...'.format(EPOCHS))
            break
    diff = abs(round(train_accuracy - test_accuracy, 3))
    result.update({'train accuracy': train_accuracy, 'test accuracy': test_accuracy, 'accuracy difference': diff})

    # Record num epochs and network info after moving to device
    result.update({'num epochs': num_epochs})
    if not os.path.isfile(result_path):
        # first time running model, record architecture info
        net.eval()
        total_layers, total_params, layers_summary = 0, 0, {}
        if DATASET == 'MNIST':
            total_layers, total_params, layers_summary = summary(net, (1, 28, 28), device=device)
        elif DATASET == 'CIFAR':
            total_layers, total_params, layers_summary = summary(net, (3, 32, 32), device=device)
        elif DATASET == 'SVHN':
            total_layers, total_params, layers_summary = summary(net, (3, 32, 32), device=device)
        result.update({'num layers': total_layers, 'num weights': total_params})
        layers = []
        for key, value in layers_summary.items():
            layer = {}
            layer['name'] = key
            layer['input shape'] = value['input_shape']
            layer['output shape'] = value['output_shape']
            if 'kernel_size' in value:
                layer['kernel size'] = value['kernel_size']
            if 'stride' in value:
                layer['stride'] = value['stride']
            if 'padding' in value:
                layer['padding'] = value['padding']
            if 'trainable' in value:
                layer['trainable'] = value['trainable']
            if 'nb_params' in value:
                layer['num params'] = int(value['nb_params'])
            layers.append(layer)
        result['layers'] = layers

    # Update results and save model if there was improvement
    handle_result(result_path, result, net, saved_model_path)


# remember to change forward implementations too
'''
v   mnist   cifar   svhn
0   1       1       1
1   1       1       1
2   1       1       1
3   1       1       1
4   1       1       1
5   1       1       1
6   1       1       1
7   1       1       1
8   1       1       1
9   1       1       1
10  1       1       1
11  1       1       1
12  1       1       1
13  1       1       1
1M  1       1       1
2M  1       1       1
3M  1       1       1
4M  1       1       1
'''
version = '-4M'
# run(alexnet(DATASET), 'alexnet' + version)  # change conv in self.features
# run(densenet121(DATASET), 'densenet121' + version)  # change end condition for block_config loop
# run(densenet161(DATASET), 'densenet161' + version)
# run(densenet169(DATASET), 'densenet169' + version)
# run(densenet201(DATASET), 'densenet201' + version)
run(googlenet(DATASET), 'googlenet' + version)  # change inception blocks
# run(inception_v3(DATASET), 'inception_v3' + version)  # change inception layers
# run(mnasnet0_5(DATASET), 'mnasnet0_5' + version)  # change mnas stacks
# run(mnasnet0_75(DATASET), 'mnasnet0_75' + version)
# run(mnasnet1_0(DATASET), 'mnasnet1_0' + version)
# run(mnasnet1_3(DATASET), 'mnasnet1_3' + version)
# run(mobilenet_v2(DATASET), 'mobilenet_v2' + version)  # change inverted_residual_setting
# run(resnet18(DATASET), 'resnet18' + version)  # change layers 1-4
# run(resnet34(DATASET), 'resnet34' + version)
# run(resnet50(DATASET), 'resnet50' + version)
# run(resnet101(DATASET), 'resnet101' + version)
# run(resnet152(DATASET), 'resnet152' + version)
# run(resnext50_32x4d(DATASET), 'resnext50_32x4d' + version)
# run(resnext101_32x8d(DATASET), 'resnext101_32x8d' + version)
# run(wide_resnet50_2(DATASET), 'wide_resnet50_2' + version)
# run(wide_resnet101_2(DATASET), 'wide_resnet101_2' + version)
# run(shufflenet_v2_x0_5(DATASET), 'shufflenet_v2_x0_5' + version)  # change stage_names 2-4
# run(shufflenet_v2_x1_0(DATASET), 'shufflenet_v2_x1_0' + version)
# run(shufflenet_v2_x1_5(DATASET), 'shufflenet_v2_x1_5' + version)
# run(shufflenet_v2_x2_0(DATASET), 'shufflenet_v2_x2_0' + version)
# run(squeezenet1_0(DATASET), 'squeezenet1_0' + version)  # change Fire layers
# run(squeezenet1_1(DATASET), 'squeezenet1_1' + version)
run(vgg11(DATASET), 'vgg11' + version)  # change cfgs lists
run(vgg11_bn(DATASET), 'vgg11_bn' + version)
run(vgg13(DATASET), 'vgg13' + version)
run(vgg13_bn(DATASET), 'vgg13_bn' + version)
run(vgg16(DATASET), 'vgg16' + version)
run(vgg16_bn(DATASET), 'vgg16_bn' + version)
run(vgg19(DATASET), 'vgg19' + version)
run(vgg19_bn(DATASET), 'vgg19_bn' + version)

directories = []
for directory in directories:
    for filename in os.listdir(directory):
        with open(directory + filename) as json_file:
            params = json.load(json_file)
        if 'densenet' in filename:
            run(custom_densenet(DATASET, params), filename[0:filename.index('.')] + version)
        elif 'resnet' in filename:
            run(custom_resnet(DATASET, params), filename[0:filename.index('.')] + version)
        elif 'shufflenetv2' in filename:
            run(custom_shufflenetv2(DATASET, params), filename[0:filename.index('.')] + version)
