import json


def generate_resnet():
    existing = [('BasicBlock', 2, 2, 2, 2, 1, 64),
                ('BasicBlock', 3, 4, 6, 3, 1, 64),
                ('Bottleneck', 3, 4, 6, 3, 1, 64),
                ('Bottleneck', 3, 4, 23, 3, 1, 64),
                ('Bottleneck', 3, 8, 36, 3, 1, 64),
                ('Bottleneck', 3, 4, 6, 3, 1, 64),
                ('Bottleneck', 3, 4, 23, 3, 32, 8),
                ('Bottleneck', 3, 4, 6, 3, 1, 64 * 2),
                ('Bottleneck', 3, 4, 23, 3, 1, 64 * 2)]
    combos = []
    for a in ['BasicBlock', 'Bottleneck']:
        for b in [2]:
            for c in [2, 8]:
                for d in [2]:
                    for e in [2]:
                        for f in [1, 32]:
                            for g in [8, 64, 64 * 2]:
                                if a == 'BasicBlock' and not (f == 1 and g == 64):  # BasicBlock only works with these
                                    continue
                                temp = (a, b, c, d, e, f, g)
                                combo = {"block": a, "layers": [b, c, d, e], "groups": f, "width per group": g}
                                if temp not in existing:
                                    combos.append(combo)
    i = 1
    param_path = './params/resnet/custom_resnet' + str(i) + '.json'
    for combo in combos:
        with open(param_path, 'w') as outfile:
            json.dump(combo, outfile)
            i += 1
            param_path = './params/resnet/custom_resnet' + str(i) + '.json'


def generate_densenet():
    existing = [(32, 6, 12, 24, 16, 64),
                (48, 6, 12, 36, 24, 96),
                (32, 6, 12, 32, 32, 64),
                (32, 6, 12, 48, 32, 64)]
    combos = []
    for a in [48]:
        for b in [6]:
            for c in [12]:
                for d in [24, 36, 32, 48]:
                    for e in [16, 24, 32]:
                        for f in [96]:
                            temp = (a, b, c, d, e, f)
                            combo = {"growth rate": a, "block config": (b, c, d, e), "num init features": f}
                            if temp not in existing:
                                combos.append(combo)
    i = 1
    param_path = './params/densenet/custom_densenet' + str(i) + '.json'
    for combo in combos:
        with open(param_path, 'w') as outfile:
            json.dump(combo, outfile)
            i += 1
            param_path = './params/densenet/custom_densenet' + str(i) + '.json'


def generate_shufflenetv2():
    existing = [(4, 8, 4, 24, 48, 96, 192, 1024),
                (4, 8, 4, 24, 116, 232, 464, 1024),
                (4, 8, 4, 24, 176, 352, 704, 1024),
                (4, 8, 4, 24, 244, 488, 976, 2048)]
    combos = []
    for a in [4]:
        for b in [8]:
            for c in [4]:
                for d in [24]:
                    for e in [48, 244]:
                        for f in [96, 488]:
                            for g in [192, 976]:
                                for h in [2048]:
                                    temp = (a, b, c, d, e, f, g, h)
                                    combo = {"stages repeats": [a, b, c], "stages out channels": [d, e, f, g, h]}
                                    if temp not in existing:
                                        combos.append(combo)
    i = 1
    param_path = './params/shufflenetv2/custom_shufflenetv2' + str(i) + '.json'
    for combo in combos:
        with open(param_path, 'w') as outfile:
            json.dump(combo, outfile)
        i += 1
        param_path = './params/shufflenetv2/custom_shufflenetv2' + str(i) + '.json'


generate_resnet()
# generate_densenet()
# generate_shufflenetv2()
