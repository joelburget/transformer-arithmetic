import numpy as np
import random
from hyperparams import frac_train, p, seed


def gen_train_test(frac_train, num, seed=0):
    # Generate train and test split
    pairs = [(i, j, num) for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train * len(pairs))
    return pairs[:div], pairs[div:]


# Creates an array of Boolean indices according to whether each data point is in
# train or test
# Used to index into the big batch of all possible data


def make_predicate_arrays(train, test):
    is_train = []
    is_test = []
    for x in range(p):
        for y in range(p):
            if (x, y, 113) in train:
                is_train.append(True)
                is_test.append(False)
            else:
                is_train.append(False)
                is_test.append(True)
    return np.array(is_train), np.array(is_test)


train, test = gen_train_test(frac_train, p, seed)

div_train = [(x, y, z) for x, y, z in train if y != 0]
div_test = [(x, y, z) for x, y, z in test if y != 0]

is_train, is_test = make_predicate_arrays(train, test)
is_div_train, is_div_test = make_predicate_arrays(div_train, div_test)
