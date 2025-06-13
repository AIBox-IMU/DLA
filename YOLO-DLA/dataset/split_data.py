import os
import random
import shutil
from itertools import product
from pathlib import Path

random.seed(0)
import numpy as np

val_size = 0.1
test_size = 0.2
postfix = 'jpg'
imgpath = Path(r'C:\data\dataset\images')
txtpath = Path(r'C:\data\dataset\labels')

for p, p1 in list(product(['labels', 'images'], ['train', 'val', 'test'])):
    os.makedirs(Path(r'C:\data\dataset-2000') / p / p1, exist_ok=True)

listdir = np.array([i for i in os.listdir(txtpath) if 'txt' in i])
random.shuffle(listdir)
train, val, test = listdir[:int(len(listdir) * (1 - val_size - test_size))], listdir[int(len(listdir) * (
        1 - val_size - test_size)):int(len(listdir) * (1 - test_size))], listdir[
                                                                         int(len(listdir) * (1 - test_size)):]
print(f'train set size:{len(train)} val set size:{len(val)} test set size:{len(test)}')

for i in train:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), 'images/train/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), 'labels/train/{}'.format(i))

for i in val:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), 'images/val/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), 'labels/val/{}'.format(i))

for i in test:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), 'images/test/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), 'labels/test/{}'.format(i))
