# -*- coding: utf-8 -*-
import numpy as np
import random
X=np.ones((5,3))
Z=np.random.randint(2,10,[5,3])
Y=np.random.randint(1,10,[2,3])
print(Z)
print('\n')
print(Y)
print('\n')
for i in range(2):
    distances = np.sum(np.abs(Z - Y[i,:]),axis=1)
    print(distances)
    print('\n')
