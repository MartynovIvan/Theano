import theano
import theano.tensor as T
import numpy as np

for start, end in zip(range(0, 1000, 128), range(128, 1000, 128)):
    print ("start=", start, " end=", end)
