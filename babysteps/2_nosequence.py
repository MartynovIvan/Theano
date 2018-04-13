import theano
import theano.tensor as T
import numpy as np

theano.config.warn.subtensor_merge_bug = False



X = T.matrix('X') # Minibatch of data
W = T.matrix('W') # Weights of the layer
b = T.vector('b') # Biases of the layer


def step(v, W, b):
    return T.dot(v, W) + b

output, updates = theano.scan(fn=step,
                              sequences=[X],
                              non_sequences=[W, b])
print(updates)

f = theano.function(inputs=[X, W, b],
                    outputs=output,
                    updates=updates)

X_value = np.arange(-3, 3).reshape(3, 2).astype(theano.config.floatX)
W_value = np.eye(2).astype(theano.config.floatX)
b_value = np.arange(2).astype(theano.config.floatX)
print(X_value)
print(W_value)
print(b_value)


print(f(X_value, W_value, b_value))

