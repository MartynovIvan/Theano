import numpy
import theano
import theano.tensor as T
theano.config.warn.subtensor_merge_bug = False

coefficients = theano.tensor.vector("coefficients")
x = T.scalar("x")
max_coefficients_supported = 10000

# Generate the components of the polynomial
full_range=theano.tensor.arange(max_coefficients_supported)
components, updates = theano.scan(fn=lambda coeff, power, free_var:
                                   coeff * (free_var ** power),
                                outputs_info=None,
                                sequences=[coefficients, full_range],
                                non_sequences=x)

polynomial = components.sum()
calculate_polynomial = theano.function(inputs=[coefficients, x],
                                     outputs=polynomial)

test_coeff = numpy.asarray([1, 2, 3], dtype=numpy.float32)
print(calculate_polynomial(test_coeff, 4))

"""print(calculate_polynomial(test_coeff, 2))
""""test_coeff = numpy.asarray([1, 2, 3], dtype=numpy.float32)
""" 1 + 2 * 2^1 + 3 * 2^2  = 17 """


"""print(calculate_polynomial(test_coeff, 4))
""""test_coeff = numpy.asarray([1, 2, 3], dtype=numpy.float32)
""" 1 + 2 * 4^1 + 3 * 4^2  = 17 """
"""

coeff 1
power 0
free_var 4

coeff 2
power 1
free_var 4


coeff 3
power 2
free_var 4
                                   coeff * (free_var ** power),
