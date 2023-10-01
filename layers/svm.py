from keras.layers import Layer
import numpy as np
from keras import backend as K


class RBF_SVMLayer(Layer):
    def __init__(self, num_support_vectors, **kwargs):
        self.num_support_vectors = num_support_vectors
        super(RBF_SVMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize support vectors and associated weights (alphas)
        self.support_vectors = self.add_weight(name='support_vectors',
                                               shape=(self.num_support_vectors, input_shape[1]),
                                               initializer='uniform',
                                               trainable=True)

        self.alphas = self.add_weight(name='alphas',
                                      shape=(self.num_support_vectors,),
                                      initializer='uniform',
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(1,),
                                    initializer='zeros',
                                    trainable=True)

        # Gaussian RBF bandwidth
        self.sigma = self.add_weight(name='sigma',
                                     shape=(),
                                     initializer='ones',
                                     trainable=True)

        super(RBF_SVMLayer, self).build(input_shape)

    def call(self, X):
        # Compute RBF kernel between input and support vectors
        pairwise_diff = K.expand_dims(X, 1) - K.expand_dims(self.support_vectors, 0)
        pairwise_dist = K.sum(K.square(pairwise_diff), axis=-1)
        rbf_kernel = K.exp(-pairwise_dist / (2 * K.square(self.sigma)))

        # SVM decision function
        decision = K.sum(rbf_kernel * self.alphas, axis=1) + self.bias
        return K.sign(decision)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],)
