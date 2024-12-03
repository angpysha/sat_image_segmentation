from keras.layers import Layer
import numpy as np
from tensorflow.keras.backend import *
import tensorflow as tf

class NaiveBayesLayer(Layer):
    def __init__(self, num_features, num_of_categories, **kwargs):
        self.num_features = num_features
        self.num_of_categories = num_of_categories
        super(NaiveBayesLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Mean and variance for each feature for each class
        self.means = self.add_weight(name='means',
                                     shape=(self.num_of_categories, self.num_features),
                                     initializer='zeros',
                                     trainable=False)
        self.variances = self.add_weight(name='variances',
                                         shape=(self.num_of_categories, self.num_features),
                                         initializer='ones',
                                         trainable=False)
        super(NaiveBayesLayer, self).build(input_shape)

    def call(self, X):
        # Gaussian PDF for each class
        probs = [prod(self.gaussian_pdf(X, self.means[i], self.variances[i]), axis=1) for i in range(self.num_of_categories)]
        probs_tensor = stack(probs, axis=1)
        # Normalizing to get the posterior probabilities for each class
        probs_normalized = probs_tensor / sum(probs_tensor, axis=1, keepdims=True)
        return probs_normalized

    def gaussian_pdf(self, X, mean, variance):
        return exp(-square(X - mean) / (2 * variance)) / sqrt(2 * np.pi * variance)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_of_categories)


class MultiClassNaiveBayesLayer(Layer):
    def __init__(self, num_features, num_of_categories, **kwargs):
        self.num_features = num_features
        self.num_of_categories = num_of_categories
        super(MultiClassNaiveBayesLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Mean and variance for each feature for each class
        self.means = self.add_weight(name='means',
                                     shape=(self.num_of_categories, self.num_features),
                                     initializer='zeros',
                                     trainable=False)
        self.variances = self.add_weight(name='variances',
                                         shape=(self.num_of_categories, self.num_features),
                                         initializer='ones',
                                         trainable=False)
        super(MultiClassNaiveBayesLayer, self).build(input_shape)

    def call(self, X):
        # Gaussian PDF for each class
        prob_list = [prod(self.gaussian_pdf(X, self.means[i], self.variances[i]), axis=1) for i in range(self.num_of_categories)]
        probs = stack(prob_list, axis=-1)
        return probs

    # def gaussian_pdf(self, X, mean, variance):
    #     return K.exp(-K.square(X - mean) / (2 * variance)) / K.sqrt(2 * np.pi * variance)
    def gaussian_pdf(self, X, mean, variance):
        # Safe division and exponentiation
        epsilon = 1e-9  # Small constant to avoid division by zero
        variance_safe = tf.where(tf.equal(variance, 0), tf.ones_like(variance) * epsilon, variance)
        pdf = exp(-square(X - mean) / (2 * variance_safe)) / sqrt(2 * np.pi * variance_safe)

        # Handle NaN values by replacing them with 0
        pdf = tf.where(tf.math.is_nan(pdf), tf.zeros_like(pdf), pdf)
        return pdf

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_of_categories)