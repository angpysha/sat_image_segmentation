import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Activation
from tensorflow.keras.models import Sequential
from sklearn.neighbors import KDTree

class KNNLayer(Layer):
    def __init__(self, X_train, y_train, k=3, **kwargs):
        self.X_train = tf.Variable(X_train, dtype=tf.float32, trainable=False)
        self.y_train = tf.Variable(y_train, dtype=tf.float32, trainable=False)
        self.k = k
        super(KNNLayer, self).__init__(**kwargs)

    def compute_distances(self, X_test):
        X_train = K.expand_dims(self.X_train, axis=0)
        X_test = K.expand_dims(X_test, axis=1)
        distances = K.sqrt(K.sum(K.square(X_train - X_test), axis=-1))
        return distances

    def call(self, inputs):
        distances = self.compute_distances(inputs)
        _, knn_indices = tf.nn.top_k(-distances, k=self.k)  # Get indices of k smallest distances

        # Get labels of the nearest neighbors
        knn_labels = tf.gather(self.y_train, knn_indices)

        # Compute a one-hot-like matrix of shape (batch_size, k, num_classes)
        y_one_hot = tf.one_hot(knn_labels, depth=self.y_train.shape[1], dtype=tf.int32)

        # Sum along the neighbors axis, resulting in a tensor of shape (batch_size, num_classes)
        y_sum = tf.reduce_sum(y_one_hot, axis=1)

        # Convert to probabilities (normalize)
        y_prob = y_sum / K.sum(y_sum, axis=-1, keepdims=True)

        return y_prob


# Create a custom KNN-C layer
class KNNCLayer(Layer):

    def __init__(self, n_neighbors=5, metric='euclidean', target_data=None, **kwargs):
        super(KNNCLayer, self).__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.target_data = target_data

    # def build(self, input_shape):
    #     # Create the weights and biases for the layer
    #     self.weights = []
    #     self.biases = []

    def call(self, inputs):
        # Perform the forward pass of the layer
        distances = K.sqrt(K.sum(K.square(inputs - self.target_data), axis=1))

        # Find the nearest neighbors of the input data
        nearest_neighbors = K.nn.top_k(distances, k=self.n_neighbors)

        # Calculate the predicted values for the input data
        predicted_values = K.mean(self.target_data[nearest_neighbors], axis=1)

        return predicted_values


# class KNNClassifierLayer(Layer):
#     def __init__(self, X_train, y_train, k=3, **kwargs):
#         self.X_train = tf.Variable(X_train, dtype=tf.float32, trainable=False)
#         self.y_train = tf.Variable(y_train, dtype=tf.int32, trainable=False)
#         self.k = k
#         super(KNNClassifierLayer, self).__init__(**kwargs)
#
#     def compute_distances(self, X_test):
#         X_train = K.expand_dims(self.X_train, axis=0)
#         X_test = K.expand_dims(X_test, axis=1)
#         distances = K.sqrt(K.sum(K.square(X_train - X_test), axis=-1))
#         return distances
#
#     def call(self, inputs):
#         distances = self.compute_distances(inputs)
#         _, knn_indices = tf.nn.top_k(-distances, k=self.k)  # Get indices of k smallest distances
#
#         # Fetch the labels of the k-nearest training samples
#         knn_labels = tf.gather(self.y_train, knn_indices)
#
#         # Majority vote: For simplicity, we'll use the mean. This works for one-hot encoded labels.
#         y_pred = tf.reduce_mean(knn_labels, axis=1)
#
#         return y_pred

class KNNClassifierLayer(Layer):
    def __init__(self, X_train, y_train, k=3, num_classes=6, **kwargs):
        self.X_train = tf.Variable(X_train, dtype=tf.float32, trainable=False)
        self.y_train = tf.Variable(y_train, dtype=tf.int32, trainable=False)
        self.k = k
        self.num_classes = num_classes
        super(KNNClassifierLayer, self).__init__(**kwargs)

    def compute_distances(self, X_test):
        X_train = K.expand_dims(self.X_train, axis=0)
        X_test = K.expand_dims(X_test, axis=1)
        distances = K.sqrt(K.sum(K.square(X_train - X_test), axis=-1))
        return distances

    def call(self, inputs):
        distances = self.compute_distances(inputs)
        _, knn_indices = tf.nn.top_k(-distances, k=self.k)  # Get indices of k smallest distances

        # Fetch the labels of the k-nearest training samples
        knn_labels = tf.gather(self.y_train, knn_indices)

        # Compute histogram of labels
        y_hist = tf.reduce_sum(tf.one_hot(knn_labels, depth=self.num_classes), axis=1)

        # Normalize histogram to get a probability distribution across classes
        y_prob = y_hist / K.sum(y_hist, axis=-1, keepdims=True)

        return y_prob



class KNNClassifierLayer2(Layer):
    def __init__(self, X_train, y_train, k=3, num_classes=6, **kwargs):
        self.X_train_np = np.array(X_train)  # Convert to numpy for KDTree
        self.y_train_np = np.array(y_train)
        self.kdtree = KDTree(self.X_train_np)
        self.k = k
        self.num_classes = num_classes
        super(KNNClassifierLayer2, self).__init__(**kwargs)

    # def knn_using_kdtree(self, X_test_np):
    #     distances, indices = self.kdtree.query(X_test_np, k=self.k)
    #     knn_labels = self.y_train_np[indices].reshape(-1)  # Ensure the shape is (num_samples * k,)
    #
    #     # print(f"indices {indices}")
    #     # print(f"knn_labels {knn_labels}")
    #     # Compute histogram of labels
    #     y_hist = np.sum(np.eye(self.num_classes)[knn_labels], axis=1)
    #
    #     # Normalize histogram to get a probability distribution across classes
    #     y_prob = y_hist / np.sum(y_hist, axis=-1, keepdims=True)
    #     return y_prob.astype(np.float32)
    def get_config(self):
        config = super(KNNClassifierLayer2, self).get_config()
        config.update({
            'k': self.k,
            'num_classes': self.num_classes,
            'X_train_np': self.X_train_np.tolist(),  # Convert numpy array to list for serialization
            'y_train_np': self.y_train_np.tolist()  # Convert numpy array to list for serialization
        })
        return config

    def knn_using_kdtree(self, X_test):
        distances, knn_indices = self.kdtree.query(X_test, k=self.k)

        # Fetch the labels of the k-nearest training samples
        knn_labels = self.y_train_np[knn_indices]

        # Find the most frequent label among the k-nearest neighbors
        y_pred = np.array([np.bincount(k_labels).argmax() for k_labels in knn_labels])

        # Convert y_pred to one-hot encoded labels
        y_one_hot = np.eye(self.num_classes)[y_pred]

        return y_one_hot

    def call(self, inputs):
        y_prob = tf.py_function(func=self.knn_using_kdtree,
                                inp=[inputs],
                                Tout=tf.float32,
                                name='knn_using_kdtree')
        y_prob.set_shape([None, self.num_classes])
        return y_prob