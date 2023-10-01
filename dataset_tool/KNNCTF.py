import tensorflow as tf

class TFKNNClassifier:

    def __init__(self, k=3, distance='euclidean'):
        self.k = k
        self.distance = distance.lower()
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        self.y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

    def calculate_distance(self, x1, x2):
        if self.distance == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x1, x2)), axis=1))
        elif self.distance == 'manhattan':
            return tf.reduce_sum(tf.abs(tf.subtract(x1, x2)), axis=1)
        else:
            raise ValueError("Invalid distance metric. Supported metrics: 'euclidean', 'manhattan'.")

    def predict(self, x_test):
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

        distances = self.calculate_distance(self.x_train, tf.expand_dims(x_test, 1))
        _, top_k_indices = tf.nn.top_k(tf.negative(distances), k=self.k)
        top_k_labels = tf.gather(self.y_train, top_k_indices)

        predictions = tf.reduce_mean(top_k_labels, axis=1)
        return predictions