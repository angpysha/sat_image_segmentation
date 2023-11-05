import dataset_tool
import tensorflow as tf
import layers.naive_bayes
from tensorflow.keras.layers import *
import dataset_tool.dataset6_generator
from sklearn.model_selection import train_test_split
import numpy as np
import dataset_tool.btree
from sklearn.metrics import accuracy_score
from glob import glob
import rasterio as rio


class Dataset6BinaryTree:
    def __init__(self, categories):
        self.dataset = None
        self.categories_map = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.models = None
        self.supported_categories = None
        self.supported_categories_map = None
        self.use_excluded_array = False
        self.categories = categories
        self.current_tree = dataset_tool.binarytree_creator.create_tree(categories)

    def create_model(self):
        """
        Creates binary model keras instance
        :return:
        """
        model = tf.keras.models.Sequential([
            layers.naive_bayes.MultiClassNaiveBayesLayer(num_features=12, num_of_categories=2, input_shape=(12,)),
            Activation("softmax")
        ])

        # Assuming you've trained or set the weights, you can then save the model:
        # save_model(model, 'naive_bayes_model.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])
        #model.summary()
        return model

    def initialize(self):
        """
        Initialize current model. This function do these actions:

        - Loads dataset
        - Creates categories map for this dataset
        - Loads supported categories
        :return:
        """
        self.dataset = dataset_tool.dataset6_generator.get_dataset6()
        self.categories_map = {idx: categorie + 1 for categorie, idx in enumerate(self.dataset.keys())}

        y_data_list = []
        x_data_list = []

        for key, value in self.dataset.items():
            category_id = self.categories_map[key]
            arr = np.full(value.shape[1], category_id)
            y_data_list.extend(arr)
            x_data_list.append(value)
        y_data = np.asarray(y_data_list)
        x_data_1 = np.hstack(x_data_list)
        x_data = np.moveaxis(x_data_1, 0, 1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data, test_size=0.15, stratify=y_data)
        self.supported_categories = self.get_supported_categories()

    def get_supported_categories(self):
        """
        Converts string of categories like '[water,[forest,[fields,cities]]]' to list of string ['water', 'forest', 'fields', 'cities'].
        :return: list of supported categories in string format
        """
        supported_cagetories = []
        current_node = self.current_tree
        while True:
            supported_cagetories.append(current_node.left)
            if type(current_node.right) != dataset_tool.btree.BTree:
                supported_cagetories.append(current_node.right)
                break
            else:
                current_node = current_node.right
        return supported_cagetories

    def create_models(self):
        """
        Creates sequence of Keras binary models to classify items
        :return: Sequence of Keras models with preseted weights
        """
        self.__remap_dataset_withsupported()
        current_tree = self.current_tree
        iteration = 0
        models = []
        while True:
            print(f"Entering into iteration {iteration+1}")
            first_category_num = iteration + 1
            model = self.__create_model_for_iteration(iteration=iteration, first_category_num=first_category_num)
            models.append(model)
            if type(current_tree.right) != dataset_tool.btree.BTree:
                break
            else:
                current_tree = current_tree.right
            iteration = iteration + 1
        self.models = models

    def __get_binary_indexes_to_exclude(self, iteration):
        return []


    def __create_model_for_iteration(self, iteration = 0, first_category_num = 1):
        """
        Creates instance of Naive bayes classifier for current iteration(binary tree level)
        :param iteration: binary tree level
        :param first_category_num:
        :return:
        """
        model = self.create_model()
        print(f"Iteration {iteration + 1} creating new arrays")
        to_train_x = self.x_train_new.copy() if self.use_excluded_array else self.x_train.copy()
        to_train_y = self.y_train_new.copy() if self.use_excluded_array else self.y_train.copy()
        idxes_to_exclude = self.__get_items_to_exclude_train(iteration, to_train_y)
        to_train_x = np.delete(to_train_x, idxes_to_exclude, axis=0)
        to_train_y = np.delete(to_train_y, idxes_to_exclude, axis=0)
        to_train_y[to_train_y == first_category_num] = 0
        to_train_y[to_train_y != 0] = 1
        print(f"Iteration {iteration + 1} calcualte weights")
        means, variances = self.__get_mean_variances(to_train_x, to_train_y)
        model.layers[0].set_weights([means, variances])
        return model

    def __get_items_to_exclude_train(self, iteration, y_data):
        """
        Get items indexes to exclude from dataset. This is using for binary tree recognition method
        :param iteration: Binary tree level
        :param y_data: Y data from dataset
        :return: Indexes to exclude from this iteration for current sample
        """
        categories_range = list(range(1, iteration+1))
        idx_to_exclude = []
        for i in categories_range:
            idxes = np.where(y_data == i)[0]
            idx_to_exclude.extend(idxes)
        return idx_to_exclude

    def __get_mean_variances(self, x_data, y_data):
        """
        Gets mean and varaince values to naive bayes implemetation
        :param x_data:
        :param y_data:
        :return:
        """
        means = []
        variances = []

        num_classes = 2  # or however many you have

        for i in range(num_classes):
            means.append(np.mean(x_data[y_data == i], axis=0))
            variances.append(np.var(x_data[y_data == i], axis=0))

        means = np.array(means)
        variances = np.array(variances)
        return means, variances


    def __remap_dataset_withsupported(self):
        """
        Creates new dataset with only supported categories and remap integer categories values to support classifier usage.

        **Example**

        There is full dataset with categories [1,2,3,4,5,6] but we want to support only half of them eg. [1,3,4].
        This function will *create new cropped arrays, creates different class insteances for compatibility, and remap this
        values to [1,2,3]*.
        """
        if len(self.categories_map) != len(self.supported_categories):
            self.supported_categories_map = {category: idx + 1 for idx, category in enumerate(self.supported_categories)}
            self.support_categories_idx_original = [self.categories_map[key] for key in self.supported_categories]
            self.categories_to_exclude = list(set(self.categories_map.values()) - set(self.support_categories_idx_original))
            indexes_to_exclude_train, indexes_to_exclude_test = self.__get_indexes_to_exclude(self.categories_to_exclude)
            self.__create_excluded_arrays(indexes_to_exclude_train, indexes_to_exclude_test)
            for category_key, category_value in self.supported_categories_map.items():
                full_cat_idx = self.categories_map[category_key]
        else:
            self.supported_categories_map == self.categories_map

    def __get_indexes_to_exclude(self, categories_to_exculde: [int]):
        """
        Gets indexes to exclude for new formed dataset. This used internally to exclude not used for training category data in dataset.
        :param categories_to_exculde: List of integer values to exclude from original dataset.
        :return: tuple with lists to exclude for train and test sample in this way `idx_train, idx_test`
        """
        indexes_train = []
        indexes_test = []
        for cat in categories_to_exculde:
            indexes_train.extend(np.where(self.y_train == cat)[0])
            indexes_test.extend(np.where(self.y_test == cat)[0])
        return np.asarray(indexes_train), np.asarray(indexes_test)

    def __create_excluded_arrays(self, to_exclude_train: [int], to_exclude_test: [int]):
        """
        Creates new arrays with excluded categories data and sets new class properties.

        - x_train_new -- new x_train data
        - y_train_new -- new y_train data
        - x_test_new -- new x_test data
        - y_test_new -- new y_test data
        :param to_exclude_train: List of indexes to exclude. Can be gotten using self.__get_indexes_to_exclude
        :param to_exclude_test: List of indexes to exclude. Can be gotten using self.__get_indexes_to_exclude
        :return:
        """
        self.use_excluded_array = True
        self.x_train_new = np.delete(self.x_train, to_exclude_train, axis=0)
        self.y_train_new = np.delete(self.y_train, to_exclude_train, axis=0)
        self.x_test_new = np.delete(self.x_test, to_exclude_test, axis=0)
        self.y_test_new = np.delete(self.y_test, to_exclude_test, axis=0)

    def validate_with_test_data(self, iterations = -1):
        """
        Validate current dataset with splited test data, which not participated in train process
        :param iterations: number of recursion of binary tree. By default all tree will be checked
        :return: Array of probabilities for each category
        """
        num_iterations = len(self.models) if iterations == -1 else iterations
        categories_idxs = {}
        predictions_percents = []
        for iter in range(num_iterations):
            current_model = self.models[iter]
            to_test_x = self.x_test_new.copy() if self.use_excluded_array else self.x_test.copy()
            to_test_y = self.y_test_new.copy() if self.use_excluded_array else self.y_test.copy()
            to_test_y_verify_all = to_test_y.copy()
            full_shape_length = to_test_y.shape[0]
            indexes_to_remove = np.concatenate(list(categories_idxs.values())) if len(categories_idxs) > 0 else []
            to_test_x = np.delete(to_test_x, indexes_to_remove, axis=0)
            to_test_y = np.delete(to_test_y, indexes_to_remove, axis=0)
            to_test_y[to_test_y == iter+1] = 0
            to_test_y[to_test_y != 0] = 1
            prediction_result = current_model.predict(to_test_x, batch_size=2048)
            prediction_result_categorized = tf.argmax(prediction_result, axis=-1).numpy()
            acc = accuracy_score(to_test_y, prediction_result_categorized)
            #predictions_percents.append(prediction_result)
            print(f"Binary acccucary for step {iter +1} is {round(acc,4) *100}%.")
            print(f"Calcaluting accuracy for {iter + 2} categories")
            prediction_result_categorized = prediction_result_categorized + iter
            combined_y_vec = np.empty(full_shape_length, dtype=int)
            if (iter == 0):
                combined_y_vec = prediction_result_categorized
            else:
                for cat_idx_key, categories_idxs_value in categories_idxs.items():
                    combined_y_vec[categories_idxs_value] = cat_idx_key
                idx_diff = np.setdiff1d(np.arange(full_shape_length), indexes_to_remove)
                combined_y_vec[idx_diff] = prediction_result_categorized
            idx_to_add = np.where(combined_y_vec == iter)[0]
            print(f"Adding idxes for category {iter+1}: {idx_to_add}")
            categories_idxs[iter] = idx_to_add
            combined_y_vec_1 = combined_y_vec + 1
            to_test_y_verify_all[to_test_y_verify_all > iter + 2] = iter + 2
            acc_all = accuracy_score(combined_y_vec_1, to_test_y_verify_all)
            print(f"Total acccucary for step {iter +1} is {round(acc_all,4) *100}%.")
            mm = BinaryTreeVerificationModel(prediction_result, combined_y_vec_1, to_test_y_verify_all)
            predictions_percents.append(mm)
        return predictions_percents

    def validate_with_test_data2(self, iterations=-1):
        num_iterations = len(self.models) if iterations == -1 else iterations

        categories_idxs = {}

        for iter in range(num_iterations):
            to_test_x, to_test_y = self._prepare_test_data(iter, categories_idxs)

            prediction_result = self.models[iter].predict(to_test_x, batch_size=2048)
            prediction_result_categorized = tf.argmax(prediction_result, axis=-1).numpy() + iter

            combined_y_vec = self._combine_prediction_results(iter, categories_idxs, to_test_y,
                                                              prediction_result_categorized)

            # Evaluate and display
            self._print_accuracy(iter, to_test_y, prediction_result_categorized)

            idx_to_add = np.where(combined_y_vec == iter)[0]
            print(f"Adding idxes for category {iter + 1}: {idx_to_add}")
            categories_idxs[iter] = idx_to_add

    def _prepare_test_data(self, iter, categories_idxs):
        to_test_x = self.x_test_new.copy() if self.use_excluded_array else self.x_test.copy()
        to_test_y = self.y_test_new.copy() if self.use_excluded_array else self.y_test.copy()

        indexes_to_remove = np.concatenate(list(categories_idxs.values())) if categories_idxs else []
        to_test_x = np.delete(to_test_x, indexes_to_remove, axis=0)
        to_test_y = np.delete(to_test_y, indexes_to_remove, axis=0)

        to_test_y[to_test_y == iter + 1] = 0
        to_test_y[to_test_y != 0] = 1

        return to_test_x, to_test_y

    def _combine_prediction_results(self, iter, categories_idxs, to_test_y, prediction_result_categorized):
        combined_y_vec = np.empty(to_test_y.shape[0], dtype=int)

        if iter == 0:
            combined_y_vec = prediction_result_categorized
        else:
            for cat_idx_key, categories_idxs_value in categories_idxs.items():
                combined_y_vec[categories_idxs_value] = cat_idx_key
            idx_diff = np.setdiff1d(np.arange(to_test_y.shape[0]), categories_idxs_value)
            combined_y_vec[idx_diff] = prediction_result_categorized

        return combined_y_vec

    def _print_accuracy(self, iter, to_test_y, prediction_result_categorized):
        acc = accuracy_score(to_test_y, prediction_result_categorized)
        print(f"Binary accuracy for step {iter + 1} is {round(acc, 4) * 100}%.")
        print(f"Calculating accuracy for {iter + 2} categories")


    def predict_image(self, image, iterations = -1):
        if (len(image.shape) != 3):
            raise AssertionError("The input image is in incorrect format, please put the image with shape (-1, -1, 12)")

        image_w, image_h = image.shape[0], image.shape[1]
        image_reshaped = image.reshape(image_w*image_h, 12)

        predictions = {}
        categories_idxs = {}
        combined_vecs = []
        num_iterations = len(self.models) if iterations == -1 else iterations
        for iter in range(num_iterations):
            current_model = self.models[iter]
            to_test_x = image_reshaped.copy()
            # to_test_y = self.y_test_new.copy() if self.use_excluded_array else self.y_test.copy()
            # to_test_y_verify_all = to_test_y.copy()
            full_shape_length = image_reshaped.shape[0]
            indexes_to_remove = np.concatenate(list(categories_idxs.values())) if len(categories_idxs) > 0 else []
            to_test_x = np.delete(to_test_x, indexes_to_remove, axis=0)
            # # to_test_y = np.delete(to_test_y, indexes_to_remove, axis=0)
            # to_test_y[to_test_y == iter + 1] = 0
            # to_test_y[to_test_y != 0] = 1
            prediction_result = current_model.predict(to_test_x, batch_size=2048)
            prediction_result_categorized = tf.argmax(prediction_result, axis=-1).numpy()
            # acc = accuracy_score(to_test_y, prediction_result_categorized)
            # print(f"Binary acccucary for step {iter + 1} is {round(acc, 4) * 100}%.")
            # print(f"Calcaluting accuracy for {iter + 2} categories")
            prediction_result_categorized = prediction_result_categorized + iter
            combined_y_vec = np.empty(full_shape_length, dtype=int)
            if (iter == 0):
                combined_y_vec = prediction_result_categorized
            else:
                for cat_idx_key, categories_idxs_value in categories_idxs.items():
                    combined_y_vec[categories_idxs_value] = cat_idx_key
                idx_diff = np.setdiff1d(np.arange(full_shape_length), indexes_to_remove)
                combined_y_vec[idx_diff] = prediction_result_categorized
            idx_to_add = np.where(combined_y_vec == iter)[0]
            print(f"Adding idxes for category {iter + 1}: {idx_to_add}")
            categories_idxs[iter] = idx_to_add
            predictions[iter] = prediction_result
            combined_y_vec_1 = combined_y_vec + 1
            combined_vecs.append(combined_y_vec_1)
        return predictions, combined_vecs
            # to_test_y_verify_all[to_test_y_verify_all > iter + 2] = iter + 2
            # acc_all = accuracy_score(combined_y_vec_1, to_test_y_verify_all)
            # print(f"Total acccucary for step {iter + 1} is {round(acc_all, 4) * 100}%.")

    def load_image(self, path):
        # if (path.contains("*B?*.tiff")):
        #     path = path + "/*B?*.tiff"

        image_bands = glob(path, recursive=True)
        image_bands.sort()

        imgs_list = []

        for tst_img in image_bands:
            with rio.open(tst_img, 'r') as img_file:
                img_array = img_file.read(1)
                imgs_list.append(img_array)

        imgs_array = np.asarray(imgs_list)
        imgs_array = np.moveaxis(imgs_array, 0, 2)
        return imgs_array

class BinaryTreeVerificationModel:
    def __init__(self, binary_predictions, array_of_categories, y_test):
        self.binary_predictions = binary_predictions
        self.array_of_categories = array_of_categories
        self.y_test = y_test
