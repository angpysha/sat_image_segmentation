import numpy as np

from .binarytree_creator import *
from .dataset_loader import loadConfig
from .btree import *
from .dataset5 import *
import json
from .iter_dataset import *

class DatasetTreeAdapter:
    """
    Transforms default dataset to binary tree
    """
    def __init__(self, dataset: Dataset5):
        """
        Create tree dataset object
        :param dataset: default dataset object
        """
        config = loadConfig()
        categories_tree = config.get("config", "categories_tree")
        print(f"categories tree string {categories_tree}")

        categoreis_tree_np = json.loads(categories_tree)

        print(f"List array {categoreis_tree_np}.")

        tree = create_tree(categoreis_tree_np)
        self.categores_tree = tree
        self.dataset = dataset
        self.dataset_tree = self.__transfrom()

    def __flatten_list(self, nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(self.__flatten_list(item))
            else:
                flat_list.append(item)
        return flat_list

    def __transfrom(self):
        tree = self.__create_tree(self.categores_tree)
        tree = self.__flatten_list(tree)
        return tree

    def __create_tree(self, base_tree: BTree, offset: int = 0, offset_test: int = 0):
        lst = []
        category_value = self.dataset.categories_map[base_tree.left]
        x_train_tmp = self.dataset.x_train[offset:].copy()
        y_train_tmp = self.dataset.y_train[offset:].copy()
        x_test_tmp = self.dataset.x_test[offset_test:].copy()
        y_test_tmp = self.dataset.y_test[offset_test:].copy()

        y_train_tmp[y_train_tmp == category_value] = 1
        y_train_tmp[y_train_tmp != 1] = 2

        y_test_tmp[y_test_tmp == category_value] = 1
        y_test_tmp[y_test_tmp != 1] = 2

        print(f"Creating dataset: train len: {len(x_train_tmp)} test len: {len(x_test_tmp)}")
        data_set = IterDataset(x_train_tmp, y_train_tmp, x_test_tmp, y_test_tmp)

        lst.append(data_set)
        if type(base_tree.right) is BTree:
            new_offset = (np.where(self.dataset.y_train == category_value)[0][-1]+1)
            new_offset_test = (np.where(self.dataset.y_test == category_value)[0][-1]+1)
            next_list = self.__create_tree(base_tree.right, new_offset, new_offset_test)
            lst.append(next_list)

        return lst
