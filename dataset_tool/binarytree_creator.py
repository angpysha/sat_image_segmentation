from .btree import *
import numpy as np
## Input data should be like this [water, [forest, [field, city]]]

def create_tree(array):
    tree = BTree(left=array[0])
    if type(array[1]) is list:
        tree.right = create_tree(array[1])
    else:
        tree.right = array[1]
    return tree