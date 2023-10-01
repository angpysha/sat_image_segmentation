from glob import glob

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from scipy.io import loadmat
from sklearn.metrics import classification_report, accuracy_score

import plotly.graph_objects as go

import configparser

import pathlib
import sys
import json
from itertools import groupby
from scipy.ndimage import median_filter

from sklearn.preprocessing import MinMaxScaler
# directory = pathlib.Path(__file__).parent.resolve()
#
# # setting path
# sys.path.append(directory.parent)

from .binarytree_creator import *

from .dataset5 import *

np.seterr(divide='ignore', invalid='ignore')

def getCurrentDir():
    """
    Returns parent paths
    :return:
    """
    return pathlib.Path(__file__).parent.resolve()

def getCategory(path):
    url = path.split("/")
    category = url[5]
    return category


def getParentImageNumber(path):
    url = path.split("/")
    category = url[6]
    return category


def getImagePartNumber(path):
    url = path.split("/")
    category = url[7]
    return category

def toGrouped(array):
    np_array = np.asarray(array)
    print(f"Shape of array is {np_array.shape}")
    grouped_items = []
    for i in np_array:
        tpl = (i, getCategory(i), getParentImageNumber(i), getImagePartNumber(i))
        grouped_items.append(tpl)

    return np.asarray(grouped_items)

def getLearnCategoriesPaths(categories_to_include, grouped_items):
    dataset_map = {}
    for category in categories_to_include:
        items = grouped_items[grouped_items[:, 1] == category]
        dataset_map[category] = items
    return dataset_map

def createCategoriesNumbersMap(categories_to_learn):
        numbers_map = {}
        for idx, category in enumerate(categories_to_learn):
            numbers_map[category] = idx + 1
        return numbers_map

def loadConfig():
    config = configparser.ConfigParser()
    currentDir = getCurrentDir()
    config_path = currentDir.parent.joinpath("dataset_config")
    print(f"COnfig path {config_path.absolute()}")
    config.read(config_path.absolute())
    return config

def prepareBeforeTraining(categories_to_learn):
    categories = {}
    for category_key, category_value in categories_to_learn.items():
        items = []
        image_part_items = groupby(category_value, key= lambda x: x[2])
        for key, group in image_part_items:
            grouped_parts = groupby(group, key=lambda x: x[3])
            for key1, group1 in grouped_parts:
                items.append(list(group1))
       # items = sorted(items)
        categories[category_key] = np.asarray(items)
    return categories

def loadImages(prepared_items):
    loaded_images = {}
    for category_key, category_value in prepared_items.items():
        ff = []
        for item in category_value:
            l = []

            if len(item) > 12:
                print(f"item {item[0][0]}")

            for file_path in item:
                with rio.open(file_path[0], 'r') as f:
                    img = f.read(1)
                    l.append(np.asarray(img))

            if len(l[0]) == len(l[1]) and len(l[0][0]) == len(l[1][0]):
                ff.append(np.asarray(l))
        loaded_images[category_key] = ff
    return loaded_images

def createVectors(loaded_images):
    cat_vec = {}
    for cat_key, cat_val in loaded_images.items():
        print(f"Processing {cat_key}")
        itms = [[] for _ in range(12)]
        for img_group in cat_val:
            #grp = [[] for _ in range(12)]
            #print(f"Size of img group {len(img_group)}")
            for grp_idx,img in enumerate(img_group):
                #print(f"Index of type {grp_idx}. Array size: {len(grp)}")
                if len(img.shape) == 2:
                    vec = img.flatten()
                    itms[grp_idx].extend(vec)
            # if cat_key == "field":
            #         print(f"Size of itwm {len(itms[0])}")
        print(f"Size of {cat_key} is {len(itms[1])}")
        cat_vec[cat_key] = np.asarray(itms,  dtype=int)
    return cat_vec

def createYDatDict(categories_map, vectors):
    y_dict = {}
    for val, num in categories_map.items():
        shape_length = vectors[val].shape[1]
        y_dict[val] = np.full(shape_length, num)
    return y_dict

def flattenXY(vectors, y_data_cat):
    x_dt = [[] for _ in range(12)]
    for item in vectors.values():
        for idx, item1 in enumerate(item):
            x_dt[idx].extend(item1)

    y_dt = []
    for item in y_data_cat.values():
        y_dt.extend(item)

    x_array = np.asarray(x_dt)
    x_array_changed = np.moveaxis(x_array, -1, 0)

    return (np.asarray(x_array_changed), np.asarray(y_dt))

def median(array, window_size):
    l_filtred = []
    for ll in array:
        l_filtred.append(median_filter(array, window_size))
    return l_filtred

import random

def split_train_data2(x, y, coef):
    cnt = x.shape[0]
    items_to_train = int(cnt * coef)
    items_to_test = cnt - items_to_train
    percentile = int((1 - coef) * 100.0)
    print(f"Train size: {items_to_train}. Test size: {items_to_test}")
    test_indexes = random.sample(range(cnt), items_to_test)

    x_train = x[np.logical_not(np.isin(np.arange(cnt), test_indexes))]
    y_train = y[np.logical_not(np.isin(np.arange(cnt), test_indexes))]
    x_test = x[np.isin(np.arange(cnt), test_indexes)]
    y_test = y[np.isin(np.arange(cnt), test_indexes)]

    return x_train, y_train, x_test, y_test

def load_dataset5(train_test_coeff = 0.8):
    from glob import glob

    S_sentinel_bands = glob("/mnt/d/shared_folder/dataset_5_fixed/**/*B?*.tiff", recursive=True)
    S_sentinel_bands.sort()
    S_sentinel_bands

    print("test")

    grouped_items = toGrouped(S_sentinel_bands)

    config = loadConfig()
    print(config)
    to_learch = config.get("config","categories")
    print(f"to_learch {to_learch}")

    categories_to_include = np.asarray(to_learch.split(","))

    categories_map = createCategoriesNumbersMap(categories_to_include)

    categories_to_learn = getLearnCategoriesPaths(categories_to_include, grouped_items)
    print(f"Categories to learn: {categories_to_learn}")
    prepared_images = prepareBeforeTraining(categories_to_learn)
    print(f"prepared images: {prepared_images}")

    loaded_images = loadImages(prepared_images)

    print(f"Imags: {loaded_images}")

    images_vectors = createVectors(loaded_images)

    y_data_cat = createYDatDict(categories_map, images_vectors)

    x_data, y_data = flattenXY(images_vectors, y_data_cat)

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # x_scaled = scaler.fit_transform(x_data)
    x_scaled = x_data/255

    x_train, y_train, x_test, y_test = split_train_data2(x_scaled, y_data, train_test_coeff)


    dataset = Dataset5(categories_to_include, categories_map, x_train, y_train, x_test, y_test)

    return dataset
    #print("Load tree config, convert to np array, and then to tree")

    #categories_tree = config.get("config", "categories_tree")
    #print(f"categories tree string {categories_tree}")

    #categoreis_tree_np = json.loads(categories_tree)

    #print(f"List array {categoreis_tree_np}.")

    #tree = create_tree(categoreis_tree_np)

    #print(f"Tree created {tree}")

    #return tree

