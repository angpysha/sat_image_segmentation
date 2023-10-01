from .binarytree_creator import *
from glob import glob
import os
import math
import json
import rasterio as rio
import numpy as np
# from .dataset5 import *

from .dataset6_helper import *

np.seterr(divide='ignore', invalid='ignore')

def floor_floats(obj):
    if isinstance(obj, float):
        return math.floor(obj)
    return obj

def getGroupedFiles(root_path):
    list = []

    subfolders = os.listdir(root_path)

    for subfolder in subfolders:
        search_expr = f"{root_path}/{subfolder}/*B?*.tiff"
        #search_expr = subfolder.join('*B?*.tiff')
        #print(f"search pattern: {search_expr}")
        files = glob(search_expr)
        list.append(files)
    return list

def load_configs(dataset_files):
    data = []
    for fldr in dataset_files:
        if len(fldr) > 0:
            fldr_first = fldr[0]
            file_dir = os.path.dirname(fldr_first)
            config_path = f"{file_dir}/dataset.config"
            with open(config_path, 'r') as config_file:
                config_json = config_file.read()
                parsed_json = json.loads(config_json, parse_float=floor_floats)
                helperItem = Dataset6HelperItem(fldr,parsed_json)
                data.append(helperItem)
                #print(config_json)
        else:
            print(f"Error read {fldr}")
    print(f"len {len(data)}")
    return data

def load_files(files_path):
    files = []
    for file_path in files_path:
        with rio.open(file_path) as rio_file:
            rio_data = rio_file.read(1)
            files.append(rio_data)
    return files

def crop_image(image_data, crop_info):
    width = int(math.floor(float(crop_info["width"])))
    height = int(math.floor(float(crop_info["height"])))
    top = int(math.floor(float(crop_info["top"])))
    left = int(math.floor(float(crop_info["left"])))

    new_data = []
    for img in image_data:
        img_cropped = img[top:top+height, left:left+width]
        new_data.append(img_cropped)
    return new_data

def load_image_parts(dataset_items: [Dataset6HelperItem]):
    categories_map = {}

    for dataset_item in dataset_items:
        files_data = load_files(dataset_item.files)
        categories = dataset_item.config['categories']
        items = dataset_item.config['items']
        #print(f"Item categories: {categories}")
        for category in categories:
            if category != '':
                filtered_items = [item for item in items if item["category"] == category]
                for filtered_item in filtered_items:
                    cropped_image = crop_image(files_data, filtered_item)
                    if categories_map.get(category) is None:
                        categories_map[category] = []
                    categories_map[category].append(cropped_image)

    return categories_map
                   #print(f"items for category {category}: {filtered_items}")

def vectorize(dataset):
    vectors_map = {}

    for item_key, item_value in dataset.items():
        item_nodes_list = []
        for item_value_node in item_value:
            item_file_list = []
            for item_value_file in item_value_node:
                vector = item_value_file.flatten()
                item_file_list.append(vector)
            item_file_array = np.asarray(item_file_list)
            # print(f"Shape: {item_file_array.shape}")
            item_nodes_list.append(item_file_array)

        if vectors_map.get(item_key) is None:
            vectors_map[item_key] = []
        vectors_map[item_key] = item_nodes_list
    return vectors_map

def stack_vectors(vectors_map):
    vectors_stacked = {}

    for item_key, item_value in vectors_map.items():
        if item_key == "field":
            item_key = "fields"
        vectors_stacked[item_key] = np.hstack(item_value)
    return vectors_stacked

def get_categories_map(categories):
    categories_map = {}
    for categorie, idx in enumerate(categories):
        categories_map[idx] = categorie
    return categories_map

def add_items(stacked_iamges):
    updated_dataset = {}

    for key, value in stacked_iamges.items():
        print(f"shape for {key} - {value.shape}")
        numOfRects = int(value.shape[1] / 64 / 64)
        itemsDiff = value.shape[1] - numOfRects * 64 * 64
        itemsToAdd = 64 * 64 - itemsDiff
        print(f"num of rects: {numOfRects}. To make full should add: {itemsToAdd}")
        if itemsToAdd > 0:
            values_sliced = value[:, :itemsToAdd]
            new_value = np.concatenate([value, values_sliced], axis=1)
            print(f"created new length with shape {new_value.shape}")
            updated_dataset[key] = new_value
        else:
            updated_dataset[key] = value
    return updated_dataset

def compact(updated_dataset, categories_map):
    y_data_list = []
    x_data_list = []

    for key, value in updated_dataset.items():
        category_id = categories_map[key]
        arr = np.full(value.shape[1], category_id)
        y_data_list.extend(arr)
        x_data_list.append(value)
    y_data = np.asarray(y_data_list)
    x_data_1 = np.hstack(x_data_list)
    x_data = np.moveaxis(x_data_1, 0, 1)
    return  x_data,y_data

def create_cubes(x_data, y_data):
    length = x_data.shape[0]
    y_data_cubes = y_data.reshape(length, 64, 64)
    x_data_cubes = x_data.reshape(length,64,64,12)
    return x_data_cubes, y_data_cubes

def get_dataset6(folder_path = None):
    # dataset_files = glob("/mnt/d/shared_folder/dataset_6/**/*B?*.tiff", recursive=True)
    # dataset_files.sort()

    dataset_files = getGroupedFiles("/mnt/d/shared_folder/dataset_6")
    dataset_items = load_configs(dataset_files)
    image_parts = load_image_parts(dataset_items)
    vectorized_images = vectorize(image_parts)
    vectorized_images_stacked = stack_vectors(vectorized_images)

    return vectorized_images_stacked
    #print(f"Files in folder: ${dataset_files}. {len(dataset_files)}")


def get_cubes_dataset(stacked_images):
    categories_map = get_categories_map(stacked_images.keys())
    updated_dataset = add_items(stacked_images)
    x_data, y_data = compact(stacked_images, categories_map)
    x_data_cubes, y_data_cubes = create_cubes(x_data, y_data)
    return x_data_cubes, y_data_cubes




