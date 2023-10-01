import tensorflow as tf
import numpy as np
from .binarytree_creator import *
from glob import glob
import os
import math
import json
import rasterio as rio
from .dataset6_helper import *

np.seterr(divide='ignore', invalid='ignore')


def read_and_preprocess_image(image_file):
    # Using rasterio to load the image
    with rio.open(image_file.numpy().decode('utf-8')) as rio_file:
        rio_data = rio_file.read(1)
    return rio_data


def tf_read_and_preprocess_image(image_file):
    # Use tf.py_function to call the function using Tensor arguments
    return tf.py_function(read_and_preprocess_image, [image_file], tf.float32)


def get_tf_dataset_for_images(file_paths):
    # Create a dataset of file paths
    file_paths_dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    # Map to load images
    image_dataset = file_paths_dataset.map(tf_read_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    return image_dataset


def get_cubes_tf_dataset(stacked_images):
    # categories_map = get_categories_map(stacked_images.keys())
    # updated_dataset = add_items(stacked_images)

    # Load and preprocess image paths to create a tf.data.Dataset
    image_file_paths = [item for sublist in getGroupedFiles("/mnt/d/shared_folder/dataset_6") for item in sublist]
    image_dataset = get_tf_dataset_for_images(image_file_paths)

    # Load labels
    _, y_data = compact(stacked_images, categories_map)
    label_dataset = tf.data.Dataset.from_tensor_slices(y_data)

    # Zip the datasets together
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    # Apply batching, shuffling, repeating, and prefetching
    dataset = dataset.batch(32).shuffle(buffer_size=1000).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset