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

class Dataset6Multiclass:
    def __init__(self, dataset6):
        self.dataset6 = dataset6

