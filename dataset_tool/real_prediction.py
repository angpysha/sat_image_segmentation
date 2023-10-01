import glob
import rasterio as rio
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler

class RealPrediction():
    def __init__(self, folder_path, model):
        self.predictions_composite_proba = None
        self.predictions_composite = None
        self.predictions = []
        self.predictions_proba = []
        self.folder_path = folder_path
        self.model = model
        self.width = 0
        self.height = 0
        self.image = []
        self.__load()
        self.arr_st = []

    def __flattenTestImage(self, image):
        imgs = []
        for img in image:
            imgs.append(list(itertools.chain(*img)))
        return imgs

    def __load(self):
        test_image_bands = glob.glob(f"{self.folder_path}/*B?*.tiff", recursive=True)
        test_image_bands.sort()

        l2 = []
        for i in test_image_bands:
            with rio.open(i, 'r') as f:
                l2.append(f.read(1))

        self.height = len(l2[0])
        self.width = len(l2[0][0])

        arr_st = np.stack(l2)

        img = arr_st[[3,2,1], :,:]
        self.image = np.moveaxis(img, 0,-1)

        tst_img = self.__flattenTestImage(l2)
        tst_img_np = np.asarray(tst_img)
        tst_img_np = np.moveaxis(tst_img_np, -1, 0)

        scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaled = scaler.fit_transform(tst_img_np)
        self.x_scaled_base = x_scaled

    def predict(self):
        items_to_skip = []
        predictions = []
        x_scaled = self.x_scaled_base.copy()
        for idx, m in enumerate(self.model):
            prediction = m.predict(x_scaled)
            if idx + 1 != len(self.model):
                condition = prediction == 1
                condition_indexes = np.where(condition)[0]
                partial_prediction = prediction[condition].copy()
                x_scaled = np.delete(x_scaled, condition_indexes,axis=0)
                predictions.append((condition_indexes, partial_prediction, idx))
            else:
                predictions.append(([], prediction, idx))
        self.predictions = predictions

    def predict_proba(self):
        items_to_skip = []
        predictions = []
        x_scaled = self.x_scaled_base.copy()
        for idx, m in enumerate(self.model):
            prediction = m.predict_proba(x_scaled)
            # predictions.append(prediction)
            if idx + 1 != len(self.model):
                condition_indexes = self.predictions[idx][0]
                # x_scaled = np.delete(x_scaled, condition_indexes, axis=0)
                predictions.append((condition_indexes,prediction,idx))
                # condition_indexes = np.where(condition)[0]
                # partial_prediction = prediction[condition].copy()
                # x_scaled = np.delete(x_scaled, condition, axis=0)
                # predictions.append((condition_indexes, partial_prediction, idx))
            else:
                predictions.append(([], prediction, idx))
        self.predictions_proba = predictions


    # def flatten(self):
    #     array = np.empty(0)
    #     for index, item in reversed(list(enumerate(self.predictions))):
    #         new_array = np.empty(array.shape[0] + item[1].shape[0], dtype=float)
    #         print(f"Predcition: {item[2]}. Len of idxs: {len(item[0])}. New array shape: {new_array.shape}")
    #         new_items = item[1] + item[2]
    #         if len(item[0]) == 0:
    #             new_array[np.arange(item[1].shape[0])] = new_items
    #         else:
    #             new_array[item[0]] = item[1]
    #             new_array[~np.isin(np.arange(array.shape[0] + item[1].shape[0]), item[0])] = array
    #         array = new_array
    #     self.predictions_composite = array

    def flatten(self):
        array = np.empty(0)
        for index, item in reversed(list(enumerate(self.predictions))):
            new_array = np.zeros(array.shape[0] + item[1].shape[0], dtype=float)
            print(f"Predcition: {item[2]}. Len of idxs: {len(item[0])}. New array shape: {new_array.shape}")
            print(
                f"items before: {item[1]}: Min value: {item[1].min()} max value: {item[1].max()}. Items with 1: {np.where(item[1][item[1] == 1])[0].shape}.Items with 2: {np.where(item[1][item[1] == 2])[0].shape}")
            new_items = item[1].copy() + item[2]
            print(
                f"items after: {new_items}: Min value: {new_items.min()} max value: {new_items.max()}. Items with {1 + item[2]}: {np.where(new_items[new_items == 1 + item[2]])[0].shape}.Items with {2 + item[2]}: {np.where(new_items[new_items == 2 + item[2]])[0].shape}")
            # print(f"items before: {new_items}")
            if len(item[0]) == 0:
                new_array[np.arange(item[1].shape[0])] = new_items.copy()
            else:
                print(f"items idx: {item[0]}  with type {type(item[0])}")
                new_array[~np.isin(np.arange(array.shape[0] + item[1].shape[0]), item[0])] = array.copy()
                new_array[np.isin(np.arange(array.shape[0] + item[1].shape[0]), item[0])] = new_items.copy()
            array = new_array
            print(f"max: {array.min()}")
        self.predictions_composite = array
        
    
    def flatten_proba(self):
        array = np.empty(0)
        for index, item in reversed(list(enumerate(self.predictions_proba))):
            new_array = np.empty(array.shape[0] + item[1].shape[0], dtype=float)
            print(f"Predcition: {item[2]}. Len of idxs: {len(item[0])}. New array shape: {new_array.shape}")
            new_items = item[1] + item[2]
            if len(item[0]) == 0:
                new_array[np.arange(item[1].shape[0])] = new_items
            else:
                new_array[item[0]] = item[1]
                new_array[~np.isin(np.arange(array.shape[0] + item[1].shape[0]), item[0])] = array
            array = new_array
        self.predictions_composite_proba = array

    def to2d(self):
        return self.predictions_composite.reshape((self.height, self.width))

    # def create_image_map(self):
    #     if not self.predictions:
    #         raise Exception("Call predict first")
    #
    #     for index, item in reversed(list(enumerate(self.predictions))):
