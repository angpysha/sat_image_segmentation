import cuml
from cuml.naive_bayes import *

from .base_model import *
from .iter_dataset import *
import copy

class GaussianNBClassiefier(BaseModel):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.predictions = []
        self.models = []

    def fit(self):
        self.models.clear()
        for item in self.dataset:
            knn = GaussianNB()
            knn.fit(item.x_train, item.y_train)
            knn.predict(item.x_test)
            self.models.append(copy.deepcopy(knn))
        if type(self.models[0]) is cuml.internals.base_helpers.BaseMetaClass:
            print("Removing class")
            self.models.remove(self.models[0])
        print(f"Models {self.models}")


    def predict_test(self):
        self.predictions = []

        print("Start prediction test data:")
        print(f"Models {self.models}")

        if type(self.models[0]) is cuml.internals.base_helpers.BaseMetaClass:
            print("Removing class")
            self.models.remove(self.models[0])

        for index, item in enumerate(self.dataset):
            print(f"Working with: idx {index} and shape {item.x_test.shape}...")
            x_test = item.x_test
            model = self.models[index]
            if isinstance(model, GaussianNB):
                prediction = model.predict(x_test)
                self.predictions.append(prediction)