import abc

from .iter_dataset import *
from abc import ABC


class BaseModel:
    """
    Base class for predictor. All classifiers have to iherit form this calls
    """
    def __init__(self, dataset: [IterDataset]):
        self.dataset = dataset

    @abc.abstractmethod
    def fit(self):
        pass
