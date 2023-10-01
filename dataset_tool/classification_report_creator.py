from cuml import accuracy_score
from sklearn.metrics import classification_report


class ClassificationReportCreator:
    def __init__(self, predictions, dataset):
        self.predictions = predictions
        self.dataset = dataset


    def display_tree_predictions(self):
        for idx, item in enumerate(self.dataset):
            prediction = self.predictions[idx]
            print(f"Get accureacy for {idx} iteration")
            acc_score = accuracy_score(prediction, item.y_test)
            class_report = classification_report(prediction, item.y_test)
            print(f"Accureacy {acc_score}")
            print(class_report)