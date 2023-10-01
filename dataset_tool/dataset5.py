
class Dataset5:
    """
    Model for current version of dataset
    """
    def __init__(self, categories_to_learn, categories_map,
                 x_train, y_train, x_test, y_test, images_cubes = None):
        self.categories_to_learn = categories_to_learn
        self.categories_map = categories_map
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.images_cubes = images_cubes
