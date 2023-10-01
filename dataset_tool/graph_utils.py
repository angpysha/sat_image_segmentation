import matplotlib.pyplot as plt
import numpy as np

def createHistGraph(probalities, categories_map):
    """
    Creates histgram for propalities
    :param probalities: probabilites list
    :param categories_map: legend map to transform numbers to human-readable categories
    :return: None
    """
    fig, axs = plt.subplots(probalities.shape[1])

    for category in range(probalities.shape[1]):
        col = (np.random.random(), np.random.random(), np.random.random())
        axs[category].hist(probalities[:, category], color=col, range=(1/probalities.shape[1], 1.01), label=list(categories_map.keys())[category])
        axs[category].legend()