from scipy.optimize import curve_fit
import numpy as np


def sigmoid(x, a, x0, k, b):
    """Sigmoid function to calculate the confidence score for any given cosine distance value

    Definition:
    y = a / (1+e(^-k*(x-x0))) + b

    The function is forced to be in between a range of [0, 1]

    :param x: input parameter (in our case the cosine distance)
    :param a:
    :param x0:
    :param k:
    :param b:
    :return: the confidence score value
    """

    y = a / (1 + np.exp(-k * (x - x0))) + b
    return y


class ConfidenceScoreCalculator:
    def __init__(self, bins: int = 2000, p0k: int = 18):
        self.bins = bins
        self.p0k = p0k

    def __call__(self, cosine_distances, labels, threshold) -> list:
        """The algorithm to derive the parameters for the sigmoid function which calculated the confidence score

        :param cosine_distances: Cosine distances for a "fold" or "dataset" of image pairs
        :param labels: Corresponding labels "True" -> genuine and "False" -> imposter image pairs
        :param threshold: The defined threshold for the given network for binary classification
        :return: Four parameters in a list: As defined in def sigmoid(..): a, x0, k, b
        """

        # Select distances for genuine and imposter pairs
        dists_pos = cosine_distances[np.where(labels)]
        dists_neg = cosine_distances[np.where(np.logical_not(labels))]

        # Generate histogram and count the number of distance for each bin
        counts_pos, _ = np.histogram(dists_pos, bins=self.bins, range=(0.0, 2.0), density=True)
        counts_neg, _ = np.histogram(dists_neg, bins=self.bins, range=(0.0, 2.0), density=True)

        # Calculate ratio between counts of pos and negative for each bin
        y1 = np.nan_to_num(counts_pos / (counts_pos + counts_neg), nan=1)[: int(threshold / (2 / self.bins))]
        y2 = np.nan_to_num(counts_pos / (counts_pos + counts_neg), nan=0)[int(threshold / (2 / self.bins)) :]
        ratios = np.concatenate([y1, y2])

        # Fit a sigmoid curve to the ratios
        p0 = [max(ratios), threshold, self.p0k, min(ratios)]
        x = np.arange(0, 2, 2 / self.bins)
        sigmoid_parameters, _ = curve_fit(sigmoid, x, ratios, p0, method="dogbox")

        return sigmoid_parameters


def calculate_score(sigmoid_parameters: list, cosine_distance: float) -> float:
    """Calculate the confidence score by using the parameters for the defined sigmoid function

    :param sigmoid_parameters: As defined in def sigmoid(..): a, x0, k, b
    :param cosine_distance: The cosine distance estimated from a network
    :return: The confidence score [0, 1]
    """

    return sigmoid(cosine_distance, *sigmoid_parameters)
