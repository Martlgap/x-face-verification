from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
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


class ConfidenceScoreGenerator:
    def __init__(self, bins: int = 2000, p0k: int = -20):
        self.bins = bins
        self.p0k = p0k

    @staticmethod
    def __acc(threshold: float, dist: np.array, issame: np.array):
        """Calculates accuracy of the binary classification for the given threshold, distances and labels

        :param threshold: The decision boundary
        :param dist: Array of distances
        :param issame: Labels for binary classification
        :return: TPR, FPR, Accuracy
        """

        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), issame))
        actual_tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        actual_fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / dist.size
        return actual_tpr, actual_fpr, acc

    def __call__(self, cosine_distances, labels) -> list:
        """The algorithm to derive the parameters for the sigmoid function which calculated the confidence score

        :param cosine_distances: Cosine distances for a dataset of image pairs
        :param labels: Corresponding labels "True" -> genuine and "False" -> imposter image pairs
        :return: - Four parameters in a list: As defined in def sigmoid(..): a, x0, k, b 
                 - The best threshold for the cosine distances
        """

        steps = np.arange(0, 2, 2 / self.bins)

        # Find the best threshold for the fold
        acc_train = np.zeros(len(steps))
        for threshold_idx, threshold_train in enumerate(steps):
            _, _, acc_train[threshold_idx] = self.__acc(threshold_train, cosine_distances, labels)
        threshold = steps[np.argmax(acc_train)]

        # Select distances for genuine and imposter pairs
        dists_pos = cosine_distances[np.where(labels)]
        dists_neg = cosine_distances[np.where(np.logical_not(labels))]

        # Generate histogram and count the number of distance for each bin
        counts_pos, _ = np.histogram(dists_pos, bins=self.bins, range=(0.0, 2.0), density=True)
        counts_neg, _ = np.histogram(dists_neg, bins=self.bins, range=(0.0, 2.0), density=True)

        # Calculate ratio between counts of pos and negative for each bin
        y1 = np.nan_to_num(counts_pos / (counts_pos + counts_neg), nan=1)[:int(threshold / (2 / self.bins))]
        y2 = np.nan_to_num(counts_pos / (counts_pos + counts_neg), nan=0)[int(threshold / (2 / self.bins)):]
        ratios = np.concatenate([y1, y2])

        # Fit a sigmoid curve to the ratios
        p0 = [max(ratios), threshold, self.p0k, min(ratios)]
        x = np.arange(0, 2, 2 / self.bins)
        sigmoid_parameters, _ = curve_fit(sigmoid, x, ratios, p0, method='dogbox')
        return sigmoid_parameters, threshold


    def foldwise(self, cosine_distances, labels, k_folds) -> list:
        """Foldwise __call__ method
        
        """

        k_fold = KFold(n_splits=k_folds, shuffle=False)
        all_sigmoid_parameters, all_thresholds = [], [] 
        # Calculate sigmoid parameters and best threshold for each fold
        for _, (train_set, _) in enumerate(k_fold.split(np.arange(len(cosine_distances)))):
            sigmoid_params, threshold = self.__call__(cosine_distances[train_set], labels[train_set])
            all_sigmoid_parameters.append(sigmoid_params)
            all_thresholds.append(threshold)
        return all_sigmoid_parameters, all_thresholds


def calculate_score(sigmoid_parameters: list, cosine_distance: float) -> float:
    """Calculate the confidence score by using the parameters for the defined sigmoid function

    :param sigmoid_parameters: As defined in def sigmoid(..): a, x0, k, b
    :param cosine_distance: The cosine distance estimated from a network
    :return: The confidence score [0, 1]
    """

    return sigmoid(cosine_distance, *sigmoid_parameters)


def threshold_flip(cosine_distances: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """ Flip the confidence score if distance is greater than the given threshold -> Make scores between 0.5 and 1
    
    :param cosine_distances: Cosine distances for a dataset of image pairs
    :param scores: Confidence scores for a dataset of image pairs
    :param threshold: The decision boundary
    :return: Flipped and clipped confidence scores
    """
    
    for idx, dist in enumerate(cosine_distances):
        if dist > threshold:
            scores[idx] = 1 - scores[idx]
    return np.clip(scores, 0.5, 1.0)
