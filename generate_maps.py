from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import cv2


class MapGenerator:
    def __init__(self, inference_fn, shape: str = "rectangle", smooth: bool = False, fill_value: float = 0.0,
                 stride: int = 5, patch_sizes=None):
        if patch_sizes is None:
            patch_sizes = [7, 14, 28]
        self.patch_sizes = patch_sizes
        self.stride = stride
        self.fill_value = fill_value
        self.shape = shape
        self.smooth = smooth
        self.inference_fn = inference_fn

    def __call__(self, img1: np.array, img2: np.array, method: str = "1") -> np.array:
        """Calling the class generates the explanation maps according to the given method for both images
        :param img1: First image of pair
        :param img2: Second image of pair
        :param method: The method used to generate the explanation maps. Can be "1", "2" or "3"
        :return: A tuple: (explanation map for first image, explanation map for second image)
        """

        def gen_map(dists: np.array, _orig_dist: np.array, _masks: np.array, _patch_size: int) -> np.array:
            """Generate a map from the distance deviations between the occluded images (For one Image)

            :param dists: distances given for occluded images of one image
            :param _orig_dist: the distance between both non occluded images
            :param _masks: the masks from systematic image occlusion algorithm
            :param _patch_size: size of the occluding patch (needed to weight the deviation)
            :return: Explanation map for one Image
            """

            deviation = (dists - _orig_dist) / (_patch_size ** 2)
            exp_map = np.mean((1. - _masks[:, :, :, 0]) * np.expand_dims(deviation, axis=(-1, -2)), axis=0)
            return cv2.GaussianBlur(exp_map, (self.stride, self.stride), self.stride)

        def cosine_distance(_emb1: np.array, _emb2: np.array) -> float:
            """Calculates the cosine distance between two embedding vectors

            :param _emb1: embedding vector of first image
            :param _emb2: embedding vector of second image
            :return: Cosine distance value
            """

            emb1_norm = _emb1 / np.linalg.norm(_emb1, axis=0)
            emb2_norm = _emb2 / np.linalg.norm(_emb2, axis=0)
            return 1. - float(np.matmul(emb1_norm, emb2_norm.T))

        emb1 = self.inference_fn(np.expand_dims(img1, axis=0))[0]
        emb2 = self.inference_fn(np.expand_dims(img2, axis=0))[0]

        orig_dist = cosine_distance(emb1, emb2)

        exp_maps1, exp_maps2 = [], []
        for patch_size in self.patch_sizes:
            imgs1_o, imgs2_o, masks = self.systematic_occlusion(img1, img2, patch_size, self.shape, self.smooth)
            embs1 = self.inference_fn(imgs1_o)
            embs2 = self.inference_fn(imgs2_o)

            if method == "1":
                dists1, dists2 = self.method_1(embs1, embs2)
            elif method == "2":
                dists1, dists2 = self.method_2(embs1, embs2, emb1=emb1, emb2=emb2)
            elif method == "3":
                dists1, dists2 = self.method_3(embs1, embs2)
            else:
                raise Exception

            exp_maps1.append(gen_map(dists1, orig_dist, masks, patch_size))
            exp_maps2.append(gen_map(dists2, orig_dist, masks, patch_size))

        def normalize(deviations: np.array) -> np.array:
            """Normalize explanation maps from [-2, 2] first to [-1, 1] and then to the range [0, 1]

            :param deviations: The explanation maps
            :return: Normalized explanation maps
            """

            if np.max(np.abs(deviations)) != 0:
                deviations = deviations / np.max(np.abs(deviations))
            return (deviations + 1.) / 2.

        exp_map1 = plt.cm.PiYG(normalize(np.mean(exp_maps1, axis=0)))[:, :, :3]
        exp_map2 = plt.cm.PiYG(normalize(np.mean(exp_maps2, axis=0)))[:, :, :3]

        return exp_map1, exp_map2

    @staticmethod
    def method_1(embs1: np.array, embs2: np.array) -> tuple:
        """Method 1 for generating our proposed explanation maps 1:all distances

        :param embs1: embeddings of occluded images of first image of pair
        :param embs2: embeddings of occluded images of second image of pair
        :return: tuple of distances (mean distances of occluded images of first image to all occluded images of second
        image, vice versa)
        """

        dists = cdist(embs1, embs2, metric="cosine")
        dists1 = np.mean(dists, axis=1)
        dists2 = np.mean(dists, axis=0)
        return dists1, dists2

    @staticmethod
    def method_2(embs1: np.array, embs2: np.array, emb1: list, emb2: list) -> tuple:
        """Method 2 for generating our proposed explanation maps
        non_occluded:all distances

        :param embs1: embeddings of occluded images of first image of pair
        :param embs2: embeddings of occluded images of second image of pair
        :param emb1: embedding of first image of pair
        :param emb2: embedding of second image of pair
        :return: tuple of distances (distances of occluded images of first image to non-occluded second image,
        vice versa)
        """

        dists1 = paired_cosine_distances(embs1, np.repeat([emb2], embs1.shape[0], axis=0))
        dists2 = paired_cosine_distances(embs2, np.repeat([emb1], embs2.shape[0], axis=0))
        return dists1, dists2

    @staticmethod
    def method_3(embs1: np.array, embs2: np.array) -> tuple:
        """Method 3 for generating our proposed explanation maps
        1:1 distances

        :param embs1: embeddings of occluded images of first image of pair
        :param embs2: embeddings of occluded images of second image of pair
        :return: tuple of distances (distances between occluded images of both images, same)
        """

        dists1 = dists2 = paired_cosine_distances(embs1, embs2)
        return dists1, dists2

    def systematic_occlusion(self, img1: np.array, img2: np.array, patch_size: int, shape: str,
                             smooth: bool = True) -> tuple:
        """Systematic Image Occlusion Algorithm

        :param img1: First image of a pair
        :param img2: Second image of a pair
        :param patch_size: size of the occluding patch in pixels
        :param shape: shape of the occluding patch. Can be: "rectangle" or "circle"
        :param smooth: smooth or sharp edges of the occluding patch. Can be: "True" -> smooth or "False" -> sharp
        :return: tuple of: (occluded images of first image, occluded images of second image, masks containing the
        occlusions)
        """

        xs = ys = np.arange(0, img1.shape[0] - patch_size, self.stride)
        w = h = patch_size
        img1_o, img2_o, patch_masks = [], [], []
        for y in ys:
            for x in xs:
                if shape == "rectangle":
                    mask = cv2.rectangle(np.ones_like(img1), (x, y), (x + w, y + h), (0, 0, 0), -1)
                elif shape == "circle":
                    mask = cv2.circle(np.ones_like(img1), (x, y), w // 2, (0, 0, 0), -1)
                else:
                    raise Exception

                if smooth:
                    mask = cv2.GaussianBlur(mask, (patch_size % 2 - 1, patch_size % 2 - 1), patch_size // 7)

                fill_img = np.ones_like(img1) * (
                    np.random.random(img1.shape) if self.fill_value == "random" else self.fill_value)

                img1_o.append(img1 * mask + fill_img * (1 - mask))
                img2_o.append(img2 * mask + fill_img * (1 - mask))
                patch_masks.append(mask)
        return np.asarray(img1_o), np.asarray(img2_o), np.asarray(patch_masks)


def save_maps(exp_map1: np.array, exp_map2: np.array, base_path: str, pair_id: int) -> None:
    """Write maps of both images to disc

    :param exp_map1: Explanation map for first image
    :param exp_map2: Explanation map for second image
    :param base_path: Path to the save directory
    :param pair_id: ID of the pair
    :return:
    """

    path = f"{base_path}/exp_map_pair_id_{pair_id:06d}_img_1.png"
    cv2.imwrite(path, np.clip(exp_map1[:, :, ::-1] * 255, 0, 255).astype(np.uint8))

    path = f"{base_path}/exp_map_pair_id_{pair_id:06d}_img_2.png"
    cv2.imwrite(path, np.clip(exp_map2[:, :, ::-1] * 255, 0, 255).astype(np.uint8))
