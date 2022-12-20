import tensorflow as tf
import numpy as np
from tqdm import tqdm


class ArcFaceOctupletLoss:
    def __init__(self, batch_size):
        self.model = tf.keras.models.load_model("./ArcFaceOctupletLoss.tf")
        self.batch_size = batch_size

    @staticmethod
    def __preprocess(img):
        if img.ndim != 4:
            img = np.expand_dims(img, axis=0)
        return img

    def __inference(self, img):
        return self.model.inference(self.__preprocess(img))

    def __call__(self, imgs):
        embs = []
        for i in tqdm(range(0, imgs.shape[0], self.batch_size), desc="Performing Inference: "):
            embs.append(self.__inference(imgs[i : i + self.batch_size]))
        return np.concatenate(embs)
