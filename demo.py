# ==================================================================================================================== #
#                                                                                                                      #
#                                 *** EXAMPLE CODE FOR CONFIDENCE SCORE CALCULATION ***                                #
#                                                                                                                      #
# ==================================================================================================================== #

import pickle
from src import ConfidenceScoreGenerator, calculate_score
from sklearn.metrics.pairwise import paired_cosine_distances


# Load data (example embeddings from XQLFW dataset with FaceTransformer model fine-tuned with OctupletLoss)
with open("./demo/embeddings1.pkl", "rb") as f:
    embeddings1 = pickle.load(f)
with open("./demo/embeddings2.pkl", "rb") as f:
    embeddings2 = pickle.load(f)
with open("./demo/labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Calculate pairwise cosine distances
distances = paired_cosine_distances(embeddings1, embeddings2)

# Define a threshold for binary decision
THRESHOLD = 0.2250

# Initiate the confidence calculation with the distances, labels and threshold of a dataset
ConfidenceScoreGenerator = ConfidenceScoreGenerator()
sigmoid_parameters = ConfidenceScoreGenerator(cosine_distances=distances, labels=labels, threshold=THRESHOLD)

# Calculate scores for specific pairs
PAIR_ID = 0
confidence_score_raw = calculate_score(sigmoid_parameters, distances[PAIR_ID])
confidence_score = (confidence_score_raw if distances[PAIR_ID] < THRESHOLD else 1 - confidence_score_raw) * 100

# Display result
print(
    f'The prediction: "{distances[PAIR_ID] < THRESHOLD}" for pair id: {PAIR_ID} '
    f"has a confidence score of: {confidence_score:.02f}%."
)


# ==================================================================================================================== #
#                                                                                                                      #
#                                 *** EXAMPLE CODE FOR GENERATING EXPLANATION MAPS ***                                 #
#                                                                                                                      #
# ==================================================================================================================== #

import matplotlib.pyplot as plt
import cv2
from src import MapGenerator, colorblend
from demo import ArcFaceOctupletLoss
import numpy as np


# Instantiate the MapGenerator
MapGenerator = MapGenerator(inference_fn=ArcFaceOctupletLoss(batch_size=64))

# Load an example image pair
image_pair = (
    cv2.cvtColor(cv2.imread("./demo/img1.png"), cv2.COLOR_BGR2RGB).astype(np.float32) / 255,
    cv2.cvtColor(cv2.imread("./demo/img2.png"), cv2.COLOR_BGR2RGB).astype(np.float32) / 255,
)

# Show example image pair
fig, ax = plt.subplots(1, 2)
fig.suptitle("Example Image Pair")
ax[0].imshow(image_pair[0]), ax[1].imshow(image_pair[1])
plt.show()

# Generate and visualize the explanation maps
fig, ax = plt.subplots(3, 2)
fig.suptitle("Explanation Maps for Method 1, 2 and 3")
map1_m1, map2_m1 = MapGenerator(*image_pair, method="1")  # using method 1 for explanation maps
ax[0, 0].imshow(map1_m1), ax[0, 1].imshow(map2_m1)
map1_m2, map2_m2 = MapGenerator(*image_pair, method="2")  # using method 2 for explanation maps
ax[1, 0].imshow(map1_m2), ax[1, 1].imshow(map2_m2)
map1_m3, map2_m3 = MapGenerator(*image_pair, method="3")  # using method 3 for explanation maps
ax[2, 0].imshow(map1_m3), ax[2, 1].imshow(map2_m3)
plt.show()

# Blend the explanations maps with the original images and visualize
fig, ax = plt.subplots(3, 2)
fig.suptitle("Blended Explanation Maps for Method 1, 2 and 3")
ax[0, 0].imshow(colorblend(image_pair[0], map1_m1)), ax[0, 1].imshow(colorblend(image_pair[1], map2_m1))
ax[1, 0].imshow(colorblend(image_pair[0], map1_m2)), ax[1, 1].imshow(colorblend(image_pair[1], map2_m2))
ax[2, 0].imshow(colorblend(image_pair[0], map1_m3)), ax[2, 1].imshow(colorblend(image_pair[1], map2_m3))
plt.show()
