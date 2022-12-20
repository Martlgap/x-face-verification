# 👋 Welcome 👋
To the official repo for the paper:

[Explainable Model-Agnostic Similarity and Confidence in Face Verification](https://arxiv.org/pdf/2211.13735.pdf)


# 📄 Abstract 📄
Recently, face recognition systems have demonstrated remarkable performances and thus gained a vital role in our daily life. They already surpass human face verification accountability in many scenarios. However, they lack explanations for their predictions. Compared to human operators, typical face recognition network system generate only binary decisions without further explanation and insights into those decisions. This work focuses on explanations for face recognition systems, vital for developers and operators. First, we introduce a confidence score for those systems based on facial feature distances between two input images and the distribution of distances across a dataset. Secondly, we establish a novel visualization approach to obtain more meaningful predictions from a face recognition system, which maps the distance deviation based on a systematic occlusion of
images. The result is blended with the original images and highlights similar and dissimilar facial regions. Lastly, we calculate confidence scores and explanation maps for several state-of-the-art face verification datasets and release the results on a web platform. We optimize the platform for a user-friendly interaction and hope to further improve the understanding of machine learning decisions. The source code is available on GitHub, and the web platform is publicly available.


# 🖥 Preview 🖥
Here is a sample of three generated explanation maps: 

<img src="examples.png" alt="Example Explanation Maps" width=500>

# 🚀 Getting Started 🚀

### Prerequisites:
- Find the code to generate our proposed explanation maps [here](code/generate_maps.py). 
- Find the code to calculate our proposed confidence score [here](code/calculate_score.py).

### How to run the Demo:

- Install all requirements for Python3.9: ```pip install -r requirements.txt```
- Download the ArcFaceOctupletLoss model from [here](https://github.com/Martlgap/octuplet-loss/releases/download/modelweights/ArcFaceOctupletLoss.tf.zip) and extract the .zip archive into the "demo" folder
- Run the [main.py](demo/main.py) script inside the [demo](demo) folder with ```python main.py```

# 🌎 Web-Platform 🌎

To explore our results we published our explainable-face-verification web-platform:

[https://explainable-face-verification.ey.r.appspot.com](https://explainable-face-verification.ey.r.appspot.com)

<img src="website.png" alt="Website Preview" width=800>

# 📚 Citation 📚
If you find our work useful, please consider a citation:

```latex
@misc{knocheexplainable2022,
  doi = {10.48550/ARXIV.2211.13735},
  url = {https://arxiv.org/abs/2211.13735},
  author = {Knoche, Martin and Teepe, Torben and Hörmann, Stefan and Rigoll, Gerhard},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Explainable Model-Agnostic Similarity and Confidence in Face Verification},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```