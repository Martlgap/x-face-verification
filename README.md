# 👋 Welcome 👋
To the official repo for the paper:

[Explainable Model-Agnostic Similarity and Confidence in Face Verification]([https://arxiv.org/pdf/2211.13735.pdf](https://openaccess.thecvf.com/content/WACV2023W/XAI4B/html/Knoche_Explainable_Model-Agnostic_Similarity_and_Confidence_in_Face_Verification_WACVW_2023_paper.html)


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

### How to run the Demo locally:

- Install all requirements for Python3.10: ```pip install -r requirements.txt```
- Download the ArcFaceOctupletLoss model from [here](https://github.com/Martlgap/octuplet-loss/releases/download/modelweights/ArcFaceOctupletLoss.tf.zip) and extract the .zip archive into the "demo" folder
- Run the [demo.py](demo.py) script: ```python demo.py```

### How to run the Demo in GitHub Codespaces:
- First install python packages and download/extract model:
```bash
pip install -r requirements.txt
wget https://github.com/Martlgap/octuplet-loss/releases/download/modelweights/ArcFaceOctupletLoss.tf.zip -P demo
unzip demo/ArcFaceOctupletLoss.tf.zip -d demo
```
- Run the [demo.py](demo.py) script in Codespace

### How to run the Demo in Google Colab:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martlgap/x-face-verification/blob/main/demo.ipynb)
- Run the Notebook in GoogleColab
# 🌎 Web-Platform 🌎

To explore our results we published our explainable-face-verification web-platform:

[https://explainable-face-verification.ey.r.appspot.com](https://explainable-face-verification.ey.r.appspot.com)

<img src="website.png" alt="Website Preview" width=800>

# 📚 Citation 📚
If you find our work useful, please consider a citation:

```latex
@inproceedings{knoche2023explainable,
  title={Explainable Model-Agnostic Similarity and Confidence in Face Verification},
  author={Knoche, Martin and Teepe, Torben and H{\"o}rmann, Stefan and Rigoll, Gerhard},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={711--718},
  year={2023}
}
```
