# <img src="./media_assets/pouring-water-logo5.png" alt="Logo" width="40">  The Sound of Water: Inferring Physical Properties from Pouring Liquids


<p align="center">
  <a href="https://arxiv.org/abs/XXXXXX" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-Paper-red" alt="arXiv">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a target="_blank" href="https://colab.research.google.com/github/bpiyush/SoundOfWater/blob/main/playground.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://your_gradio_demo_link" target="_blank">
    <img src="https://img.shields.io/badge/Gradio-Demo-orange" alt="Gradio">
  </a>
</p>

<!-- Add a teaser image. -->
<p align="center">
  <img src="./media_assets/pitch_on_spectrogram-compressed.gif" alt="Teaser" width="100%">
</p>

*Key insight*: As water is poured, the fundamental frequency that we hear changes predictably over time as a function of physical properties (e.g., container dimensions).


**TL;DR**: We present a method to infer physical properties of liquids from *just* the sound of pouring. We show in theory how *pitch* can be used to derive various physical properties such as container height, flow rate, etc. Then, we train a pitch detection network (`wav2vec2`) using simulated and real data. The resulting model can predict the physical properties of pouring liquids with high accuracy. The latent representations learned also encode information about liquid mass and container shape.


## 📅 Updates

## 📑 Table of Contents

- [  The Sound of Water: Inferring Physical Properties from Pouring Liquids](#--the-sound-of-water-inferring-physical-properties-from-pouring-liquids)
  - [📅 Updates](#-updates)
  - [📑 Table of Contents](#-table-of-contents)
  - [✨ Highlights](#-highlights)
  - [📂 Dataset](#-dataset)
  - [🤖 Models](#-models)
  - [🎮 Playground](#-playground)
  - [📊 Results](#-results)
  - [📜 Citation](#-citation)
  - [🙏 Acknowledgements](#-acknowledgements)


## ✨ Highlights

1. We train a `wav2vec2` model to estimate the pitch of pouring water. We use supervision from simulated data and fine-tune on real data using visual co-supervision.
2. We show physical property estimation from pitch. For example, in estimating the height of the container, we achieve a mean absolute error of 2.2 cm, in radius estimation, 1.6 cm and in estimating length of air column, 0.6 cm.
3. We show strong generalisation to other datasets (e.g., [Wilson et al.](https://gamma.cs.unc.edu/PSNN/)) and some videos from YouTube.
4. We also show that the learned representations can be regressed to estimate the mass of the liquid and the shape of the container.
5. We release a clean dataset of 805 videos of water pouring with annotations for physical properties.

## 📂 Dataset

We collect a dataset of 805 clean videos that show the action of pouring water in a container. Our dataset spans over 50 unique containers made of 5 different materials, 4 different shapes and with hot and cold water. Some example containers are shown below.

<p align="center">
  <img width="650" alt="image" src="./media_assets/containers-v2.png">
</p>

The dataset is available to download [here]([.](https://huggingface.co/datasets/bpiyush/sound-of-water)).

**Option 1:** Download from `huggingface` 

```py
# Note: this shall take 5-10 mins.

# Optionally, disable progress bars
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = True

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="bpiyush/sound-of-water",
    repo_type="dataset",
    local_dir="/path/to/dataset/SoundOfWater",
)
```
The total size of the dataset is 1.4 GB.

**Option 2:** Download from VGG servers

Coming soon!


## 🤖 Models

We provide trained models for pitch estimation.

<table style="font-size: 12px;">
<tr>
  <th>File link</th>
  <th>Description</th>
  <th>Size</th>
</tr>
<tr>
  <td> <a href="url">synthetic_pretrained.pth</a> </td>
  <td>Pre-trained on synthetic data &nbsp;&nbsp;&nbsp;</td>
  <td>361M</td>
</tr>
<tr>
  <td> <a href="url">real_finetuned_visual_cosupervision.pth</a> </td>
  <td>Trained with visual co-supervision &nbsp;&nbsp;&nbsp;</td>
  <td>361M</td>
</tr>
</table>

The models are available to download [here](https://huggingface.co/bpiyush/sound-of-water-models).


**Option 1:** Download from `huggingface`. Use this snippet to download the models:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bpiyush/sound-of-water-models",
    local_dir="/path/to/download/",
)
```

**Option 2:** Download from VGG servers

Coming soon!


## 🎮 Playground

We provide a single [notebook](./playground.ipynb) to run the model and visualise results.
We walk you through the following steps:
- Load data
- Demo the physics behind pouring water
- Load and run the model
- Visualise the results

Before running the notebook, be sure to install the required dependencies:

```bash
conda create -n sow python=3.8
conda activate sow

# Install desired torch version
# NOTE: change the version if you are using a different CUDA version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Additional packages
pip install lightning==2.1.2
pip install timm==0.9.10
pip install pandas
pip install decord==0.6.0
pip install librosa==0.10.1
pip install einops==0.7.0
pip install ipywidgets jupyterlab seaborn

# if you find a package is missing, please install it with pip
```

Remember to download the model in the previous step. Then, run the notebook.

## 📊 Results

We show key results in this section. Please refer to the paper for more details.

<p align="center">
<img width="650" alt="image" src="https://github.com/user-attachments/assets/34b0ea66-5ee7-4338-bf04-f0b20f87d0de">

<img width="650" alt="image" src="https://github.com/user-attachments/assets/7193001b-1485-42b5-aa25-feab777e9921">

<img width="650" alt="image" src="https://github.com/user-attachments/assets/9cf2a960-af8b-4df3-b714-6755b5bb90f6">
</p>


<!-- Add a citation -->
## 📜 Citation

If you find this repository useful, please consider giving a star ⭐ and citation

```bibtex
@article{sound_of_water_bagad,
  title={The Sound of Water: Inferring Physical Properties from Pouring Liquids},
  author={Bagad, Piyush and Tapaswi, Makarand and Snoek, Cees G. M. and Zisserman, Andrew},
  journal={arXiv},
  year={2024}
}
```

<!-- Add acknowledgements, license, etc. here. -->
## 🙏 Acknowledgements

* We thank Ashish Thandavan for support with infrastructure and Sindhu
Hegde, Ragav Sachdeva, Jaesung Huh, Vladimir Iashin, Prajwal KR, and Aditya Singh for useful
discussions.
* This research is funded by EPSRC Programme Grant VisualAI EP/T028572/1, and a Royal Society Research Professorship RP / R1 / 191132.

We also want to highlight closely related work that could be of interest:

* [Analyzing Liquid Pouring Sequences via Audio-Visual Neural Networks](https://gamma.cs.unc.edu/PSNN/). IROS (2019).
* [Human sensitivity to acoustic information from vessel filling](https://psycnet.apa.org/record/2000-13210-019). Journal of Experimental Psychology (2020).
* [See the Glass Half Full: Reasoning About Liquid Containers, Their Volume and Content](https://arxiv.org/abs/1701.02718). ICCV (2017).
* [CREPE: A Convolutional Representation for Pitch Estimation](https://arxiv.org/abs/1802.06182). ICASSP (2018).
