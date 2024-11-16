# <img src="./media_assets/pouring-water-logo5.png" alt="Logo" width="40">  The Sound of Water: Inferring Physical Properties from Pouring Liquids


<p align="center">
  <a href="https://arxiv.org/abs/XXXXXX" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-Paper-red" alt="arXiv">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://colab.research.google.com/your_notebook_link" target="_blank">
    <img src="https://img.shields.io/badge/Colab-Demo-brightgreen" alt="Colab">
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

*Our key observations*: As water is poured, the fundamental frequency that we hear changes predictably over time as a function of physical properties (e.g., container dimensions).


**TL;DR**: We present a method to infer physical properties of liquids from *just* the sound of pouring. We show in theory how *pitch* can be used to derive various physical properties such as container height, flow rate, etc. Then, we train a pitch detection network (`wav2vec2`) using simulated and real data. The resulting model can predict the physical properties of pouring liquids with high accuracy. The latent representations learned also encode information about liquid mass and container shape.


## 📅 Updates

## Table of Contents

- [  The Sound of Water: Inferring Physical Properties from Pouring Liquids](#--the-sound-of-water-inferring-physical-properties-from-pouring-liquids)
  - [📅 Updates](#-updates)
  - [Table of Contents](#table-of-contents)
  - [Highlights](#highlights)
  - [Dataset](#dataset)
  - [Playground](#playground)
  - [Results](#results)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)


## Highlights

1. We train a `wav2vec2` model to estimate the pitch of pouring water. We use supervision from simulated data and fine-tune on real data using visual co-supervision.
2. We show physical property estimation from pitch. For example, in estimating the height of the container, we achieve a mean absolute error of 2.2 cm, in radius estimation, 1.6 cm and in estimating length of air column, 0.6 cm.
3. We show strong generalisation to other datasets (e.g., [Wilson et al.](https://gamma.cs.unc.edu/PSNN/)) and some videos from YouTube.
4. We also show that the learned representations can be regressed to estimate the mass of the liquid and the shape of the container.
5. We release a clean dataset of 805 videos of water pouring with annotations for physical properties.

## Dataset

This should include visualisation and description of the dataset. This should also include instructions to download the dataset.

## Playground

We provide a single notebook to run the model and visualise results.
This needs to be the inference pipeline notebook.

This should start with instructions for installation.
Then, it should include a brief description of the notebook and how to use it.
Finally, it should include a link to the notebook.

## Results

We show key results in this section. Please refer to the paper for more details.

<img width="778" alt="image" src="https://github.com/user-attachments/assets/34b0ea66-5ee7-4338-bf04-f0b20f87d0de">

<img width="778" alt="image" src="https://github.com/user-attachments/assets/7193001b-1485-42b5-aa25-feab777e9921">

<img width="778" alt="image" src="https://github.com/user-attachments/assets/9cf2a960-af8b-4df3-b714-6755b5bb90f6">


<!-- Add a citation -->
## Citation

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
## Acknowledgements

We want to thank ....

We also want to highlight closely related work that could be of interest:

* Paper 1
* Paper 2
* ...
