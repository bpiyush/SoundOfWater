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


**TL;DR**: We present a method to infer physical properties of liquids from *just* the sound of pouring. We show in theory how *pitch* can be used to derive various physical properties such as container height, flow rate, etc. Then, we train a pitch detection network (`wav2vec2`) using simulated and real data. The resulting model can predict the physical properties of pouring liquids with high accuracy. The latent representations learned also encode information about liquid mass and container shape.


<!-- Add a teaser image. -->
<p align="center">
  <img src="./media_assets/teaser.png" alt="Teaser" width="100%">
</p>

## 📅 Updates

## Table of Contents


## Highlights

This should give the key highlight results.


## Dataset

This should include visualisation and description of the dataset. This should also include instructions to download the dataset.

## Playground

We provide a single notebook to run the model and visualise results.
This needs to be the inference pipeline notebook.

This should start with instructions for installation.
Then, it should include a brief description of the notebook and how to use it.
Finally, it should include a link to the notebook.

## Results

This should include quantitative and some qualitative results to show what the method can do.

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