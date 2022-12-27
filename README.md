<p align="center">
    <br>
    <img src="https://user-images.githubusercontent.com/66180831/209478341-a1b4d80b-dbcb-448c-a3e0-109e27590ec5.png" width="400"/>
    <br>
<p>

# XAISuite: Training and Explaining Machine Learning Models
<div align="center">
  <a href="#">
  <img src="https://img.shields.io/badge/Python-3.7, 3.8, 3.9, 3.10-blue">
  </a>
  
  <a href="https://pypi.python.org/pypi/XAISuite">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/XAISuite"/>
  </a>
  
  <a href="https://pepy.tech/project/XAISuite">
  <img alt="Downloads" src="https://pepy.tech/badge/xaisuite">
  </a>
  
  <a href="https://github.com/11301858/XAISuite">
  <img alt="Documentation" src="https://github.com/11301858/XAISuite/actions/workflows/docs.yml/badge.svg"/>
  </a>
  
  <!-- Some more badges to display, upon release
  <a href="https://arxiv.org/abs/2206.01612">
  <img alt="DOI" src="https://zenodo.org/badge/DOI/10.48550/ARXIV.2206.01612.svg"/>
  </a>
  -->
</div>

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [How to Contribute](#how-to-contribute)
5. [Technical Report and Citing XAISuite](#technical-report-and-citing-xaisuite)


## Introduction

XAISuite (E<u><b>x</b></u>planatory <u><b>A</b></u>rtificial <u><b>I</b></u>ntelligence <u><b>Suite</b></u>) is a library for training and explaining machine learning models for tabular datasets in Python. It provides a unified interface for training any sklearn model using just a line of code and allows users to easily comparing the results of different explainers!

XAISuite accomplishes machine learning model training and explanation generation in three steps: (1) data loading, (2) model training and (3)
explanation generation. Each of these steps are delved into more detail in our [documentation](https://11301858.github.io/XAISuite/v0.6.0-beta/index.html) and in the demo tutorials.


![Screen_Shot_2022-12-27_at_12 10 36_AM-removebg-preview](https://user-images.githubusercontent.com/66180831/209634297-296fa5d8-4429-434c-afaa-7500d776cd75.png)

XAISuite was created as a helper library to [this paper](insert link), which studied the difference in SHAP and LIME explanations for different models on tabular datasets.

## Installation
You can install the ``XAI Suite`` through PyPI:

``
pip install XAISuite
``

## Getting Started

For example code and an introduction to the library, see the Demo Folder. 

If you are looking for a model or dataset to use, [sklearn](https://scikit-learn.org/stable/) has several cool options.


## How to Contribute

We welcome the contribution from the open-source community to improve the library!

To add a new functionality into the library, please follow the template and steps demonstrated in this 
[documentation](https://11301858.github.io/XAISuite/v0.6.0-beta/index.html). Note that, for the time being, XAISuite only supports tabular datasets and image datasets in tabular form.

## Technical Report and Citing XAISuite
A paper proposing and using XAISuite to compare explanatory methods is still in pre-publication. Use the following BibTex to cite XAISuite:

```
@article{mitra2022-xaisuite,
  author    = {Shreyan Mitra and Leilani Gilpin},
  title     = {Comparison of SHAP and LIME Explanations for Supervised
Machine Learning Models Trained on Tabular Datasets},
  year      = {2022},
  doi       = {},
  url       = {},
  archivePrefix = {},
  eprint    = {},
}
```


## Contact Us
If you have any questions, comments or suggestions, please do not hesitate to contact us at shreyan.m.mitra@gmail.com

## License

This work is licensed under a [BSD 3-Clause License](LICENSE). 
