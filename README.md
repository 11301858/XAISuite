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
  <img alt="Downloads" src="https://static.pepy.tech/badge/xaisuite">   
  </a>
  
  <a href="https://github.com/11301858/XAISuite">
  <img alt="Documentation" src="https://github.com/11301858/XAISuite/actions/workflows/docs.yml/badge.svg"/>
  </a>
  
  <!-- Some more badges to display, upon release of research paper
  <a href="https://arxiv.org/abs/2206.01612">
  <img alt="DOI" src="https://zenodo.org/badge/DOI/.svg"/>
  </a>
  -->
</div>

Welcome to our source page. Our mission is to make machine learning available to all! Whether you are a data scientist, researcher, or just a person curious about how you can use artificial intelligence to your advantage, XAISuite is the library for you. Please be sure to contribute and contact us if you have any questions. 

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [How to Contribute](#how-to-contribute)
5. [Technical Report and Citing XAISuite](#technical-report-and-citing-xaisuite)


## Introduction

XAISuite (E<b>x</b>planatory <b>A</b>rtificial <b>I</b>ntelligence <b>Suite</b>) is a library for *convenient* training and explaining machine learning models for tabular datasets in Python (Note the emphasis on the word convenient). It provides a unified interface for training and explaining any machine learning model using at most just a line of code. It allows users to easily compare the results of different explainers. It is based on the XAISuite framework, which we propose in our paper. 

What are explanations? Machine learning models are opaque models, so we have no idea what's going on inside of them. Explainers help us understand machine learning models we have trained and therefore give us aa better idea of why machine learning models fail in particular instances.

XAISuite accomplishes machine learning model training and explanation generation in three steps: (1) data loading, (2) model training and explanation generation, and (3) explanation comparison. Each of these steps are delved into more detail in our [documentation](https://11301858.github.io/XAISuite/v0.6.7-beta/index.html) and in the demo tutorials. A detailed flowchart is presented in our paper.

A key part of XAISuite is flexibility, and, in our mission to make machine learning available to all, we have made or plan to make XAISuite available in the following formats:

1. As a Python Library (with XAISuite and XAISuiteGUI)
2. On the Command Line (with XAISuiteCLI)
3. In block-code (with XAISuiteBlock)
4. In the XAI Programming Language (Pending)

![XAISuite options](https://user-images.githubusercontent.com/66180831/222034540-5ae92a6f-2100-4c5c-ad60-aa47857fef4c.png)


As far as we know, XAISuite is among the first comprehensive libraries that allow users to both train and explain models, and the first to provide utilities for explanation comparison. XAISuite was created with a focus on users, and our interface reflects that.

We also pioneered the ability to interact with machine learning models on the command line. 

## Installation

You can install the ``XAI Suite`` through PyPI:

``
pip install XAISuite
``

This will automatically install the latest version and is the reccomended way to download the library. The version on Github may not be stable. If yu already have XAISuite and want to upgrade it, do:

``
pip install XAISuite --upgrade
``
Follow the instructions in individual folder READMEs for further installation instructions. For example, to install the command-line tool for XAISuite, do


``
brew install xaisuitecli
``

## Getting Started

For comprehensive example code and an introduction to the library, see the Demo Folder. The Demo folder is never fully complete and we will add more and more tutorials as the project progresses.

If you are looking for a model or dataset to use, [sklearn](https://scikit-learn.org/stable/) has several cool options.

Examples of graphs and tables generated by the XAISuite Library can be found [here](https://drive.google.com/drive/u/2/folders/10t4_GYDPJl2sM9hDOuezbum-yqKpN4fc).

Follow the instructions in individual folder READMEs for further installation instructions.

Below, we include an example of explaining a Tensorflow Keras Model as a demonstration of what XAISuite can accomplish. This example was partially taken from the SciKeras Getting Started Example to help beginners learning Tensorflow.

```python
import numpy as np
from sklearn.datasets import make_classification
from tensorflow import keras
from xaisuite import*

def get_model(hidden_layer_dim, meta):
    # note that meta is a special argument that will be
    # handed a dict containing input metadata
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:]))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(hidden_layer_dim))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(n_classes_))
    model.add(keras.layers.Activation("softmax"))
    return model


train_and_explainModel("KerasClassifier"
                      , generate_data("classification", "target", n_samples = 1000, n_features = 20, n_informative=10, random_state=0)
                      , build_fn=get_model
                      , loss="sparse_categorical_crossentropy"
                      , hidden_layer_dim=100
                      , epochs = 51
                      )
```

## How to Contribute

We welcome the contribution from the open-source community to improve the library!

To add a new functionality into the library or point out a flaw, please create a new issue on Github. We'll try to look into your requests as soon as we can. Keep in mind that, as this is an open-source project, you release any copyright protection over code you may contribute to the XAISuite Project.

## Technical Report and Citing XAISuite
A paper proposing and using XAISuite to compare explanatory methods is still in pre-publication. Use the following BibTex to cite XAISuite for now:

```
@article{mitra2022-xaisuite,
  author    = {Shreyan Mitra and Leilani Gilpin},
  title     = {The XAISuite Framework and Implications on Explanatory System Dissonance},
  year      = {2022},
  doi       = {},
  url       = {},
  archivePrefix = {},
  eprint    = {},
}
```
The paper uses XAISuite to compare SHAP and LIME explanations for different machine learning models. 

## Contact Us
If you have any questions, comments or suggestions, please do not hesitate to contact us at xaisuite@gmail.com 

## License

This work is licensed under a [BSD 3-Clause License](LICENSE). 
