.. raw:: html

   <p align="center">

.. raw:: html

   <p>

XAISuite: Training and Explaining Machine Learning Models
=========================================================
Welcome to XAISuite's documentation!

But first, let's get past the formalities.

Here's a brief overview of this project. We'd love for you to get to know us better. However, if you want to go straight to the documentation, scroll to the bottom of the page or use the menu on the left sidebar. Good luck!

1. `Introduction <#introduction>`__
2. `Installation <#installation>`__
3. `Getting Started <#getting-started>`__
4. `How to Contribute <#how-to-contribute>`__
5. `Technical Report and Citing
   XAISuite <#technical-report-and-citing-xaisuite>`__

Introduction
------------

XAISuite (Explanatory Artificial Intelligence Suite is a library for
training and explaining machine learning models for tabular datasets in
Python. It provides a unified interface for training any sklearn model
using just a line of code and allows users to easily comparing the
results of different explainers!

XAISuite accomplishes machine learning model training and explanation
generation in three steps: (1) data loading, (2) model training and (3)
explanation generation. Each of these steps are delved into more detail
in our documentation
and demo tutorial.

.. figure:: https://user-images.githubusercontent.com/66180831/209634297-296fa5d8-4429-434c-afaa-7500d776cd75.png
   :alt: Screen_Shot_2022-12-27_at_12 10 36_AM-removebg-preview

   Screen_Shot_2022-12-27_at_12 10 36_AM-removebg-preview

XAISuite was created as a helper library to `this
paper <insert%20link>`__, which studied the difference in SHAP and LIME
explanations for different models on tabular datasets.

Installation
------------

You can install the ``XAI Suite`` through PyPI:

``pip install XAISuite``

Getting Started
---------------

For example code and an introduction to the library, see the Demo
Folder.

If you are looking for a model or dataset to use,
`sklearn <https://scikit-learn.org/stable/>`__ has several cool options.

How to Contribute
-----------------

We welcome the contribution from the open-source community to improve
the library!

To add a new functionality into the library, please follow the template
and steps demonstrated in our documentation. 

Note that, for the time being, XAISuite only supports tabular datasets
and image datasets in tabular form.

Technical Report and Citing XAISuite
------------------------------------

A paper proposing and using XAISuite to compare explanatory methods is
still in pre-publication. Use the following BibTex to cite XAISuite:

::

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

Contact Us
----------

If you have any questions, comments or suggestions, please do not
hesitate to contact us at shreyan.m.mitra@gmail.com

License
-------

This work is licensed under a `BSD 3-Clause License <LICENSE>`__.


Documentation
------------
Finally. 

Here's what you were actually looking for! Click the following link for access to the tutorial and XAISuite documentation.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   xaisuite
   demo
