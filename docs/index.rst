.. tangermeme documentation master file, created by
   sphinx-quickstart on Tue Feb 20 13:46:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


    .. image:: logo/pomegranate-logo.png
        :width: 300px


    .. image:: https://readthedocs.org/projects/pomegranate/badge/?version=latest
       :target: http://pomegranate.readthedocs.io/en/latest/?badge=latest


tangermeme
==========

tangermeme is a Python package that implements the basic operations necessary to perform sophisticated genomic analyses using machine learning models. Essentially, tangermeme aims to implement everything except for the model that you'd like to use, including I/O, identifying matched region sets, altering sequences (e.g., inserting a motif or scrambling out a motif), running marginalization experiments, and annotating regions. These functions are meant to be used by themselves but also can easily be built on top of if you'd like to customize your analyses. 

Another way of looking at tangermeme is that, if the MEME suite is meant to do sequence analyses when you have nly biological sequences (or maybe priors derived from experimental data), tangermeme is meant to do sequence analyses when you have these sequences *and* a predictive machine learning model. How does motif discovery or annotation differ when you have attribution values highlighting nucleotides based on how important they are to the predictions? Accordingly, tangermeme implements several command-line tools that are similar to those in the MEME suite, such as FIMO/TOMTOM/MEME, but also extends the capabilities of these tools to handle attributions, and implements new methods that answer additional questions.

Installation
============

`pip install tangermeme`


Thank You
=========

No good project is done alone, and so I'd like to thank everyone who tested tangermeme, provided feedback, and contributed during the development process. 


Contributions
=============

Contributions are eagerly accepted! If you would like to contribute a feature then fork the master branch and be sure to run the tests before changing any code. Let us know what you want to do on the issue tracker just in case we're already working on an implementation of something similar. Also, please don't forget to add tests for any new functions. 

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   self
   whats_new.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Features

   tutorials/C_Feature_Tutorial_1_GPU_Usage.ipynb
   tutorials/C_Feature_Tutorial_2_Mixed_Precision_and_DataTypes.ipynb
   tutorials/C_Feature_Tutorial_3_Out_Of_Core_Learning.ipynb
   tutorials/C_Feature_Tutorial_4_Priors_and_Semi-supervised_Learning.ipynb

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Models

   tutorials/B_Model_Tutorial_1_Distributions.ipynb
   tutorials/B_Model_Tutorial_2_General_Mixture_Models.ipynb
   tutorials/B_Model_Tutorial_3_Bayes_Classifier.ipynb
   tutorials/B_Model_Tutorial_4_Hidden_Markov_Models.ipynb
   tutorials/B_Model_Tutorial_5_Markov_Chains.ipynb
   tutorials/B_Model_Tutorial_6_Bayesian_Networks.ipynb
   tutorials/B_Model_Tutorial_7_Factor_Graphs.ipynb
