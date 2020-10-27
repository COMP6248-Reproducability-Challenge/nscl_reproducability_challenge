# THE NEURO-SYMBOLIC CONCEPT LEARNER

This repository contains the code and reproducibility [report](https://github.com/markvasin/nscl_reproducability_challenge/blob/master/NSCL_reproducability_challenge.pdf) for the [ICLR](https://iclr.cc/) 2019 paper - [The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision](https://openreview.net/forum?id=rJgMlhRctm). This was done as part of the COMP6248 Reproducibility Challenge, a group coursework from Deep Learning class at University of Southampton.

## Paper Summary

[Mao et al.](https://openreview.net/forum?id=rJgMlhRctm) proposed Neuro-Symbolic Concept Learner, inspired by human learning process, to learn visual concepts and accurately answer questions on the [CLEVR dataset](https://cs.stanford.edu/people/jcjohns/clevr/). Their model achieved remarkable accuracy in visual concept learning, visual question answering, and generalised well to images with more objects and more complex questions. The model is composed of three parts: a visual module to extract features from the scene, a semantic parser to translate questions into programs, and a symbolic reasoning module to execute the program to obtain an answer and bridge the learning between a visual module and a semantic parser. The training strategy is also motivated by human learning. The author employed curriculum learning strategy to help with optimisation, where the questions and scenes are presented to the model with increasing difficulty.

![framework](Images/framework.png)

## Experiments

**Concept classification experiment**
The objective of this experiment is to evaluate whether the model learns generic concept representation. The point is that accurate and generic representation of object properties would then be useful for various applications, including image captioning and retrieval. In this experiment, we evaluated the classification accuracy for all attributes (colour, material, shape, size) using the validation set. In the original paper, they reported 99\% accuracy for all object properties, however we were able to achieve around 99\% accuracy for all attributes except the shape attribute which is around 81.5\% accuracy. 

**Concept count experiment**
Next, we performed the experiment to evaluate the accuracy of the count concept, which has been known to be difficult in deep learning due to the discrete nature of the problem. For this experiment, we evaluated our model on the validation set by generating 5 synthetic counting questions for each image. The questions generated are of the following form: "How many \texttt{Cube} objects are there?". The accuracy for all attributes except shape are quite close to what is reported in the original paper. 

## Library
- Python 3
- PyTorch
- NumPy

## Usage
1. Run download.sh to download all the necessary data
2. To train the model, run nscl_trainer.py
3. To evaluate the model on the validation set, run nscl_evaluator.py
4. To run concept classification experiment, use concept_classification_experiment.py
5. To run concept COUNT experiment, use concept_count_experiment.py
