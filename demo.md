## Getting Started with the Demo

To use the demo, simply choose the model for which you want to simulate the training, and the dataset you want to train it on, using the two dropdown menus immediately above. Training starts as soon as you choose an option. For every dataset, we trained a simple 1-layer Neural Network, and a small Convolutional Neural Network that were taken from the official Tensorflow and Keras tutorials.

At the moment, we have the following datasets:
* __CIFAR10:__ 50,000 RGB images of size 32x32. It contains 10 common objects. [Link]()
* __MNIST:__ 60,000 grayscale images of size 28x28. It contains handwritten digits from 0 to 9. [Link]()
* __Fashion MNIST:__ 60,000 grayscale images of size 28x28. It contains 10 commonly found fashion items. [Link]()

## What does the app do?
For the majority of Deep Learning models, it is extremely helpful to keep track of the accuracy and loss as it is training. At the moment, the best application to do that is the Tensorboard, which is a collection of visualization tools (metrics plots, image examples, graph representation, weight histogram, etc.) useful to debug and monitor the training of your model.

_Dash's Live Model Training Viewer_ complements the Tensorboard by offering the following:
* __Small and Lightweight__: The viewer loads a small number of important visualization, so that it loads and runs quickly.
* __
