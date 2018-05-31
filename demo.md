## Getting Started with the Demo

To use the demo, simply choose the model for which you want to simulate the training, and the dataset you want to train it on, using the two dropdown menus immediately above. Training starts as soon as you choose an option. For every dataset, we trained a simple 1-layer Neural Network, and a small Convolutional Neural Network that were taken from the official Tensorflow and Keras tutorials.

At the moment, we have the following datasets:
* __CIFAR10:__ 50,000 RGB images of size 32x32. It contains 10 common objects. [Link]()
* __MNIST:__ 60,000 grayscale images of size 28x28. It contains handwritten digits from 0 to 9. [Link]()
* __Fashion MNIST:__ 60,000 grayscale images of size 28x28. It contains 10 commonly found fashion items. [Link]()

## What does the app do?
For the majority of Deep Learning models, it is extremely helpful to keep track of the accuracy and loss as it is training. At the moment, the best application to do that is the Tensorboard, which is a collection of visualization tools (metrics plots, image examples, graph representation, weight histogram, etc.) useful to debug and monitor the training of your model.

_Dash's Live Model Training Viewer_ is a compact visualization app that monitors core metrics of your __Tensorflow model__ during training. It complements the Tensorboard by offering the following:
* __Real-time visualization__: The app is designed to visualize your metrics as they are updated inside your model.
* __Small and Lightweight__: The viewer loads a small number of important visualization, so that it loads and runs quickly.
* __Simple to use__: For simpler tensorflow models, all you need to do is to call `add_eval` to add the accuracy and cross entropy operations in the graph, and generate a log of the metrics using `write_data`. Both functions are inside `tfutils.py`, and examples are included in the `examples` directory.
* __Easy to modify__: The app is stored inside one module, and is written in under 400 lines. You can quickly modify and improve the app without breaking anything.
* __Plotly Graphs and Dash Integration__: Easily integrate the app into more complex Dash Apps, and includes all the tools found in Plotly graphs.

At the moment, the logging only works for iterative Tensorflow models. We are planning to extend it for PyTorch. You are encouraged to port the logging function (which is a simple csv logging) to Keras, Tensorflow's eval() function, MXNet, etc.
