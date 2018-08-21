# ShapeNet

A VGG-style convolutional nueral network written in pure python. First step in a bigger project involving the Baby AI question / answer dataset. 

## Training

To begin training the model and watching the loss fall, simply download the source and run the babyAImodel.py file (through terminal). This will print out the loss and performance benchmarks periodicly. 

## About the model

The model I've decided on contains 3 sets of conv3-conv-3-maxpool2 layers for feature extraction followed by 2 dense layers for classification. The padding and kernel size of the conv layers are VGG-style, as to retain as much important spatial data as possible, as well as keep the model simple. The conv and dense layer activations are ReLU and TANH, respectivly. An adam optimizer is used with standard hyperparameters, as it performs the best out of other tested optimizers. The model also uses a batch size of 20 and learning rate of 0.01. 

## Performance

While a beautiful language to play around with Machine Learning, Python isn't very fast. To attempt to counteract this, I've implemented im2col for forward and backprop. This reduced the time of a single training pass from about 1.4 seconds to 0.8. As I learn more about threading and improving performance, I intend on coming back to update the model. 
