# SimpleCNN-OOP
This repository is an implementation of [*SimpleCNN*](https://github.com/Bengal1/Simple-CNN-Guide) in Object Oriented Programming (OOP) in C++ language. *SimpleCNN* is a repository I have written as practical guidance for codeing a simple Convolutional Neural Network using python (Pytorch).<br/>
In this project, I have built the same *SimpleCNN*, Convolutional Neural Network (CNN) in a lower level language, C++, and in a form of object oriented programming. The mechanism of the network can be seen more clear, because the CNN is written from scratch without any Machine learning library containing predefined models, structures or functions. <br/>
I advise you before diving in to this project, see [*SimpleCNN*](https://github.com/Bengal1/Simple-CNN-Guide).

## *MNIST Database*
This network is trained on MNIST database, a simple gray-scale images of a writen one-digit numbers (0-9), such that the network gets an image and it's target to classify it as the correct number (class).

<img src="https://user-images.githubusercontent.com/34989887/204675687-03f39aeb-1039-4abc-aec5-7f1d6cbbe52e.png" align="center" width="350" height="350"/>

The MNIST database has 70,000 images, such that the training dataset is 60,000 images and the test dataset is 10,000 images that is commonly used for training various image processing systems. The MNIST database of handwritten digits, it is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. The binary format of the dataset is also available for download on [Yann LeCun - THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/), and as it can be seen, those are the files that is being use in this repository.

For more imformation on [MNIST Dataset](https://en.wikipedia.org/wiki/MNIST_database).


## *Simple CNN*
Our Network is consist of 6 layers:
1. Convolution Layer with a kernel size of 5x5, and [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function.
2. Max-pool Layer with a kernel size of 2x2.
3. Convolution Layer with a kernel size of 5x5 and ReLU activation function.
4. Max-pool Layer with a kernel size of 2x2.
5. Fully-connected Layer with input layer of 1024 and output layer of 512 and ReLU activation function.
6. Fully-connected Layer with input layer of 512 and output layer of 10 (classes) and [Softmax](https://en.wikipedia.org/wiki/Softmax_function) activation function.

![simpleCNN](https://user-images.githubusercontent.com/34989887/206905433-34b42cbf-3ce3-4703-a575-d48f2cc95c09.png)

### The Model
The Simple CNN is implemented un C++ as Object Oriented Programming. In order to implement the network layers and methods [Eigen Library](https://eigen.tuxfamily.org/index.php?title=Main_Page) is being used. Eigen is a C++-based open-source linear algebra library. 

Every Layer apart of the fully connected can gets an input of 4-dimentions *(N,C,H,W)*, were *N* is the batch size, *C* is the number of the channels and *H,W* are height and width respectively, the resolution of the images.
In order to write the *Simple CNN* network  in C++ we constructed:

* ***Convolution2D*** - Applies a 2D convolution over an input signal composed of singular or several data inputs. See [Convolution2D.h](https://github.com/Bengal1/Simple-CNN-OOP/blob/main/Layers/Convolution2D.hpp).
* ***MaxPooling*** - Applies a 2D max pooling over an input signal composed of singular or several data inputs. See [MaxPooling.h](https://github.com/Bengal1/Simple-CNN-OOP/blob/main/Layers/MaxPooling.hpp).
* ***FullyConnected*** - Applies a linear transformation to the layer's input, *y=xA<sup>T</sup>+b*. In that case the input is 2-dimentions, *(N,H)* with the same notations above. See [FullyConnected.h](https://github.com/Bengal1/Simple-CNN-OOP/blob/main/Layers/FullyConnected.hpp).



**Dropout** - During training, randomly zeroes some of the elements of the input tensor with a given probability *p* using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.

**BatchNormalization** - Applies Batch Normalization over a 4D input, sclicing through *N* and computing statistics on *(N,H,W)* slices.


#### *Loss & Optimization*
* **Cross Entropy Loss** - This criterion computes the cross entropy loss between input logits and target. Loss function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "loss" associated with the event. The Cross Entropy Loss function is commonly used in classification tasks both in traditional ML and deep learning, and it also has its advantages. For more information on [Loss function](https://en.wikipedia.org/wiki/Loss_function) and [Cross Entropy Loss function](https://wandb.ai/sauravmaheshkar/cross-entropy/reports/What-Is-Cross-Entropy-Loss-A-Tutorial-With-Code--VmlldzoxMDA5NTMx). Also see [LossFunction.h](https://github.com/Bengal1/SimpleCNN-OOP/blob/master/SimpleCNN/LossFunction.h)

* **Adam optimizer** - The Adam optimization algorithm is an extension to stochastic gradient descent (SGD). Unlike SGD, The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients. For more information on [Stochastic gradient descent, extensions and variants](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). Also see [Optimizer.h](https://github.com/Bengal1/SimpleCNN-OOP/blob/master/SimpleCNN/Optimizer.h)

## Typical Run

## References
[The Back Propagation Method for CNN](https://ieeexplore.ieee.org/abstract/document/409626)

[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)


### TODO:
* initialization - He, Xavier, ...
* model in depth
* activation functions in depth
* loss function in depth
