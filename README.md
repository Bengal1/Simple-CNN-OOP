# Simple-CNN-OOP
This project is a C++ Object-Oriented Programming (OOP) version of my earlier SimpleCNN implementation in Python (PyTorch). While the original project served as a practical introduction to coding a simple CNN, this C++ version takes a lower-level approach: the entire network is built from scratch, without predefined ML frameworks or utilities.

By implementing each layer, operation, and optimization step manually, this project highlights the inner workings of convolutional networks—such as convolution, activation functions, pooling, normalization, and backpropagation. The OOP design ensures modularity, making it easier to extend, experiment with new components, or adapt the network to different tasks.

I advise you before diving in to this project, see [*SimpleCNN*](https://github.com/Bengal1/Simple-CNN-Guide).

## Requirements
- [![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
- [![Eigen](https://img.shields.io/badge/Eigen-3.4%2B-lightgrey?logo=eigen&logoColor=blue)](https://eigen.tuxfamily.org)


## *MNIST Database*
This network is trained on MNIST database, a simple gray-scale images of a writen one-digit numbers (0-9), such that the network gets an image and it's target to classify it as the correct number (class).

<img src="https://user-images.githubusercontent.com/34989887/204675687-03f39aeb-1039-4abc-aec5-7f1d6cbbe52e.png" align="center" width="350" height="350"/>

The MNIST database has 70,000 images, such that the training dataset is 60,000 images and the test dataset is 10,000 images that is commonly used for training various image processing systems. The MNIST database of handwritten digits, it is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. The binary format of the dataset is also available for download on [Yann LeCun - THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/), and available for download on [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=t10k-images.idx3-ubyte).

For more imformation on [MNIST Dataset](https://en.wikipedia.org/wiki/MNIST_database).


##  The Model - *Simple CNN*
Our Network is consist of 6 layers:
1. Convolution Layer with a kernel size of 5x5, and [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function.
2. Max-pool Layer with a kernel size of 2x2.
3. Convolution Layer with a kernel size of 5x5 and ReLU activation function.
4. Max-pool Layer with a kernel size of 2x2.
5. Fully-connected Layer with input layer of 1024 and output layer of 512 and ReLU activation function.
6. Fully-connected Layer with input layer of 512 and output layer of 10 (classes) and [Softmax](https://en.wikipedia.org/wiki/Softmax_function) activation function.

<img width="3967" height="1296" alt="simpleCNNarchitecture" src="https://github.com/user-attachments/assets/5d267466-c0c1-4bf6-8d65-87680788814e" />

The Simple CNN is implemented in C++ as Object Oriented Programming. In order to implement the network layers and methods [Eigen Library](https://eigen.tuxfamily.org/index.php?title=Main_Page) is being used. Eigen is a C++-based open-source linear algebra library. 

Every Layer apart of the fully connected can gets an input of 4-dimentions *(N,C,H,W)*, were *N* is the batch size, *C* is the number of the channels and *H,W* are height and width respectively, the resolution of the images. <br/>
Also see [SimpleCNN.hpp](https://github.com/Bengal1/Simple-CNN-OOP/blob/master/SimpleCNN/SimpleCNN.hpp).

## The Layers & Model's Components
### Convolution2D

<img align="right" width="370" alt="conv_cnn" src="https://github.com/user-attachments/assets/6ed279be-7253-49c4-9b9b-22e6a104c9f9" />

Applies a 2D convolution over an input signal composed of singular or several data inputs. See [Convolution2D.hpp](https://github.com/Bengal1/Simple-CNN-OOP/blob/main/Layers/Convolution2D.hpp).<br/>

The *Convolutional Layer* is the core building block of a Convolutional Neural Network (CNN), commonly used in image processing and computer vision. A convolutional layer applies a small filter (or kernel) across the input to extract features such as edges, textures, and shapes. <br/>
Given an input $`X`$ of size $`H×W`$ and a kernel $`K`$ of size $`k×k`$:

$$
Y(i,j)=\sum_m \sum_n X(i + m, j + n) \cdot K(m, n)
$$

Then, the output dimensions are:

$$
H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1 \quad;\quad W_{out} = \left\lfloor \frac{W + 2p - k}{s} \right\rfloor + 1
$$

Where:
* $`s`$ - stride, how far the filter moves at each step. A stride of 1 means one pixel at a time.
* $`p`$ - padding, adds zeros around the input to control the size of the output.

### MaxPooling

<img align="right" width="350" alt="max_pooling" src="https://github.com/user-attachments/assets/64d79582-196d-4832-a2a8-b46655abebdf" />

Applies a 2D max pooling over an input signal composed of singular or several data inputs. See [MaxPooling.hpp](https://github.com/Bengal1/Simple-CNN-OOP/blob/main/Layers/MaxPooling.hpp). <br/>
A pooling layer is used in Convolutional Neural Networks (CNNs) to reduce the spatial dimensions (height and width) of feature maps while preserving the most important information.
Max pooling is the most common type of pooling. It works by sliding a small window (like 2×2 or 3×3) over the input and taking the maximum value in each region. <br/>
For an input of size $`H×W`$ and a pooling window (kernel) of size $`k×k`$ on each step, the max pooling layer applies:

$$
Y(i, j) = \max_{\substack{(m, n) \in \text{window}}} X(i + m, j + n)
$$

Then, the output dimensions, like the convolutional layer, are:

$$
H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1 \quad;\quad W_{out} = \left\lfloor \frac{W + 2p - k}{s} \right\rfloor + 1
$$

### FullyConnected

<img align="right" width="350" alt="fc_cnn" src="https://github.com/user-attachments/assets/3cb0895f-58fc-45f5-b2b9-064d0603f0fd" />

Applies a linear transformation to the layer's input on a 2-dimentional input, $`(N,H)`$. See [FullyConnected.hpp](https://github.com/Bengal1/Simple-CNN-OOP/blob/main/Layers/FullyConnected.hpp). <br/>
A fully-connected layer is a type of neural network layer where every neuron is connected to every neuron in the previous layer. It is usually placed at the end of convolutional or recurrent layers to make final predictions. <br/>
Each output neuron computes a weighted sum of all inputs, adds a bias, and then applies an activation function.

$$
y = f\bigg(Wx+b\bigg)
$$

Where:
* $`x \in \mathbb{R}^{n_{\text{in}}}`$ is the input vector.
* $`W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}`$ is the weight matrix.
* $`b \in \mathbb{R}^{n_{\text{out}}}`$ is the bias vector.
* $`f(\cdot)`$ is an activation function (e.g., ReLU, sigmoid).
* $`y \in \mathbb{R}^{n_{\text{out}}}`$ is the output vector.



#### He Weight initialization 
Weight initialization helps neural networks train efficiently by keeping activations and gradients stable across layers. Xavier initialization works well with sigmoid or tanh activations, using a variance of $`\frac{2}{n_{in}+n_{out}​}`$​. He initialization is better for ReLU-based activations, using $`\frac{2}{n_{in}}`$​ to account for the fact that ReLU drops negative inputs. <br/>
**He Initialization** (also called Kaiming Initialization) is designed for neural networks that use ReLU or Leaky ReLU activations. Since ReLU sets all negative inputs to zero, it effectively reduces the number of active neurons by half, which can shrink the variance of outputs layer by layer if not handled properly.
To compensate, He Initialization sets the weights to have a higher variance:
* For a normal distribution: $$W \sim \mathcal{N}\Bigg(0,\quad \frac{2}{n_{in}}\Bigg)$$
  
* For a uniform distribution: $`W \sim \mathcal{U}\Bigg(-\sqrt{\frac{6}{n_{in}}},\quad \sqrt{\frac{6}{n_{in}}}\Bigg)`$

### Activation Functions
An activation function is a mathematical function applied to the output of each neuron in a neural network. It introduces non-linearity to the model, allowing it to learn complex patterns beyond just linear relationships. <br/>
Without activation functions, a neural network—no matter how deep—would behave like a simple linear model.
* **ReLU** (Rectified Linear Unit) is one of the most popular activation functions in deep learning.<br/>

```math
\mathrm{ReLU}(\mathbf{x}) = \mathbf{max}(0, \mathbf{x})
```

* **Softmax** is usually used in the output layer for classification, especially multi-class problems.<br/>

```math
\mathrm{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, 2, \ldots, K \quad ; \quad \mathbf{z} \in \mathbb{R}^{K}
```

### Regularization
#### Dropout
Dropout is a regularization technique used in neural networks to prevent overfitting by randomly "dropping out" (setting to zero) a fraction of the neurons during training. This forces the network to learn redundant representations and helps it generalize better on unseen data. with a given probability *p* using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.

$$
y = \begin{cases} 
\frac{x}{1 - p} & \text{with probability } 1 - p \\
0 & \text{with probability } p
\end{cases}
$$

Where $`p`$ is the probability of a neuron being dropped.

#### BatchNormalization
Batch Normalization (BatchNorm) is a  regularization technique used to improve the training of deep neural networks by normalizing the activations of each layer. It ensures that the output of each layer has a consistent distribution, reducing issues related to internal covariate shift, where the distribution of activations changes as the model trains. By doing so, BatchNorm can make the network train faster, stabilize training, and reduce the reliance on careful initialization or dropout.<br/>
The Batch Normalization process steps:
1. Compute Mean and Variance:

$$ \mu_B = \frac{1}{m} \sum_{i=1}^m x_i \quad;\quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$

2. Normalize the Input:

$$ \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} $$

&emsp;Where $`\epsilon`$ is a small constant to avoid division by zero.

3. Scale and Shift:

$$ y_i = \gamma \hat{x}_i + \beta $$

&emsp;Where $`\gamma`$ and $`\beta`$ are learned scale and shift parameters.

## *Loss & Optimization*
### Cross-Entropy Loss Function
This criterion computes the cross entropy loss between input logits and target. [Loss function](https://en.wikipedia.org/wiki/Loss_function) is a function that maps an event or values of one or more variables onto a real number intuitively representing some "loss" associated with the event. [The Cross Enthropy Loss function](https://wandb.ai/sauravmaheshkar/cross-entropy/reports/What-Is-Cross-Entropy-Loss-A-Tutorial-With-Code--VmlldzoxMDA5NTMx) is commonly used in classification tasks both in traditional ML and deep learning. It compares the predicted probability distribution over classes (logits) with the true class labels and penalizes incorrect or uncertain predictions. Also see [LossFunction.hpp](https://github.com/Bengal1/Simple-CNN-OOP/blob/main/LossFunction.hpp).

$$
Loss = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

Where:
* $`C`$  is the number of classes.
* $`y_i`$​  is the true probability for class *i* (usually 1 for the correct class and 0 for others).
* $`\hat{y}_i`$  is the predicted probability for class *i*.

### Adam Optimizer
The Adam optimization algorithm is an extension to stochastic gradient descent (SGD). Unlike SGD, The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients Adam combines the benefits of two other methods: momentum and RMSProp. For more information on [Stochastic gradient descent, extensions and variants](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). Also see [Optimizer.hpp](https://github.com/Bengal1/Simple-CNN-OOP/blob/main/Optimizer.hpp)

#### Adam Algorithm:
* $`\theta_t`$​ : parameters at time step *t*.
* $`\beta_1,\beta_2​`$: exponential decay rates for moments estimation.
* $`\alpha`$ : learning rate.
* $`\epsilon`$ : small constant to prevent division by zero.
* $`\lambda`$ : weight decay coefficient. <br/>

1. Compute gradients:

$$
g_t = \nabla_{\theta} J(\theta_t)
$$

2. Update first moment estimate (mean):

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

3. Update second moment estimate (uncentered variance):

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

4. Bias correction:

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad ; \quad \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

5. Update parameters:

$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

## Typical Run

![vscode_15epochs_lr5e-5](https://github.com/user-attachments/assets/b955aa68-a914-4a26-a91a-ee6471a10035)

## References
[The Back Propagation Method for CNN](https://ieeexplore.ieee.org/abstract/document/409626)

[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

