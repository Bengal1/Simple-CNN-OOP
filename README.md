# Simple-CNN-OOP

A from-scratch implementation of a Convolutional Neural Network in modern C++ using Object-Oriented Programming.

## 📌 Overview
<img align="right" width="355"  alt="CNN_conv" src="https://github.com/user-attachments/assets/7ff34252-ae2e-4575-846c-28a7881c6e0b" />

Simple-CNN-OOP implements a Convolutional Neural Network for MNIST digit classification without using machine learning frameworks such as PyTorch or TensorFlow.

The project focuses on building the main CNN components manually and organizing them through a clear object-oriented design. This includes convolution layers, activation functions, pooling, fully connected layers, loss calculation, optimizers, regularization, forward propagation, and backpropagation.

The goal of this project is to better understand how convolutional neural networks work internally and how their components can be structured in C++ as modular, reusable classes.

This project is based on my earlier Simple CNN Guide project in Python/PyTorch. While that project demonstrates the high-level CNN training pipeline using a machine learning framework, this version takes a lower-level approach by implementing the network logic from scratch.

For readers who are new to CNNs and deep learning, I recommend starting with my earlier project [*Simple CNN Guide*](https://github.com/Bengal1/Simple-CNN-Guide). It focuses more on intuition and CNN fundamentals before diving into lower-level implementation details.

## 🛠️ Build and Run

### ✅ Requirements

- C++17 or later
- CMake or Make
- Eigen 3.4
- MNIST IDX dataset files

### 📁 MNIST Dataset Files

Place the MNIST files in the project’s dataset directory:

```text
MNIST/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```
> **Note:** The original MNIST binary files can be downloaded from [Yann LeCun - THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/). MNIST is also available through [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=t10k-images.idx3-ubyte).

### ⚙️ Build

This project can be built using either **CMake** or the included **Makefile**.

Using CMake:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Using Makefile:

```bash
make
```

If Eigen is installed in a different location, pass its include path when using the Makefile:

```bash
make EIGEN_PATH=/path/to/eigen
```

### ▶️ Run

After building the project with CMake, run the generated executable from the `build/` directory:

```bash
./SimpleCNN
```

On Windows:

```bash
SimpleCNN.exe
```

If you are using the provided Makefile, you can run the project with:

```bash
make run
```


## 📊 Results and Evaluation

The model was trained and evaluated on the MNIST handwritten digit classification task.

The purpose of this project is not to outperform optimized machine learning frameworks, but to validate that the from-scratch C++ implementation can successfully train and generalize on a standard image classification task.

### ⚙️ Training Pipeline

The training pipeline includes:

- Loading the MNIST IDX files
- Splitting the data into training and validation sets
- Running forward propagation through the CNN
- Computing the loss using Cross-Entropy
- Running backpropagation manually
- Updating parameters using an optimizer
- Evaluating the model on validation and test data
- Exporting training metrics for analysis

### 🔁 Python/PyTorch vs C++ OOP

This project is based on my earlier **Simple CNN Guide** project, which implements a similar CNN using Python and PyTorch.

| Project | Implementation | Epochs | Test Accuracy |
|---|---|---:|---:|
| Simple CNN Guide | Python / PyTorch | 10 | 99.31% |
| Simple-CNN-OOP | C++ / From Scratch | 15 | 98.55% |

The Python/PyTorch version achieves slightly higher accuracy, as expected from an optimized deep learning framework with automatic differentiation and highly optimized training utilities.

The C++ OOP version reaches **98.55% test accuracy** while implementing the CNN pipeline manually, including forward propagation, backpropagation, loss calculation, optimizer updates, and regularization components.

<img width="2872" height="1058" alt="image" src="https://github.com/user-attachments/assets/6a335c55-6b30-4e05-8d90-246463d71161" />


### 🎯 Key Takeaway

The main result of this project is not only the final accuracy, but the successful implementation of a complete CNN training pipeline from scratch in C++.

The comparison shows that the manually implemented C++ model can achieve strong MNIST performance while exposing the internal mechanics that PyTorch normally abstracts away.


## 🔨 Building the CNN, Not Just Using It

Many CNN projects focus on using existing frameworks to train a model. This project focuses on implementing the model and its training pipeline from the ground up in modern C++.

Instead of relying on predefined PyTorch or TensorFlow layers, the project manually implements the core CNN components and organizes them through an object-oriented design. Each major part of the system is separated into clear, reusable components, including layers, activations, loss functions, optimizers, regularization methods, and training logic.

MNIST is used as a controlled classification task, but the main value of the project is the implementation itself: translating neural network concepts into structured, maintainable C++ code.

This project demonstrates:

* Object-Oriented Programming in C++
* Modular software design
* Manual implementation of CNN components
* Forward propagation and backpropagation logic
* Loss calculation and optimizer-based parameter updates
* Integration of Dropout and Batch Normalization
* A complete training, validation, and testing workflow


## 🏗️ Project Architecture

The model is implemented as a compact Convolutional Neural Network designed for MNIST digit classification.

The architecture follows a standard image-classification pipeline: convolutional layers extract spatial features from the input image, pooling layers reduce the spatial resolution of the feature maps, and fully connected layers use the extracted representation to produce the final class prediction.

<img width="3967" height="1296" alt="simpleCNNarchitecture" src="https://github.com/user-attachments/assets/5d267466-c0c1-4bf6-8d65-87680788814e" />

### Model Breakdown

This architecture uses MNIST as a controlled environment for demonstrating the core CNN pipeline.

The convolutional part of the model acts as a feature extractor. It gradually transforms the input image from raw pixel values into a set of learned feature maps. The first convolution layer captures simple local patterns, while the second convolution layer builds on them to form more useful digit-level representations.

Pooling is used after each convolution block to reduce spatial resolution and keep the representation compact. This helps the later fully connected layers operate on a smaller, more focused feature representation instead of the full image.

After feature extraction, the model flattens the feature maps into a vector and passes it through fully connected layers. At this point, the network is no longer working with spatial image regions, but with a learned representation of the digit.

The final output contains 10 class scores, one for each MNIST digit. Softmax converts these scores into probabilities for prediction.

### Design Goal

The architecture is intentionally compact. It is large enough to demonstrate the main CNN building blocks, but simple enough to keep the implementation readable and traceable.

The focus of this project is not on reaching state-of-the-art MNIST accuracy, but on showing how the CNN pipeline can be implemented clearly from scratch in C++.


## 🧱 Object-Oriented Implementation

The CNN is implemented using an object-oriented structure, where each major part of the network is represented as a separate component with a clear responsibility.

Instead of writing the model as one large procedural script, the project separates the implementation into modules such as layers, activation functions, loss functions, optimizers, regularization components, and data loading.

This structure makes the code easier to read, debug, extend, and reason about.

### Core Components

```text
SimpleCNN
├── Layers
│   ├── Convolution2D
│   ├── MaxPooling
│   └── FullyConnected
│
├── Activations
│   ├── ReLU
│   └── Softmax
│
├── Loss
│   └── CrossEntropy
│
├── Optimizers
│   ├── SGD
│   └── Adam
│
├── Regularization
│   ├── Dropout
│   └── BatchNormalization
│
└── Data
    └── MNISTLoader
```

Each component is responsible for a specific part of the CNN pipeline:

* Layers perform the main neural network computations.
* Activations apply non-linear transformations.
* Loss functions measure prediction error.
* Optimizers update trainable parameters.
* Regularization components help improve training behavior.
* Data loading handles the MNIST IDX files.

This design allows individual components to be developed, tested, and extended independently. For example, a new optimizer, activation function, or layer can be added without rewriting the full model pipeline.


## 📚 References
- [The Back Propagation Method for CNN](https://ieeexplore.ieee.org/abstract/document/409626)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)


## 📄 License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for more details.
