/**
 * @file SimpleCNN.hpp
 * @brief Header file for the SimpleCNN class.
 *
 * This header defines the SimpleCNN class, a basic Convolutional Neural Network
 * implementation designed for image classification tasks, such as the MNIST dataset.
 * The class provides methods for training, testing, and managing the model's
 * state and data.
 */
#pragma once

#include "../include/Layers/Convolution2D.hpp"
#include "../include/Layers/FullyConnected.hpp"
#include "../include/Layers/MaxPooling.hpp"
#include "Activation/ReLU.hpp"
#include "Activation/Softmax.hpp"
#include "LossFunction/LossTypes.hpp"
#include "MNISTLoader.hpp"
#include "Regularization/BatchNormalization.hpp"
#include "Regularization/Dropout.hpp"

/**
 * @class SimpleCNN
 * @brief A simple Convolutional Neural Network implementation.
 *
 * The SimpleCNN class encapsulates a complete CNN architecture with convolutional,
 * pooling, and fully connected layers, along with helper methods for training,
 * testing, and data management. It uses a series of layers to process image data
 * and classify it into one of several predefined classes.
 */
class SimpleCNN
{
   private:
    // Default parameters
    static constexpr int MNISTClasses = 10;
    static constexpr int DefaultEpochs = 10;
    // Training statistics index
    enum TrainingStatIndex
    {
        TRAIN_LOSS = 0,
        TRAIN_ACCURACY,
        VALIDATION_LOSS,
        VALIDATION_ACCURACY,
        NUM_TRAIN_STATS
    };

    std::vector<std::vector<double>> _trainingStats;
    double _testAccuracy = 0.0;

    Convolution2D _conv1, _conv2;
    FullyConnected _fc1, _fc2;
    MaxPooling _pool1, _pool2;
    BatchNormalization _bn1, _bn2;
    Dropout _dropout1, _dropout2;

    ReLU<std::vector<Eigen::MatrixXd>> _relu1, _relu2;
    ReLU<Eigen::VectorXd> _relu3;
    Softmax<Eigen::VectorXd> _softmax;
    CrossEntropy CEloss;

    const size_t _classes;

   public:
    explicit SimpleCNN(size_t classes = MNISTClasses);

    void trainSimpleCNN(MNISTLoader& dataLoader, const size_t epochs = DefaultEpochs);

    void testSimpleCNN(MNISTLoader& dataLoader);

    std::vector<std::vector<double>> getLastTrainingStats() const;

    double getLastTestAccuracy() const;

    void exportTrainingDataToCSV(std::filesystem::path targetPath = "training_data.csv") const;

   private:
    Eigen::VectorXd _ForwardPass(const Eigen::MatrixXd& input);

    void _Backpropagation(const Eigen::VectorXd& softmaxCrossEntropyGradient);

    void _updateParameters();

    const double _trainEpoch(const std::vector<Eigen::MatrixXd>& trainImages,
                             const std::vector<Eigen::VectorXd>& oneHotTrainLabels,
                             std::vector<Eigen::VectorXd>& trainOutput);

    const double _validateEpoch(const std::vector<Eigen::MatrixXd>& validationImages,
                                const std::vector<Eigen::VectorXd>& oneHotValidationLabels,
                                std::vector<Eigen::VectorXd>& validationOutput);

    const double _accuracyCalculation(const std::vector<Eigen::VectorXd>& modelOutput,
                                      const std::vector<Eigen::VectorXd>& oneHotTargets);

    void _setTrainingMode(bool isTraining);

    void _initializeTrainingStats(size_t epochs);

    void _printEpoch(size_t epoch) const;

    void _printTrainingStats() const;
};
