/**
 * @file SimpleCNN.hpp
 * @brief Declaration of the SimpleCNN convolutional neural network.
 *
 * This header defines the SimpleCNN class, which encapsulates a complete
 * convolutional neural network architecture for image classification tasks.
 * The class exposes a minimal public interface for training, evaluation, and
 * metric export, while hiding all implementation details related to network
 * structure, optimization, and gradient propagation.
 *
 * The implementation is designed around grayscale image inputs (e.g., MNIST)
 * and follows a conventional CNN architecture consisting of convolutional,
 * pooling, normalization, and fully connected layers.
 *
 * @author Bengal1
 * @version 1.0
 * @date May 2025
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
 * @brief A compact convolutional neural network for image classification.
 *
 * SimpleCNN implements an end-to-end convolutional neural network with
 * integrated training, validation, and testing workflows. The network is
 * composed of two convolutional blocks followed by a fully connected
 * classifier and softmax output layer.
 *
 * Design principles:
 *  - Clear separation between public API and internal training mechanics
 *  - Explicit training vs. inference mode control
 *  - Modular layer composition
 *
 * The class is intended as a reference implementation and educational baseline,
 * while maintaining production-grade code structure and documentation quality.
 */
class SimpleCNN
{
   private:
    /**
     * @brief Default number of output classes for MNIST classification.
     */
    static constexpr int MNISTClasses = 10;
    
    /**
     * @brief Default number of training epochs.
     */
    static constexpr int DefaultEpochs = 10;
    
    /**
     * @brief Indices for accessing training statistics.
     *
     * Used internally to index per-epoch loss and accuracy metrics.
     */
    enum TrainingStatIndex
    {
        TRAIN_LOSS = 0,
        TRAIN_ACCURACY,
        VALIDATION_LOSS,
        VALIDATION_ACCURACY,
        NUM_TRAIN_STATS
    };

    /**
     * @brief Per-epoch training and validation metrics.
     *
     * Layout:
     *  - TRAIN_LOSS
     *  - TRAIN_ACCURACY
     *  - VALIDATION_LOSS
     *  - VALIDATION_ACCURACY
     */
    std::vector<std::vector<double>> _trainingStats;
    
    /**
     * @brief Test accuracy from the most recent evaluation run.
     */
    double _testAccuracy = 0.0;

    /* Network layers */
    Convolution2D _conv1, _conv2;
    FullyConnected _fc1, _fc2;
    MaxPooling _pool1, _pool2;
    
    /* Regularization */
    BatchNormalization _bn1, _bn2;
    Dropout _dropout1, _dropout2;

    /* Activation functions */
    ReLU<std::vector<Eigen::MatrixXd>> _relu1, _relu2;
    ReLU<Eigen::VectorXd> _relu3;
    Softmax<Eigen::VectorXd> _softmax;

    /* Loss function */
    CrossEntropy CEloss;

    /**
     * @brief Number of output classes.
     *
     * Immutable after construction.
     */
    const size_t _classes;

   public:
    
   /**
     * @brief Constructs a SimpleCNN instance.
     *
     * Initializes all network layers and validates the requested number of
     * output classes.
     *
     * @param classes Number of output classes.
     */
    explicit SimpleCNN(size_t classes = MNISTClasses);

    /**
     * @brief Trains the network for a specified number of epochs.
     *
     * Performs supervised training using data provided by the MNISTLoader,
     * including per-epoch validation and metric tracking.
     *
     * @param dataLoader Reference to an initialized MNISTLoader instance.
     * @param epochs Number of training epochs.
     */
    void trainSimpleCNN(MNISTLoader& dataLoader, 
                        const size_t epochs = DefaultEpochs);
    
    /**
     * @brief Evaluates the trained network on the test dataset.
     *
     * Computes and stores the final classification accuracy.
     *
     * @param dataLoader Reference to an initialized MNISTLoader instance.
     */
    void testSimpleCNN(MNISTLoader& dataLoader);
    
    /**
     * @brief Returns training and validation metrics from the last run.
     *
     * @return std::vector<std::vector<double>>
     *         Per-epoch training statistics.
     */
    std::vector<std::vector<double>> getLastTrainingStats() const;
    
    /**
     * @brief Returns the test accuracy from the last evaluation.
     *
     * @return double Test accuracy in percentage.
     */
    double getLastTestAccuracy() const;
    
    /**
     * @brief Exports training metrics to a CSV file.
     *
     * The output file contains loss and accuracy values for both training
     * and validation sets across all epochs.
     *
     * @param targetPath Output file path.
     */
    void exportTrainingDataToCSV(
        std::filesystem::path targetPath = "training_data.csv") const;

   private:
    /**
     * @brief Executes a forward pass through the network.
     *
     * @param input Input image.
     * @return Eigen::VectorXd Softmax-normalized class probabilities.
     */
    Eigen::VectorXd _ForwardPass(const Eigen::MatrixXd& input);
    
    /**
     * @brief Performs backpropagation through the network.
     *
     * @param softmaxCrossEntropyGradient Gradient of the loss with respect to
     *                                    the softmax output.
     */
    void _Backpropagation(const Eigen::VectorXd& softmaxCrossEntropyGradient);
    
    /**
     * @brief Applies optimizer updates to all trainable parameters.
     */
    void _updateParameters();
    
    /**
     * @brief Executes one training epoch.
     *
     * @return double Average training loss.
     */
    const double _trainEpoch(
        const std::vector<Eigen::MatrixXd>& trainImages,
        const std::vector<Eigen::VectorXd>& oneHotTrainLabels,
        std::vector<Eigen::VectorXd>& trainOutput);
    
    /**
     * @brief Executes one validation epoch.
     *
     * @return double Average validation loss.
     */
    const double _validateEpoch(
        const std::vector<Eigen::MatrixXd>& validationImages,
        const std::vector<Eigen::VectorXd>& oneHotValidationLabels,
        std::vector<Eigen::VectorXd>& validationOutput);
    
    /**
     * @brief Computes classification accuracy.
     *
     * @return double Accuracy in percentage.
     */
    const double _accuracyCalculation(
        const std::vector<Eigen::VectorXd>& modelOutput,
        const std::vector<Eigen::VectorXd>& oneHotTargets);
    
    /**
     * @brief Enables or disables training mode.
     *
     * Controls behavior of batch normalization and dropout layers.
     *
     * @param isTraining True for training mode, false for inference mode.
     */
    void _setTrainingMode(bool isTraining);
    
    /**
     * @brief Initializes internal storage for training statistics.
     */
    void _initializeTrainingStats(size_t epochs);
    
    /**
     * @brief Prints metrics for a single epoch.
     */
    void _printEpoch(size_t epoch) const;
    
    /**
     * @brief Prints all recorded training statistics.
     */
    void _printTrainingStats() const;
};
