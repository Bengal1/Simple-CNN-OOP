/**
 * @file FullyConnected.hpp
 * @brief Header file for the FullyConnected layer.
 *
 * This header defines the FullyConnected class, which implements a fully
 * connected (dense) neural network layer. It handles the forward and backward
 * passes, parameter initialization, and updates.
 */
#pragma once

#include <Eigen/Dense>
#include <memory>

#include "../Optimizer/Adam.hpp"

/**
 * @class FullyConnected
 * @brief Implements a fully connected (dense) neural network layer.
 *
 * This class provides the functionality of a fully connected layer, including
 * handling different input shapes, forward and backward passes, and
 * parameter updates.
 */
class FullyConnected
{
   public:
    enum class InitMethod
    {
        He,     ///< He initialization.
        Xaviar, ///< Xaviar initialization.
        Random  ///< Simple random initialization.
    };

   private:
    // layer dimensions
    const size_t _inputSize;
    const size_t _outputSize;
    // input dimensions
    size_t _inputChannels = 0;
    size_t _inputHeight = 0;
    size_t _inputWidth = 0;
    // learnable parameters
    Eigen::MatrixXd _weights;
    Eigen::VectorXd _bias;
    // gradients
    Eigen::MatrixXd _weightsGradient;
    Eigen::VectorXd _biasGradient;
    // input data
    Eigen::VectorXd _flatInput;
    // Optimizer
    std::unique_ptr<Optimizer> _optimizer;

   public:
    FullyConnected(size_t inputSize, size_t outputSize, InitMethod method = InitMethod::He);
    ~FullyConnected() = default;

    template <typename T>
    Eigen::VectorXd forward(const T& input);

    template <typename T>
    T backward(const Eigen::VectorXd& dLoss_dOutput);

    void updateParameters();

    Eigen::MatrixXd getWeights();
    Eigen::VectorXd getBias();
    void setParameters(const Eigen::MatrixXd& weights, const Eigen::VectorXd& bias);

   private:
    void _initializeWeights(InitMethod method);

    template <typename T>
    void _getInputDimensions(const T& input);

    template <typename T>
    Eigen::VectorXd _flattenInput(const T& input);

    template <typename T>
    T _restoreInputShape(const Eigen::VectorXd& flat);
};

#include "FullyConnected.tpp"