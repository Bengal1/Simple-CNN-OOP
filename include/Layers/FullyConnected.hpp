#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "../Optimizer/Adam.hpp"

class FullyConnected
{
   public:
    FullyConnected(size_t inputSize, size_t outputSize, double maxGradNorm = -1.0,
                   double weightDecay = 0.0, size_t batchSize = 1);
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
    enum class InputType
    {
        Vector,
        Matrix,
        Tensor3D
    };

    const size_t _inputSize;
    const size_t _outputSize;
    const size_t _batchSize;

    size_t _inputChannels = 0;
    size_t _inputHeight = 0;
    size_t _inputWidth = 0;
    InputType _inputType;

    Eigen::MatrixXd _weights;
    Eigen::VectorXd _bias;
    Eigen::MatrixXd _weightsGradient;
    Eigen::VectorXd _biasGradient;
    Eigen::VectorXd _flatInput;

    std::unique_ptr<Optimizer> _optimizer;

    void _initializeWeights();

    template <typename T>
    void _getInputDimensions(const T& input);

    template <typename T>
    Eigen::VectorXd _flattenInput(const T& input);

    template <typename T>
    T _restoreInputShape(const Eigen::VectorXd& flat);
};

#include "FullyConnected.tpp"