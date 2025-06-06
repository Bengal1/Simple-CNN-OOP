#include "../../include/Layers/FullyConnected.hpp"

#include <random>
#include <stdexcept>

FullyConnected::FullyConnected(size_t inputSize, size_t outputSize, InitMethod method)
    : _inputSize(inputSize),
      _outputSize(outputSize),
      _optimizer(std::make_unique<Adam>(Adam::OptimizerMode::FullyConnected))
{
    if (_inputSize == 0)
        throw std::invalid_argument("[FullyConnected]: Input size must be greater than zero.");
    if (_outputSize == 0)
        throw std::invalid_argument("[FullyConnected]: Output size must be greater than zero.");
<<<<<<< HEAD
    
=======

>>>>>>> a5ddb32dc529fc62bbd81954f9f072e9b9c078ab
    _initializeWeights(method);
    _flatInput = Eigen::VectorXd::Zero(_inputSize);
}

void FullyConnected::updateParameters()
{
    _optimizer->updateStep(_weights, _weightsGradient);
    _optimizer->updateStep(_bias, _biasGradient);
    _weightsGradient.setZero();
    _biasGradient.setZero();
}

Eigen::MatrixXd FullyConnected::getWeights()
{
    return _weights;
}

Eigen::VectorXd FullyConnected::getBias()
{
    return _bias;
}

void FullyConnected::setParameters(const Eigen::MatrixXd& weights, const Eigen::VectorXd& bias)
{
    if (weights.rows() != _outputSize || weights.cols() != _inputSize)
        throw std::invalid_argument("[FullyConnected]: Weights size does not match.");
    if (bias.size() != _outputSize)
        throw std::invalid_argument("[FullyConnected]: Bias size does not match.");

    _weights = weights;
    _bias = bias;
}

<<<<<<< HEAD
/*void FullyConnected::_initializeWeights()
=======
void FullyConnected::_initializeWeights(InitMethod method)
>>>>>>> a5ddb32dc529fc62bbd81954f9f072e9b9c078ab
{
    _weights.resize(_outputSize, _inputSize);
    _bias.resize(_outputSize);

    std::random_device rd;
    std::mt19937 rng(rd());

    std::normal_distribution<double> dist;

    if (method == InitMethod::Xaviar)
    {
        double stddev = std::sqrt(1.0 / (_inputSize + _outputSize));
        dist = std::normal_distribution<double>(0, stddev);
    }
    else if (method == InitMethod::He)
    {
        double stddev = std::sqrt(2.0 / _inputSize);
        dist = std::normal_distribution<double>(0, stddev);
    }
    else if (method == InitMethod::Random)
    {
        dist = std::normal_distribution<double>(0, 1.0); // standard normal
    }
    else
    {
        throw std::invalid_argument("[FullyConnected]: Unknown initialization method.");
    }

    auto generator = [&]() { return dist(rng); };
    _weights = _weights.unaryExpr([&](double) { return generator(); });
    _bias.setZero();

    _weightsGradient = Eigen::MatrixXd::Zero(_outputSize, _inputSize);
    _biasGradient = Eigen::VectorXd::Zero(_outputSize);
}*/

void FullyConnected::_initializeWeights(InitMethod method)
{
    _weights.resize(_outputSize, _inputSize);
    _bias.resize(_outputSize);

    std::random_device rd;
    std::mt19937 rng(rd());

    std::normal_distribution<double> dist;

    if (method == InitMethod::Xaviar) {
        double stddev = std::sqrt(1.0 / (_inputSize + _outputSize));
        dist = std::normal_distribution<double>(0, stddev);
    }
    else if (method == InitMethod::He) {
        double stddev = std::sqrt(2.0 / _inputSize);
        dist = std::normal_distribution<double>(0, stddev);
    }
    else if (method == InitMethod::Random) {
        dist = std::normal_distribution<double>(0, 1.0);  // standard normal
    }
    else {
        throw std::invalid_argument("[FullyConnected]: Unknown initialization method.");
    }

    auto generator = [&]() { return dist(rng); };
    _weights = _weights.unaryExpr([&](double) { return generator(); });
    _bias.setZero();

    _weightsGradient = Eigen::MatrixXd::Zero(_outputSize, _inputSize);
    _biasGradient = Eigen::VectorXd::Zero(_outputSize);
}
