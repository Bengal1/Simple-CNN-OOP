#include "../../include/Regularization//BatchNormalization.hpp"

#include <cmath>
#include <stdexcept>

#include "../../include/Optimizer/Adam.hpp"

BatchNormalization::BatchNormalization(double momentum)
    : _momentum(momentum), _optimizer(std::make_unique<Adam>(Adam::OptimizerMode::BatchNormalization))
{
    if (_momentum <= 0.0 || _momentum > 1.0)
    {
        throw std::invalid_argument("[BatchNormalization]: Momentum must be in the range (0, 1].");
    }
}

std::vector<Eigen::MatrixXd> BatchNormalization::forward(const std::vector<Eigen::MatrixXd>& input)
{
    if (!_initialized)
    {
        _InitializeParameters(input);
    }

    if (input.empty())
    {
        throw std::invalid_argument("[BatchNormalization]: Input is empty.");
    }

    if (input.size() != _numChannels || input[0].rows() != _channelHeight ||
        input[0].cols() != _channelWidth)
    {
        throw std::invalid_argument("[BatchNormalization]: Input shape mismatch.");
    }

    std::vector<Eigen::MatrixXd> outputBN(_numChannels,
                                          Eigen::MatrixXd::Zero(_channelHeight, _channelWidth));
    _input = input;

    for (size_t c = 0; c < _numChannels; ++c)
    {
        _channelMean[c] = input[c].sum() / (_channelHeight * _channelWidth);
        _channelVariance[c] =
            (input[c].array() - _channelMean[c]).square().sum() / (_channelHeight * _channelWidth);

        if (_isTraining)
        {
            _runningMean[c] = (1.0 - _momentum) * _runningMean[c] + _momentum * _channelMean[c];
            _runningVariance[c] =
                (1.0 - _momentum) * _runningVariance[c] + _momentum * _channelVariance[c];
        }

        double mean = _isTraining ? _channelMean[c] : _runningMean[c];
        double var = _isTraining ? _channelVariance[c] : _runningVariance[c];

        outputBN[c] =
            _gamma[c] * ((input[c].array() - mean) / std::sqrt(var + _epsilon)) + _beta[c];
    }

    return outputBN;
}

std::vector<Eigen::MatrixXd> BatchNormalization::backward(
    const std::vector<Eigen::MatrixXd>& dOutput)
{
    if (dOutput.size() != _numChannels)
    {
        throw std::invalid_argument("[BatchNormalization]: Gradient size mismatch.");
    }

    std::vector<Eigen::MatrixXd> inputGradient(
        _numChannels, Eigen::MatrixXd::Zero(_channelHeight, _channelWidth));
    size_t N = _channelHeight * _channelWidth;

    for (size_t c = 0; c < _numChannels; ++c)
    {
        double mean = _channelMean[c];
        double var = _channelVariance[c];
        double invStd = 1.0 / std::sqrt(var + _epsilon);

        Eigen::MatrixXd x_hat = (_input[c].array() - mean) * invStd;
        _dGamma[c] = (dOutput[c].array() * x_hat.array()).sum();
        _dBeta[c] = dOutput[c].sum();

        Eigen::MatrixXd dXhat = dOutput[c].array() * _gamma[c];
        Eigen::MatrixXd x_mu = _input[c].array() - mean;

        double dVar = (dXhat.array() * x_mu.array() * -0.5 * std::pow(var + _epsilon, -1.5)).sum();
        double dMean = (dXhat.array() * -invStd).sum() + dVar * (-2.0 / N) * x_mu.sum();

        inputGradient[c] = dXhat.array() * invStd + dVar * 2.0 / N * x_mu.array() + dMean / N;
    }

    return inputGradient;
}

void BatchNormalization::updateParameters()
{
    _optimizer->updateStep(_gamma, _dGamma, 0);
    _optimizer->updateStep(_beta, _dBeta, 1);
    _dGamma.setZero();
    _dBeta.setZero();
}

void BatchNormalization::setTrainingMode(bool isTraining)
{
    _isTraining = isTraining;
}

void BatchNormalization::_InitializeParameters(const std::vector<Eigen::MatrixXd>& input)
{
    if (_optimizer == nullptr)
    {
        throw std::invalid_argument("[BatchNormalization]: Optimizer is not set.");
    }

    if (input.empty() || input[0].rows() == 0 || input[0].cols() == 0)
    {
        throw std::invalid_argument("[BatchNormalization]: Invalid input dimensions.");
    }

    _numChannels = input.size();
    _channelHeight = input[0].rows();
    _channelWidth = input[0].cols();
    _initialized = true;

    _channelMean.assign(_numChannels, 0.0);
    _channelVariance.assign(_numChannels, 0.0);
    _runningMean.assign(_numChannels, 0.0);
    _runningVariance.assign(_numChannels, 1.0);

    _gamma = Eigen::VectorXd::Constant(_numChannels, 1.0);
    _beta = Eigen::VectorXd::Constant(_numChannels, 0.0);
    _dGamma = Eigen::VectorXd::Zero(_numChannels);
    _dBeta = Eigen::VectorXd::Zero(_numChannels);
}
