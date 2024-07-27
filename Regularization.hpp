#ifndef REGULARIZATION_HPP
#define REGULARIZATION_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <Eigen/Dense>
#include "Optimizer.hpp"


class BatchNormalization {
private:
    double _epsilon;
    bool _initialized;
    bool _isTraining;

    int _numChannels;
    int _channelHeight;
    int _channelWidth;

    std::vector<double> _runningMean;
    std::vector<double> _runningVariance;

    Eigen::VectorXd _gamma;
    Eigen::VectorXd _beta;
    Eigen::VectorXd _dGamma;
    Eigen::VectorXd _dBeta;
    std::vector<Eigen::MatrixXd> _input;

    std::unique_ptr<AdamOptimizer> _optimizer;

public:
    BatchNormalization()
        : _epsilon(1e-8), _initialized(false), _isTraining(true),
        _numChannels(0), _channelHeight(0), _channelWidth(0),
        _optimizer(std::make_unique<AdamOptimizer>(-2)) {}

    std::vector<Eigen::MatrixXd> forward(std::vector<Eigen::MatrixXd>&
        input) {

        if (!_initialized) {
            _InitializeParameters(input);
        }
        std::vector<Eigen::MatrixXd> outputBN(_numChannels,
            Eigen::MatrixXd::Zero(_channelHeight, _channelWidth));

        _input = input;

        for (int c = 0; c < _numChannels; ++c) {
            // Calculate mean and variance for each channel
            double channelMean = input[c].sum() / (_channelHeight *
                _channelWidth);
            double channelVariance = (input[c].array() - channelMean).
                square().sum() / (_channelHeight * _channelWidth);

            if (_isTraining) {
                // Update running mean and variance - exponential moving average
                _runningMean[c] = 0.9 * _runningMean[c] + 0.1 * channelMean;
                _runningVariance[c] = 0.9 * _runningVariance[c] + 0.1 *
                    channelVariance;
            }

            // Normalizing the channel 
            double meanToUse = _isTraining ? channelMean : _runningMean[c];
            double varianceToUse = _isTraining ? channelVariance :
                _runningVariance[c];

            // Scale and shift using gamma and beta
            outputBN[c] = _gamma[c] * ((input[c].array() - meanToUse) /
                (std::sqrt(varianceToUse) + _epsilon)) + _beta[c];
        }

        return outputBN;
    }

    std::vector<Eigen::MatrixXd> backward(std::vector<Eigen::MatrixXd>&
        dInput) {

        std::vector<Eigen::MatrixXd> inputGradient(_numChannels,
            Eigen::MatrixXd::Zero(_channelHeight, _channelWidth));

        for (int c = 0; c < _numChannels; ++c) {
            // Gradient w.r.t. normalized input
            Eigen::MatrixXd dNormalized = _gamma(c) * dInput[c].array();

            // Calculate meanDiff using the input from the forward pass
            Eigen::MatrixXd inputMeanDeviation = _input[c].array() -
                _runningMean[c];

            // Intermediate values
            double invStd = 1.0 / (sqrt(_runningVariance[c]) + _epsilon);

            // Gradient w.r.t. learnable gamma and beta
            _dGamma(c) = (dNormalized.array() * (_input[c].array() -
                _runningMean[c])).sum();
            _dBeta(c) = dNormalized.sum();

            // Gradient w.r.t. input
            Eigen::MatrixXd dInputNormalized = dNormalized * invStd;
            double dInputMean = dInputNormalized.sum();
            Eigen::MatrixXd varianceGradientScaling = (-0.5 *
                dInputNormalized.array() * inputMeanDeviation.array() *
                invStd * invStd * invStd).matrix();

            inputGradient[c] = dInputNormalized.array() - (dInputMean /
                (_channelHeight * _channelWidth)) + (inputMeanDeviation.
                    array() * varianceGradientScaling.array());
        }
        return inputGradient;
    }

    void updateParameters() {
        // Update learnable parameters 
        _optimizer->updateStep(_gamma, _dGamma, 0);
        _optimizer->updateStep(_beta, _dBeta, 1);
        // Reset gradients
        _dGamma.setZero();
        _dBeta.setZero();
    }

    void SetTestMode() {
        _isTraining = false;
        _dGamma.resize(0);
        _dBeta.resize(0);
        //_input.clear();
    }

    void SetTrainingMode() {
        _isTraining = true;
        _dGamma.setConstant(_numChannels, 0.0);
        _dBeta.setConstant(_numChannels, 0.0);
    }

private:
    void _InitializeParameters(const std::vector<Eigen::MatrixXd>& input) {
        //Initialize variables
        _numChannels = input.size();
        _initialized = true;
        _channelHeight = input[0].rows();
        _channelWidth = input[0].cols();
        //Initialize parameters
        _runningMean.assign(_numChannels, 0.0);
        _runningVariance.assign(_numChannels, 1.0);
        _gamma.setConstant(_numChannels, 1.0);
        _beta.setConstant(_numChannels, 0.0);
        _dGamma.setConstant(_numChannels, 0.0);
        _dBeta.setConstant(_numChannels, 0.0);
    }
};

class Dropout {
private:
    int _inputHeight;
    int _inputWidth;
    int _numChannels;
    double _dropoutRate;
    bool _isTraining;

public:
    Dropout(double dropoutRate = 0.5)
        : _dropoutRate(dropoutRate), _isTraining(true), _inputHeight(0),
        _inputWidth(0), _numChannels(0) {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) {

        if (!_inputHeight) {
            _inputHeight = input.rows();
            _inputWidth = input.cols();
        }
        if (!_isTraining || _dropoutRate == 0.0) {
            // No dropout
            return input;
        }

        // Create a random mask with values between -1 and 1
        Eigen::MatrixXd dropoutMask = CreateRandomMask();
        double randomThreshold = 2 * _dropoutRate - 1;

        // Apply dropout
        dropoutMask = (dropoutMask.array() > randomThreshold).cast<double>();
        return input.array() * dropoutMask.array() / (1.0 - _dropoutRate);
    }

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) {

        if (!_numChannels) {
            _numChannels = input.size();
            _inputHeight = input[0].rows();
            _inputWidth = input[0].cols();
        }
        if (!_isTraining || _dropoutRate == 0.0) {
            // No dropout
            return input;
        }
        std::vector<Eigen::MatrixXd> dropedOutInput(_numChannels,
            Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
        for (int c = 0; c < _numChannels; ++c) {
            // Create a random mask with values between -1 and 1
            Eigen::MatrixXd dropoutMask = CreateRandomMask();
            double randomThreshold = 2 * _dropoutRate - 1;

            // Apply dropout
            dropoutMask = (dropoutMask.array() > randomThreshold).cast<double>();
            dropedOutInput[c] = input[c].array() * dropoutMask.array() / (1.0 -
                _dropoutRate);
        }
        return dropedOutInput;
    }

    void SetTestMode()
    {
        _isTraining = false;
    }

    void SetTrainingMode()
    {
        _isTraining = true;
    }

private:
    Eigen::MatrixXd CreateRandomMask() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        Eigen::MatrixXd randomMask = Eigen::MatrixXd::NullaryExpr(_inputHeight,
            _inputWidth, [&dist, &gen]() { return dist(gen); });

        return randomMask;
    }
};

#endif // REGULARIZATION_HPP