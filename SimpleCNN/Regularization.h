#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "Optimizer.h"


class BatchNormalization {
private:
    double _epsilon;
    bool _initialized;

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
    BatchNormalization(double epsilon = 1e-5)
        : _epsilon(epsilon), _initialized(false), 
        _optimizer(std::make_unique<AdamOptimizer>(-2)) {}

    void Forward(std::vector<Eigen::MatrixXd>& input, bool training = true) {
        if (!_initialized) {
            _InitializeParameters(input);
        }

        _input = input;

        for (int c = 0; c < _numChannels; ++c) {
            // Calculate mean and variance for each channel
            double channelMean = input[c].sum() / (_channelHeight * _channelWidth);
            double channelVariance = (input[c].array() - channelMean).square().sum() / (_channelHeight * _channelWidth - 1);

            if (training) {
                // Update running mean and variance using exponential moving average during training
                _runningMean[c] = 0.9 * _runningMean[c] + 0.1 * channelMean;
                _runningVariance[c] = 0.9 * _runningVariance[c] + 0.1 * channelVariance;
            }

            // Normalize the channel using running mean and variance during inference
            double meanToUse = training ? channelMean : _runningMean[c];
            double varianceToUse = training ? channelVariance : _runningVariance[c];

            // Scale and shift using gamma and beta
            input[c] = (_gamma[c] * (input[c].array() - meanToUse)) / (sqrt(varianceToUse) + _epsilon) + _beta[c];
        }
    }

    std::vector<Eigen::MatrixXd> Backward(std::vector<Eigen::MatrixXd>& dInput) {

        std::vector<Eigen::MatrixXd> inputGradient(_numChannels, Eigen::MatrixXd::Zero(_channelHeight, _channelWidth));

        for (int c = 0; c < _numChannels; ++c) {
            // Gradient w.r.t. normalized input
            Eigen::MatrixXd dNormalized = _gamma(c) * dInput[c].array();

            // Calculate meanDiff using the input from the forward pass
            Eigen::MatrixXd inputMeanDeviation = _input[c].array() - _runningMean[c];

            // Intermediate values
            double invStd = 1.0 / (sqrt(_runningVariance[c]) + _epsilon);

            // Gradient w.r.t. learnable gamma and beta
            _dGamma(c) = (dNormalized.array() * (_input[c].array() - _runningMean[c])).sum();
            _dBeta(c) = dNormalized.sum();

            // Gradient w.r.t. input
            Eigen::MatrixXd dInputNormalized = dNormalized * invStd;
            double dInputMean = dInputNormalized.sum();
            Eigen::MatrixXd varianceGradientScaling = (-0.5 * dInputNormalized.array() * inputMeanDeviation.array() * invStd * invStd * invStd).matrix();

            inputGradient[c] = dInputNormalized.array() - (dInputMean / (_channelHeight * _channelWidth)) + (inputMeanDeviation.array() * varianceGradientScaling.array());
        }
        return inputGradient;
    }

    void updateParameters() {
        _optimizer->updateStep(_gamma, _dGamma, 0);
        _optimizer->updateStep(_beta, _dBeta, 1);
        // Reset gradients
        _dGamma.setZero();
        _dBeta.setZero();
    }


private:
    void _InitializeParameters(const std::vector<Eigen::MatrixXd>& input) {
        _numChannels = input.size();
        _initialized = true;
        _channelHeight = input[0].rows();
        _channelWidth = input[0].cols();

        // Initialize running mean, variance, gamma, and beta for each channel
        _runningMean.resize(_numChannels, 0.0);
        _runningVariance.resize(_numChannels, 1.0);
        _gamma.resize(_numChannels, 1.0);
        _beta.resize(_numChannels, 0.0);
    }
};


class Dropout {

};