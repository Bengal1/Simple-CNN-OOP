#pragma once

#include <vector>
#include <memory>
#include <random>
#include <iostream>
#include <Eigen/Dense>
#include "Optimizer.hpp"



class BatchNormalization {
private:
    size_t _numChannels;
    size_t _channelHeight;
    size_t _channelWidth;
    
    bool _initialized;
    bool _isTraining;

    double _epsilon;
    double _momentum;

    std::vector<double> _batchMean;
    std::vector<double> _batchVariance;
    std::vector<double> _runningMean;
    std::vector<double> _runningVariance;
    
    Eigen::VectorXd _gamma;
    Eigen::VectorXd _beta;
    Eigen::VectorXd _dGamma;
    Eigen::VectorXd _dBeta;
    std::vector<Eigen::MatrixXd> _input; 

    std::unique_ptr<AdamOptimizer> _optimizer;

public:
    BatchNormalization(double momentum = 0.1)
		: _numChannels(0), 
          _channelHeight(0), 
          _channelWidth(0),
          _epsilon(1e-8), 
          _initialized(false), 
          _isTraining(true), 
          _momentum(momentum),
          _optimizer(std::make_unique<AdamOptimizer>(-2))
    {
        if (_momentum < 0.0 || _momentum > 1.0) {
            throw std::invalid_argument("Momentum must be in the range [0, 1].");
        }
    }

    std::vector<Eigen::MatrixXd> forward(std::vector<Eigen::MatrixXd>& input) 
    {
        if (!_initialized) {
            _InitializeParameters(input);
        }
        std::vector<Eigen::MatrixXd> outputBN(_numChannels, 
            Eigen::MatrixXd::Zero(_channelHeight, _channelWidth));

        _input = input;
        
        for (size_t c = 0; c < _numChannels; ++c) {
            // Calculate mean and variance for each channel
            _batchMean[c] = input[c].sum() / (_channelHeight *
                            _channelWidth);
            _batchVariance[c] = (input[c].array() - _batchMean[c]).
                    square().sum() / (_channelHeight * _channelWidth);

            if (_isTraining) {
                // Update running mean and variance - exponential moving average
                _runningMean[c] = (1.0 - _momentum) * _runningMean[c] + _momentum * _batchMean[c];
                _runningVariance[c] = (1.0 - _momentum) * _runningVariance[c] + _momentum *
                                      _batchVariance[c];
            }
            
            // Normalizing the channel 
            double meanToUse = _isTraining ? _batchMean[c] : _runningMean[c];
            double varianceToUse = _isTraining ? _batchVariance[c] : _runningVariance[c];

            // Scale and shift using gamma and beta
            outputBN[c] = _gamma[c] * ((input[c].array() - meanToUse) / 
                          std::sqrt(varianceToUse + _epsilon)) + _beta[c];
        }
        return outputBN;
    }

    std::vector<Eigen::MatrixXd> backward(std::vector<Eigen::MatrixXd>& dInput) 
    {
        std::vector<Eigen::MatrixXd> inputGradient(_numChannels, 
                    Eigen::MatrixXd::Zero(_channelHeight, _channelWidth));
        size_t N = _channelHeight * _channelWidth;

        for (size_t c = 0; c < _numChannels; ++c) {
            double mean = _batchMean[c];
            double variance = _batchVariance[c];
            double invStd = 1.0 / (sqrt(variance) + _epsilon);

            // Gradients w.r.t gamma and beta
            Eigen::MatrixXd x_hat = (_input[c].array() - mean) * invStd;
            _dGamma(c) = (dInput[c].array() * x_hat.array()).sum();
            _dBeta(c) = dInput[c].sum();

            // Gradient w.r.t x_hat
            Eigen::MatrixXd dXhat = dInput[c].array() * _gamma(c);

            // Gradient w.r.t variance
            Eigen::MatrixXd x_mu = _input[c].array() - mean;
            double dVar = (dXhat.array() * x_mu.array() * (-0.5) * 
                           pow(variance + _epsilon, -1.5)).sum();

            // Gradient w.r.t mean
            double dMean = (dXhat.array() * (-invStd)).sum() + dVar * (-2.0 / N) * x_mu.sum();

            // Gradient w.r.t input
            inputGradient[c] = (dXhat.array() * invStd)
                             + (dVar * 2.0 / N * x_mu.array())
                             + (dMean / N);
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

private:
    void _InitializeParameters(const std::vector<Eigen::MatrixXd>& input) 
    {
        //Initialize variables
        _numChannels = input.size();
        _initialized = true;
        _channelHeight = input[0].rows();
        _channelWidth = input[0].cols();
        //Initialize parameters
        _batchMean.assign(_numChannels, 0.0);
		_batchVariance.assign(_numChannels, 0.0);
        _runningMean.assign(_numChannels, 0.0);
        _runningVariance.assign(_numChannels, 1.0);
        _gamma.setConstant(_numChannels, 1.0);
        _beta.setConstant(_numChannels, 0.0);
        _dGamma.setConstant(_numChannels, 0.0);
        _dBeta.setConstant(_numChannels, 0.0);
    }
};


class GradientNormClipping {
private:
    size_t _numChannels;
    double _maxValue;
    bool _isTraining;

public:
    GradientNormClipping(double maxValue = 1.5, bool isTraining = true) 
        : _maxValue(maxValue),  
          _numChannels(0), 
          _isTraining(isTraining)
    {
		if (_maxValue < 0.0) {
			throw std::invalid_argument("Max value must be non-negative.");
		}
    }


    Eigen::MatrixXd ClipGradient(const Eigen::MatrixXd& gradient)
    { // Matrix gradient

        if (!_isTraining || _maxValue == 0.0) {
            // No clipping
            return gradient;
        }

        double gradientNorm = gradient.norm();
        Eigen::MatrixXd clippedGradient = gradient * (_maxValue / gradientNorm);

        return clippedGradient;
    }

    std::vector<Eigen::MatrixXd> ClipGradient(const std::vector<Eigen::MatrixXd>& gradient) 
    { // 3D Tensor gradient

        if (!_numChannels) {
            _numChannels = gradient.size();
        }
        if (!_isTraining || _maxValue == 0.0) {
            // No clipping
            return gradient;
        }

        std::vector<Eigen::MatrixXd> clippedGradient(_numChannels);
        for (size_t c = 0; c < _numChannels; ++c) {
            double gradientNorm = gradient[c].norm();
            clippedGradient[c] = gradient[c] * (_maxValue / gradientNorm);
        }

        return clippedGradient;
    }
};


class WeightsRegularization {
private:
    size_t _degree;
    double _lambda;
    double _learningRate;

public:
    WeightsRegularization(size_t degree = 2, double lambda = 0.5, double learningRate = 0.01)
        : _degree(degree), 
          _lambda(lambda), 
          _learningRate(learningRate)
    {
		if (_degree < 0) {
			throw std::invalid_argument("Degree must be non-negative.");
		}
		if (_lambda < 0.0) {
			throw std::invalid_argument("Lambda must be non-negative.");
		}
		if (_learningRate <= 0.0) {
			throw std::invalid_argument("Learning rate must be positive.");
		}
    }

    Eigen::MatrixXd Regularize(Eigen::MatrixXd& weights, Eigen::MatrixXd& dW) {
        Eigen::MatrixXd regulizedWeights;

        Eigen::MatrixXd dW_reg = weights * (_degree * _lambda);

        regulizedWeights = weights - _learningRate * (dW + dW_reg);

        return regulizedWeights;
    }
};

class Dropout {
private:
    size_t _inputHeight;
    size_t _inputWidth;
    size_t _numChannels;
    double _dropoutRate;
    bool _isTraining;

public:
    Dropout(double dropoutRate = 0.5)
        : _dropoutRate(dropoutRate), 
          _isTraining(true), 
          _inputHeight(0), 
          _inputWidth(0), 
          _numChannels(0)
    {
        if (_dropoutRate < 0.0 || _dropoutRate >= 1.0) {
            throw std::invalid_argument("Dropout rate must be in the range [0, 1).");
        }
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) 
    {
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
        for (size_t c = 0; c < _numChannels; ++c) {
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

private: 
    Eigen::MatrixXd CreateRandomMask() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        Eigen::MatrixXd randomMask = Eigen::MatrixXd::NullaryExpr(_inputHeight, 
            _inputWidth,[&dist, &gen]() { return dist(gen); });

        return randomMask;
    }
};