#pragma once

#include <vector>
#include <memory>
#include <random>
#include <iostream>
#include <Eigen/Dense>
#include "Optimizer.hpp"



class BatchNormalization {
private:
	// Input dimensions
    size_t _numChannels;
    size_t _channelHeight;
    size_t _channelWidth;
    // Operations flags
    bool _initialized;
    bool _isTraining;
	// Hyperparameters
    double _epsilon;
    double _momentum;
	// Mean and variance
    std::vector<double> _channelMean;
    std::vector<double> _channelVariance;
    std::vector<double> _runningMean;
    std::vector<double> _runningVariance;
	// Learnable parameters
    Eigen::VectorXd _gamma;
    Eigen::VectorXd _beta;
    Eigen::VectorXd _dGamma;
    Eigen::VectorXd _dBeta;
    std::vector<Eigen::MatrixXd> _input; 
	// Optimizer
	std::unique_ptr<Optimizer> _optimizer;

public:
    BatchNormalization(double maxGradNorm = -1.0, double weightDecay = 0.0, double momentum = 0.1)
        : _numChannels(0),
        _channelHeight(0),
        _channelWidth(0),
        _epsilon(1e-6),
        _initialized(false),
        _isTraining(true),
        _momentum(momentum),
        _optimizer(std::make_unique<AdamOptimizer>(-2, maxGradNorm, weightDecay))
    {
        if (_momentum < 0.0 || _momentum > 1.0) {
            throw std::invalid_argument("[BatchNormalization]: Momentum must be in the range [0, 1].");
        }
    }

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) 
    {
        if (!_initialized) {
            _InitializeParameters(input);
        }
		if (_optimizer == nullptr) {
			throw std::invalid_argument("[BatchNormalization]: Optimizer is not set.");
		}   
        if (input.empty()) {
            throw std::invalid_argument("[BatchNormalization]: Input is empty.");
        }
        if (input.size() != _numChannels) {
            throw std::invalid_argument("[BatchNormalization]: Input channels do not match.");
        }
        
		// Initialize output
        std::vector<Eigen::MatrixXd> outputBN(_numChannels, 
            Eigen::MatrixXd::Zero(_channelHeight, _channelWidth));

        _input = input;
        
        for (size_t c = 0; c < _numChannels; ++c) {
            // Calculate mean and variance for each channel
            _channelMean[c] = input[c].sum() / (_channelHeight *
                            _channelWidth);
            _channelVariance[c] = (input[c].array() - _channelMean[c]).
                    square().sum() / (_channelHeight * _channelWidth);

            if (_isTraining) {
                // Update running mean and variance - exponential moving average
                _runningMean[c] = (1.0 - _momentum) * _runningMean[c] + _momentum * _channelMean[c];
                _runningVariance[c] = (1.0 - _momentum) * _runningVariance[c] + _momentum *
                                      _channelVariance[c];
            }
            
            // Normalizing the channel 
            double meanToUse = _isTraining ? _channelMean[c] : _runningMean[c];
            double varianceToUse = _isTraining ? _channelVariance[c] : _runningVariance[c];

            // Scale and shift using gamma and beta
            outputBN[c] = _gamma[c] * ((input[c].array() - meanToUse) / 
                          std::sqrt(varianceToUse + _epsilon)) + _beta[c];
        }
        return outputBN;
    }

    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& dOutput) 
    {
		if (dOutput.size() != _numChannels) {
			throw std::invalid_argument("[BatchNormalization]: Gradient size does not match number of channels.");
		}

        std::vector<Eigen::MatrixXd> inputGradient(_numChannels, 
                    Eigen::MatrixXd::Zero(_channelHeight, _channelWidth));
        size_t N = _channelHeight * _channelWidth;

        for (size_t c = 0; c < _numChannels; ++c) {
            double mean = _channelMean[c];
            double variance = _channelVariance[c];
            double invStd = 1.0 / std::sqrt(variance + _epsilon);

            // Gradients w.r.t gamma and beta
            Eigen::MatrixXd x_hat = (_input[c].array() - mean) * invStd;
            _dGamma(c) = (dOutput[c].array() * x_hat.array()).sum();
            _dBeta(c) = dOutput[c].sum();

            // Gradient w.r.t x_hat
            Eigen::MatrixXd dXhat = dOutput[c].array() * _gamma(c);

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

	void setTrainingMode(bool isTraining) {
		_isTraining = isTraining;
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
        _channelMean.assign(_numChannels, 0.0);
		_channelVariance.assign(_numChannels, 0.0);
        _runningMean.assign(_numChannels, 0.0);
        _runningVariance.assign(_numChannels, 1.0);
        _gamma = Eigen::VectorXd::Constant(_numChannels, 1.0);
        _beta = Eigen::VectorXd::Constant(_numChannels, 0.0);
        _dGamma = Eigen::VectorXd::Zero(_numChannels);
        _dBeta = Eigen::VectorXd::Zero(_numChannels);
    }
};


class GradientNormClipping {
private:
    size_t _numChannels;
    double _maxValue;
    bool _isTraining;

public:
    GradientNormClipping(double maxValue = 1.0, bool isTraining = true) 
        : _maxValue(maxValue),  
          _numChannels(0), 
          _isTraining(isTraining)
    {
		if (_maxValue < 0.0) {
			throw std::invalid_argument("[GradientNormClipping]: Max value must be non-negative.");
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
    double _lambda;
    double _learningRate;

public:
    WeightsRegularization( 
        double lambda = 0.5, double learningRate = 0.01)
        : _lambda(lambda), 
          _learningRate(learningRate)
    {
		if (_lambda < 0.0) {
			throw std::invalid_argument("[WeightsRegularization]: Lambda must be non-negative.");
		}
		if (_learningRate <= 0.0) {
			throw std::invalid_argument("[WeightsRegularization]: Learning rate must be positive.");
		}
    }

    Eigen::MatrixXd Regularize(const Eigen::MatrixXd& weights, 
                               const Eigen::MatrixXd& dW) 
    {
        Eigen::MatrixXd regulizedWeights;

        Eigen::MatrixXd dW_reg = weights * _lambda;

        regulizedWeights = weights - _learningRate * (dW + dW_reg);

        return regulizedWeights;
    }
};

class Dropout {
private:
	// Input dimensions
    size_t _inputHeight;
    size_t _inputWidth;
    size_t _numChannels;
	// Hyperparameters
    double _dropoutRate;
	double _dropoutScale;
    // Training flag
    bool _isTraining;
	// Dropout mask
    Eigen::MatrixXd _dropoutMask;
    //std::vector<Eigen::MatrixXd> _dropoutMask3D;
	// Random number generator
    std::mt19937 _randGen;
    std::uniform_real_distribution<double> _dist;

public:
    Dropout(double dropoutRate = 0.5)
        : _dropoutRate(dropoutRate),
          _isTraining(true), 
          _inputHeight(0), 
          _inputWidth(0), 
          _numChannels(0),
		  _randGen(std::random_device{}()),
		  _dist(0.0, 1.0)
    {
        if (_dropoutRate < 0.0 || _dropoutRate >= 1.0) {
            throw std::invalid_argument("[Dropout]: Dropout rate must be in the range [0, 1).");
        }
		// Dropout scale
		_dropoutScale = 1.0 / (1.0 - _dropoutRate);
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) 
    {
		if (!_numChannels) { //initialization
            _numChannels = 1;
            _inputHeight = input.rows();
            _inputWidth = input.cols();
			_dropoutMask = Eigen::MatrixXd::Zero(_inputHeight, _inputWidth);
        }
        if (!_isTraining || _dropoutRate == 0.0) {
            // No dropout
            return input;
        }
		if (_inputHeight != input.rows() || _inputWidth != input.cols()) {
			throw std::invalid_argument("[Dropout]: Input dimensions do not match.");
		}
        // Create a random mask with values between 0 and 1
        Eigen::MatrixXd randomMask = _createRandomMask();

        // Apply dropout
        _dropoutMask = (randomMask.array() > _dropoutRate).cast<double>();
        return input.array() * _dropoutMask.array() * _dropoutScale;
    }

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) {

		if (!_numChannels) { //initialization
            _numChannels = input.size();
            _inputHeight = input[0].rows();
            _inputWidth = input[0].cols();
            _dropoutMask = Eigen::MatrixXd::Zero(_inputHeight, _inputWidth);
			/*_dropoutMask3D.assign(_numChannels,
                Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));*/
        }
        if (!_isTraining || _dropoutRate == 0.0) {
            // No dropout
            return input;
        }
		if (_numChannels != input.size()) {
			throw std::invalid_argument("[Dropout]: Input channels do not match.");
		}
		if (_inputHeight != input[0].rows() || _inputWidth != input[0].cols()) {
			throw std::invalid_argument("[Dropout]: Input dimensions do not match.");
		}
        std::vector<Eigen::MatrixXd> dropedOutInput(_numChannels, 
                Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

        /*for (size_t c = 0; c < _numChannels; ++c) {
            // Create a random mask with values between 0 and 1
            Eigen::MatrixXd randomMask = _createRandomMask();
            
            // Apply dropout
            _dropoutMask3D[c] = (randomMask.array() > _dropoutRate).cast<double>();
            dropedOutInput[c] = input[c].array() * _dropoutMask3D[c].array() * _dropoutScale;
        }*/

        // Create a random Dropout mask
        Eigen::MatrixXd randomMask = _createRandomMask();
        _dropoutMask = (randomMask.array() > _dropoutRate).cast<double>();

        for (size_t c = 0; c < _numChannels; ++c) {
            // Apply dropout
            dropedOutInput[c] = input[c].array() * _dropoutMask.array() * _dropoutScale;
        }

        return dropedOutInput;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& dOutput) {
        if (!_isTraining || _dropoutRate == 0.0) {
            return dOutput;
        }
        return dOutput.array() * _dropoutMask.array() * _dropoutScale;
    }

	std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& dOutput) {
		if (!_isTraining || _dropoutRate == 0.0) {
			return dOutput;
		}
		std::vector<Eigen::MatrixXd> dOutput3D(_numChannels,
			Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

		for (size_t c = 0; c < _numChannels; ++c) {
			dOutput3D[c] = dOutput[c].array() * _dropoutMask.array() * _dropoutScale;
		}
		return dOutput3D;
	}

	void setTrainingMode(bool isTraining) {
		_isTraining = isTraining;
	}

private: 
    Eigen::MatrixXd _createRandomMask() {
        return Eigen::MatrixXd::NullaryExpr(_inputHeight, _inputWidth,
            [this]() { return _dist(_randGen); });
    }
};