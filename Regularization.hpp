#pragma once

#include <memory>
#include <random>
#include <iostream>
#include "Optimizer.hpp"



class BatchNormalization {
private:
	// Input dimensions
    size_t _numChannels = 0;
    size_t _channelHeight = 0;
    size_t _channelWidth = 0;
    // Operations flags
    bool _initialized = false;
    bool _isTraining = true;
	// Hyperparameters
    double _epsilon = 1e-6;
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
	// Input data
    std::vector<Eigen::MatrixXd> _input; 
	// Optimizer
	std::unique_ptr<Optimizer> _optimizer;

public:
    BatchNormalization(double maxGradNorm = -1.0, 
        double weightDecay = 0.0, double momentum = 0.1)
        :_momentum(momentum),
        _optimizer(std::make_unique<AdamOptimizer>(-2, maxGradNorm, weightDecay))
    {
        if (_momentum <= 0.0 || _momentum > 1.0) {
            throw std::invalid_argument("[BatchNormalization]: Momentum must be in the range [0, 1).");
        }
    }
    ~BatchNormalization() = default;

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) 
    {
        if (!_initialized) {
            _InitializeParameters(input);
        }   
        if (input.empty()) {
            throw std::invalid_argument("[BatchNormalization]: Input is empty.");
        }
        if (input.size() != _numChannels) {
            throw std::invalid_argument("[BatchNormalization]: Input channels do not match.");
        }
		if (input[0].rows() != _channelHeight || input[0].cols() != _channelWidth) {
			throw std::invalid_argument("[BatchNormalization]: Input dimensions do not match.");
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
            double dMean = (dXhat.array() * (-invStd)).sum() + 
                            dVar * (-2.0 / N) * x_mu.sum();

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
        if (_optimizer == nullptr) {
            throw std::invalid_argument("[BatchNormalization]: Optimizer is not set.");
        }
		if (input.empty()) {
			throw std::invalid_argument("[BatchNormalization]: Input is empty.");
		}
		if (input[0].rows() == 0 || input[0].cols() == 0) {
			throw std::invalid_argument("[BatchNormalization]: Input dimensions are zero.");
		}
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


class Dropout {
private:
    // Input dimensions
    size_t _inputHeight;
    size_t _inputWidth;
    size_t _numChannels;
    // Hyperparameters
    const double _dropoutRate;
    const double _dropoutScale;
    // Training flag
    bool _isTraining;
    // Dropout mask
    Eigen::MatrixXd _dropoutMask;
	std::vector<Eigen::MatrixXd> _dropoutMask3D;
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
        _dist(0.0, 1.0),
		_dropoutScale(1.0 / (1.0 - dropoutRate))
    {
        if (_dropoutRate < 0.0 || _dropoutRate >= 1.0) {
            throw std::invalid_argument("[Dropout]: Dropout rate must be in the range [0, 1).");
        }
    }
	~Dropout() = default;

	template<typename T>
	T forward(const T& input) {
		// Handle input dimension initialization lazily
		if (_numChannels == 0) {
			_getInputDimensions(input);
		}
		if (!_isTraining || _dropoutRate == 0.0) {
			// No dropout
			return input;
		}
		if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			if (_inputHeight != input.rows() || _inputWidth != input.cols()) {
				throw std::invalid_argument("[Dropout]: Input dimensions do not match.");
			}
			// Create a random mask with values between 0 and 1
			Eigen::MatrixXd randomMask = _createRandomMask();
			// Apply dropout
			_dropoutMask = (randomMask.array() > _dropoutRate).cast<double>();
			return input.array() * _dropoutMask.array() * _dropoutScale;
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			if (_numChannels != input.size()) {
				throw std::invalid_argument("[Dropout]: Input channels do not match.");
			}
			if (_inputHeight != input[0].rows() || _inputWidth != input[0].cols()) {
				throw std::invalid_argument("[Dropout]: Input dimensions do not match.");
			}
			std::vector<Eigen::MatrixXd> droppedOutInput(_numChannels,
				Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
			// Reset Dropout masks
            _dropoutMask3D.clear();
            _dropoutMask3D.reserve(_numChannels);

            for (size_t c = 0; c < _numChannels; ++c) {
                Eigen::MatrixXd randomMask = _createRandomMask();
                Eigen::MatrixXd mask = (randomMask.array() > _dropoutRate).cast<double>();
                _dropoutMask3D.push_back(mask);
                droppedOutInput[c] = input[c].array() * mask.array() * _dropoutScale;
            }

			return droppedOutInput;
		}
	}

	template<typename T>
	T backward(const T& dOutput) {
		if (!_isTraining || _dropoutRate == 0.0) {
			return dOutput;
		}
		if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			if (_inputHeight != dOutput.rows() || _inputWidth != dOutput.cols()) {
				throw std::invalid_argument("[Dropout]: Gradient dimensions do not match.");
			}
			return dOutput.array() * _dropoutMask.array() * _dropoutScale;
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			if (_numChannels != dOutput.size()) {
				throw std::invalid_argument("[Dropout]: Gradient size does not match number of channels.");
			}
			if (_inputHeight != dOutput[0].rows() || _inputWidth != dOutput[0].cols()) {
				throw std::invalid_argument("[Dropout]: Gradient dimensions do not match.");
			}
			std::vector<Eigen::MatrixXd> dOutput3D(_numChannels,
				Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
			for (size_t c = 0; c < _numChannels; ++c) {
				dOutput3D[c] = dOutput[c].array() * _dropoutMask3D[c].array() * _dropoutScale;
			}
			return dOutput3D;
		}
		else {
			throw std::invalid_argument("[Dropout]: Unsupported input type.");
		}
	}

    void setTrainingMode(bool isTraining) {
        _isTraining = isTraining;
    }

private:
    Eigen::MatrixXd _createRandomMask() {
        return Eigen::MatrixXd::NullaryExpr(_inputHeight, _inputWidth,
            [this]() { return _dist(_randGen); });
    }

	template<typename T>
    void _getInputDimensions(const T& input) {
        if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
            _numChannels = 1;
            _inputHeight = input.size();
            _inputWidth = 1;
            _dropoutMask = Eigen::MatrixXd::Zero(_inputHeight, _inputWidth);
        }
        else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
            _numChannels = 1;
            _inputHeight = input.rows();
            _inputWidth = input.cols();
            _dropoutMask = Eigen::MatrixXd::Zero(_inputHeight, _inputWidth);
        }
        else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
            _numChannels = input.size();
            _inputHeight = input[0].rows();
            _inputWidth = input[0].cols();
            _dropoutMask = Eigen::MatrixXd::Zero(_inputHeight, _inputWidth);
        }
        else {
            throw std::invalid_argument("[Dropout]: Unsupported input type.");
        }
    }
};



