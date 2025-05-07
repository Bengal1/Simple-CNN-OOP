#pragma once

#include <iostream>
#include <random>
#include <memory>
#include <Eigen/Dense>
#include "../Optimizer.hpp"
#include "../Activation.hpp"
#include "../Regularization.hpp"

class FullyConnected {
private:
	// layer dimensions
	const size_t _inputSize;
	const size_t _outputSize;
	const size_t _batchSize;
	// input dimensions
	size_t _inputChannels;
	size_t _inputHeight;
	size_t _inputWidth;
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
	FullyConnected(size_t inputSize, size_t outputSize, double maxGradNorm = -1.0, double weightDecay = 0.0, size_t batchSize = 1)
		: _inputSize(inputSize),
		_outputSize(outputSize),
		_batchSize(batchSize),
		_inputChannels(0),
		_inputHeight(0),
		_inputWidth(0),
		_optimizer(std::make_unique<AdamOptimizer>(-1, maxGradNorm, weightDecay))
	{
		if (_inputSize == 0) {
			throw std::invalid_argument("[FullyConnected]: Input size must be greater than zero.");
		}
		if (_outputSize == 0) {
			throw std::invalid_argument("[FullyConnected]: Output size must be greater than zero.");
		}
		if (_batchSize == 0) {
			throw std::invalid_argument("[FullyConnected]: Batch size must be greater than zero.");
		}
		// Initialize weights and gradients
		_initializeWeights();
		// Initialize input
		_flatInput = Eigen::VectorXd::Zero(_inputSize);
	}

	Eigen::VectorXd forward(const Eigen::VectorXd& input) {
		if (_inputChannels == 0) _inferShape(input);
		if (input.size() != _inputSize) {
			throw std::invalid_argument("[FullyConnected]: Input size does not match.");
		}
		_flatInput = input;
		return _weights * _flatInput + _bias;
	}

    Eigen::VectorXd forward(const Eigen::MatrixXd& input) {
        if (_inputChannels == 0) _inferShape(input);
		if (input.rows() != _inputHeight || input.cols() != _inputWidth) {
			throw std::invalid_argument("[FullyConnected]: Input dimensions do not match.");
		}
		_flatInput = _flattenData(input);
		return _weights * _flatInput + _bias;
    }

    Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input) {
        if (_inputChannels == 0) _inferShape(input);
		if (input.size() != _inputChannels) {
			throw std::invalid_argument("[FullyConnected]: Input channels do not match.");
		}
		// Flatten the input data
		_flatInput = _flattenData(input);

		// Perform the forward pass
		return _weights * _flatInput + _bias;
    }

	Eigen::VectorXd backward(const Eigen::VectorXd& dLoss_dOutput) {
		if (dLoss_dOutput.size() != _outputSize) {
			throw std::invalid_argument("[FullyConnected]: Loss gradient size does not match.");
		}
		// Compute gradient w.r.t. weights and bias
		_weightsGradient = dLoss_dOutput * _flatInput.transpose();
		_biasGradient = dLoss_dOutput;

		// Return gradient w.r.t. input
		return _weights.transpose() * dLoss_dOutput;
	}

	std::vector<Eigen::MatrixXd> backward(Eigen::VectorXd& dLoss_dOutput, bool input3D) {
		if (dLoss_dOutput.size() != _outputSize) {
			throw std::invalid_argument("[FullyConnected]: Loss gradient size does not match.");
		}
		Eigen::VectorXd flatGradient = backward(dLoss_dOutput);
		// Return gradient w.r.t.input (3D)
		return _unflattenInputGradient(flatGradient);
	}

	void updateParameters() {
		// Update weights and bias
		_optimizer->updateStep(_weights, _weightsGradient);
		_optimizer->updateStep(_bias, _biasGradient);
		// Reset gradients
		_weightsGradient.setZero();
		_biasGradient.setZero();
	}
	
	Eigen::MatrixXd getWeights() {
		return _weights;
	}
	
	Eigen::VectorXd getBias() {
		return _bias;
	}

	void setParameters(const Eigen::MatrixXd& weights, const Eigen::VectorXd& bias) {
		if (weights.rows() != _outputSize || weights.cols() != _inputSize) {
			throw std::invalid_argument("[FullyConnected]: Weights size does not match.");
		}
		if (bias.size() != _outputSize) {
			throw std::invalid_argument("[FullyConnected]: Bias size does not match.");
		}
		_weights = weights;
		_bias = bias;
	}

private:
	void _initializeWeights() {
		_weights.resize(_outputSize, _inputSize);
		_bias.resize(_outputSize);

		std::random_device rd;
		std::mt19937 rng(rd());
		std::normal_distribution<double> dist(0, std::sqrt(2.0 / _inputSize));
		// Initialize weights with He initialization
		auto generator = [&]() { return dist(rng); };
		_weights = _weights.unaryExpr([&](double) { return generator(); });
		// Initialize bias to zero
		_bias.setZero();
		// Initialize gradients to zero
		_weightsGradient = Eigen::MatrixXd::Zero(_outputSize, _inputSize);
		_biasGradient = Eigen::VectorXd::Zero(_outputSize);
	}

	void _inferShape(const Eigen::VectorXd& input) {
		_inputChannels = 1;
		_inputHeight = input.size();
		_inputWidth = 1;
		if (_inputSize != _inputHeight * _inputWidth) {
			throw std::invalid_argument("[FullyConnected]: Input size does not match the expected size.");
		}
	}

	void _inferShape(const Eigen::MatrixXd& input) {
		_inputChannels = 1;
		_inputHeight = input.rows();
		_inputWidth = input.cols();
		if (_inputSize != _inputHeight * _inputWidth) {
			throw std::invalid_argument("[FullyConnected]: Input size does not match the expected size.");
		}
	}

	void _inferShape(const std::vector<Eigen::MatrixXd>& input) {
		_inputChannels = input.size();
		_inputHeight = input[0].rows();
		_inputWidth = input[0].cols();
		if (_inputSize != _inputChannels * _inputHeight * _inputWidth) {
			throw std::invalid_argument("[FullyConnected]:Input size does not match the expected size.");
		}
	}

	Eigen::VectorXd _flattenData(const Eigen::MatrixXd& data) {
		// Flatten the input data
		Eigen::Map<const Eigen::VectorXd> map(data.data(), data.size());
		return map;
	}

	Eigen::VectorXd _flattenData(const std::vector<Eigen::MatrixXd>& data) {
		Eigen::VectorXd flat(_inputSize);
		Eigen::Index offset = 0;
		// Flatten the input data
		for (const auto& mat : data) {
			Eigen::Map<const Eigen::VectorXd> map(mat.data(), mat.size());
			flat.segment(offset, map.size()) = map;
			offset += map.size();
		}
		return flat;
	}

	std::vector<Eigen::MatrixXd> _unflattenInputGradient(const Eigen::VectorXd& flat) {
		std::vector<Eigen::MatrixXd> unflat(_inputChannels, 
					  Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
		// Unflatten input
		for (size_t c = 0; c < _inputChannels; ++c)
			for (size_t h = 0; h < _inputHeight; ++h)
				for (size_t w = 0; w < _inputWidth; ++w) {
					size_t index = c * _inputHeight * _inputWidth + h * _inputWidth + w;
					unflat[c](h, w) = flat(index);
				}
		return unflat;
	}
};

