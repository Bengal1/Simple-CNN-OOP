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
	const size_t _inputSize;
	const size_t _outputSize;
	const size_t _batchSize;

	size_t _inputChannels;
	size_t _inputHeight;
	size_t _inputWidth;

	Eigen::MatrixXd _weights;
	Eigen::VectorXd _bias;
	Eigen::MatrixXd _weightsGradient;
	Eigen::VectorXd _biasGradient;

	Eigen::VectorXd _flatInput;
	Eigen::VectorXd _preActivationOutput;

	std::unique_ptr<Activation> _activation;
	std::unique_ptr<AdamOptimizer> _optimizer;
	//GradientNormClipping _gnc;

public:
	FullyConnected(size_t inputSize, size_t outputSize,
		std::unique_ptr<Activation> activationFunction, size_t batchSize = 1)
		: _inputSize(inputSize), 
		  _outputSize(outputSize), 
		  _batchSize(batchSize),
		  _inputChannels(0),
		  _inputHeight(0),
	      _inputWidth(0),
		  _activation(std::move(activationFunction)), 
		  _optimizer(std::make_unique<AdamOptimizer>(-1))//,
		  //_gnc(1.0)
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
		// Initialize input and output
		_flatInput = Eigen::VectorXd::Zero(_inputSize);
		_preActivationOutput = Eigen::VectorXd::Zero(_outputSize);
	}

	Eigen::VectorXd forward(const Eigen::VectorXd& input) {
		if (_inputChannels == 0) _inferShape(input);
		_flatInput = input;
		_preActivationOutput = _weights * _flatInput + _bias;
		return _activation->Activate(_preActivationOutput);
	}

    Eigen::VectorXd forward(const Eigen::MatrixXd& input) {
        if (_inputChannels == 0) _inferShape(input);
        _flatInput = _flattenData(input);
		_preActivationOutput = _weights * _flatInput + _bias;
        return _activation->Activate(_preActivationOutput);
    }

    Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input) {
        if (_inputChannels == 0) _inferShape(input);
        _flatInput = _flattenData(input);
		_preActivationOutput = _weights * _flatInput + _bias;
        return _activation->Activate(_preActivationOutput);
    }

	Eigen::VectorXd backward(const Eigen::VectorXd& lossGradient) {
		Eigen::VectorXd dLoss_dPreActivation = _activation->computeGradient(
											lossGradient, _preActivationOutput);
		_weightsGradient = dLoss_dPreActivation * _flatInput.transpose();
		_biasGradient = dLoss_dPreActivation;
		/*_biasGradient = _gnc.ClipGradient(dLoss_dPreActivation);
		/*double lambda = 0.5;
		_weightsGradient += _weights * lambda;
		_weightsGradient = _gnc.ClipGradient(_weightsGradient);
		updateParameters();
		return _gnc.ClipGradient( _weights.transpose() * dLoss_dPreActivation);*/
		return _weights.transpose() * dLoss_dPreActivation;
	}

	std::vector<Eigen::MatrixXd> backward(Eigen::VectorXd& lossGradient, bool input3D) {
		Eigen::VectorXd flatGradient = backward(lossGradient);
		return _unflattenInputGradient(flatGradient);
	}

	void updateParameters() {
		_optimizer->updateStep(_weights, _weightsGradient);
		_optimizer->updateStep(_bias, _biasGradient);
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
		Eigen::Map<const Eigen::VectorXd> map(data.data(), data.size());
		return map;
	}

	Eigen::VectorXd _flattenData(const std::vector<Eigen::MatrixXd>& data) {
		Eigen::VectorXd flat(_inputSize);
		Eigen::Index offset = 0;
		for (const auto& mat : data) {
			Eigen::Map<const Eigen::VectorXd> map(mat.data(), mat.size());
			flat.segment(offset, map.size()) = map;
			offset += map.size();
		}
		return flat;
	}

	std::vector<Eigen::MatrixXd> _unflattenInputGradient(const Eigen::VectorXd& flat) {
		std::vector<Eigen::MatrixXd> unflat(_inputChannels, Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
		for (size_t c = 0; c < _inputChannels; ++c)
			for (size_t h = 0; h < _inputHeight; ++h)
				for (size_t w = 0; w < _inputWidth; ++w) {
					size_t index = c * _inputHeight * _inputWidth + h * _inputWidth + w;
					unflat[c](h, w) = flat(index);
				}
		return unflat;
	}
};

