#pragma once

#include <iostream>
#include <random>
#include <memory>
#include <cassert>
#include <Eigen/Dense>
#include "../Optimizer.h"
#include "../Activation.h"
#include "../Regularization.h"

class FullyConnected {
private:
	const int _inputSize;
	const int _outputSize;
	const int _batchSize;

	int _inputChannels = 0;
	int _inputHeight = 0;
	int _inputWidth = 0;

	Eigen::MatrixXd _weights;
	Eigen::VectorXd _bias;
	Eigen::MatrixXd _weightsGradient;
	Eigen::VectorXd _biasGradient;

	Eigen::VectorXd _input;
	Eigen::VectorXd _output;

	std::unique_ptr<Activation> _activation;
	std::unique_ptr<AdamOptimizer> _optimizer;
	GradientNormClipping _gnc;

public:
	FullyConnected(int inputSize, int outputSize, std::unique_ptr<Activation> activationFunction, int batchSize = 1)
		: _inputSize(inputSize), _outputSize(outputSize), _batchSize(batchSize),
		_activation(std::move(activationFunction)), 
		_optimizer(std::make_unique<AdamOptimizer>(-1)),
		_gnc(1.0)
	{
		if (_inputSize <= 0) {
			throw std::invalid_argument("Input size must be greater than zero.");
		}
		if (_outputSize <= 0) {
			throw std::invalid_argument("Output size must be greater than zero.");
		}
		if (_batchSize <= 0) {
			throw std::invalid_argument("Batch size must be greater than zero.");
		}
		// Initialize weights and gradients
		_initializeWeights();
		_weightsGradient.setZero(_outputSize, _inputSize);
		_biasGradient.setZero(_outputSize);
		_input.setZero(_inputSize);
		_output.setZero(_outputSize);
	}

	Eigen::VectorXd forward(const Eigen::VectorXd& input) {
		if (_inputChannels == 0) _inferShape(input);
		_input = input;
		_output = _activation->Activate(static_cast<Eigen::VectorXd>(_weights * _input + _bias));
		return _output;
	}

    Eigen::VectorXd forward(const Eigen::MatrixXd& input) {
        if (_inputChannels == 0) _inferShape(input);
        _input = _flattenData(input);
        _output = _activation->Activate(static_cast<Eigen::VectorXd>(_weights * _input + _bias));
        return _output;
    }

    Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input) {
        if (_inputChannels == 0) _inferShape(input);
        _input = _flattenData(input);
        _output = _activation->Activate(static_cast<Eigen::VectorXd>(_weights * _input + _bias));
        return _output;
    }

	Eigen::VectorXd backward(Eigen::VectorXd& lossGradient) {
		Eigen::VectorXd dLoss_dPreActivation = _activation->computeGradient(lossGradient, _output);
		_weightsGradient = dLoss_dPreActivation * _input.transpose();
		_biasGradient = _gnc.ClipGradient(dLoss_dPreActivation);

		double lambda = 0.05;
		_weightsGradient += _weights * (2.0 * lambda);
		_weightsGradient = _gnc.ClipGradient(_weightsGradient);

		updateParameters();
		return _gnc.ClipGradient(_weights.transpose() * dLoss_dPreActivation);
	}

	std::vector<Eigen::MatrixXd> backward(Eigen::VectorXd& lossGradient, bool input3D) {
		Eigen::VectorXd flatGrad = backward(lossGradient);
		return _unflattenInputGradient(flatGrad);
	}

	void updateParameters() {
		_optimizer->updateStep(_weights, _weightsGradient);
		_optimizer->updateStep(_bias, _biasGradient);
		_weightsGradient.setZero();
		_biasGradient.setZero();
	}

	void SetTestMode() {
		_weightsGradient.resize(0, 0);
		_biasGradient.resize(0);
	}

	void SetTrainingMode() {
		_weightsGradient.setConstant(_outputSize, _inputSize, 0.0);
		_biasGradient.setConstant(_outputSize, 0.0);
	}

private:
	void _initializeWeights() {
		_weights.resize(_outputSize, _inputSize);
		_bias.setZero(_outputSize);

		std::random_device rd;
		std::mt19937 rng(rd());
		std::normal_distribution<double> dist(0, std::sqrt(2.0 / _inputSize));
		// Initialize weights with He initialization
		auto generator = [&]() { return dist(rng); };
		_weights = _weights.unaryExpr([&](double) { return generator(); });
	}

	//void _initializeWeights() {
	//	_weights.resize(_outputSize, _inputSize);
	//	_bias.setZero(_outputSize);

	//	std::random_device rd;
	//	std::mt19937 rng(rd());
	//	std::normal_distribution<double> dist(0, std::sqrt(2.0 / _inputSize));
	//	// Initialize weights with He initialization
	//	for (int i = 0; i < _outputSize; ++i)
	//		for (int j = 0; j < _inputSize; ++j)
	//			_weights(i, j) = dist(rng);
	//}

	void _inferShape(const Eigen::VectorXd& input) {
		_inputChannels = 1;
		_inputHeight = input.size();
		_inputWidth = 1;
		if (_inputSize != _inputHeight * _inputWidth) {
			throw std::invalid_argument("Input size does not match the expected size.");
		}
	}

	void _inferShape(const Eigen::MatrixXd& input) {
		_inputChannels = 1;
		_inputHeight = input.rows();
		_inputWidth = input.cols();
		if (_inputSize != _inputHeight * _inputWidth) {
			throw std::invalid_argument("Input size does not match the expected size.");
		}
	}

	void _inferShape(const std::vector<Eigen::MatrixXd>& input) {
		_inputChannels = input.size();
		_inputHeight = input[0].rows();
		_inputWidth = input[0].cols();
		if (_inputSize != _inputChannels * _inputHeight * _inputWidth) {
			throw std::invalid_argument("Input size does not match the expected size.");
		}
	}

	Eigen::VectorXd _flattenData(const Eigen::MatrixXd& data) {
		Eigen::Map<const Eigen::VectorXd> map(data.data(), data.size());
		return map;
	}

	Eigen::VectorXd _flattenData(const std::vector<Eigen::MatrixXd>& data) {
		Eigen::VectorXd flat(_inputSize);
		int offset = 0;
		for (const auto& mat : data) {
			Eigen::Map<const Eigen::VectorXd> map(mat.data(), mat.size());
			flat.segment(offset, map.size()) = map;
			offset += map.size();
		}
		return flat;
	}

	std::vector<Eigen::MatrixXd> _unflattenInputGradient(const Eigen::VectorXd& flat) {
		std::vector<Eigen::MatrixXd> unflat(_inputChannels, Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
		for (int c = 0; c < _inputChannels; ++c)
			for (int h = 0; h < _inputHeight; ++h)
				for (int w = 0; w < _inputWidth; ++w) {
					int index = c * _inputHeight * _inputWidth + h * _inputWidth + w;
					unflat[c](h, w) = flat(index);
				}
		return unflat;
	}
};

