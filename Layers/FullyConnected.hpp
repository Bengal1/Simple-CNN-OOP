#pragma once

#include <iostream>
#include <random>
#include <memory>
#include <cassert>
#include <Eigen/Dense>
#include "../Optimizer.hpp"
#include "../Activation.hpp"
#include "../Regularization.hpp"


class FullyConnected {
private:
	const int _inputSize;
	const int _outputSize;
	const int _batchSize;

	size_t _inputChannels;
	size_t _inputHeight;
	size_t _inputWidth;

	Eigen::MatrixXd _weights;
	Eigen::VectorXd _bias;

	Eigen::VectorXd _input; //flatten/vector
	Eigen::VectorXd _output;
	Eigen::MatrixXd _weightsGradient;
	Eigen::VectorXd _biasGradient;

	std::unique_ptr<Activation> _activation;
	std::unique_ptr<AdamOptimizer> _optimizer;


public:
	FullyConnected(int inputSize, int outputSize, std::unique_ptr<Activation>
		activationFunction, int batchSize = 1)
		:_inputSize(inputSize), _outputSize(outputSize), _inputChannels(0),
		_batchSize(batchSize),
		_optimizer(std::make_unique<AdamOptimizer>(-1)),
		_activation(std::move(activationFunction))
	{
		_initializeWeights();

		_weightsGradient.resize(_outputSize, _inputSize);
		_biasGradient.resize(_outputSize);

		if (_batchSize == 1) {
			_input.resize(_inputSize);
			_output.resize(_outputSize);
		}
		else {
			std::cerr << "Non-positive Batch size is not valid! " <<
				_batchSize << std::endl;
			exit(-1);
		}
	}

	Eigen::VectorXd forward(const Eigen::VectorXd& input)
	{ //Vector to vector (from fully-connected)
		if (!_inputChannels) {
			_inputChannels = 1;
			_inputHeight = input.rows();
			_inputWidth = input.cols();
		}

		assert(_inputSize == _inputHeight * _inputWidth);

		if (_batchSize == 1) {
			_input = input;
		}

		Eigen::VectorXd preActivationOut = _weights * _input + _bias;
		_output = _activation->Activate(preActivationOut);

		return _output;
	}

	Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input)
	{ //3D input to vector (from convolutional)
		if (!_inputChannels) {
			_inputChannels = input.size();
			_inputHeight = input[0].rows();
			_inputWidth = input[0].cols();
		}

		assert(_inputSize == _inputChannels * _inputHeight * _inputWidth);

		_input = _flattenData(input);
		Eigen::VectorXd preActivationOut = _weights * _input + _bias;
		_output = _activation->Activate(preActivationOut);

		return _output;
	}

	Eigen::VectorXd backward(Eigen::VectorXd& lossGradient)
	{ //Backward pass to FC layer
		Eigen::VectorXd dLoss_dPreActivation = _activation->computeGradient(
			lossGradient, _output);

		// Calculate the gradient w.r.t the parameters
		_weightsGradient = dLoss_dPreActivation * _input.transpose();
		_biasGradient = dLoss_dPreActivation;

		// L2 Regularization
		/*double lambda = 5.0;
		Eigen::MatrixXd dW_reg = _weights * (2.0 * lambda);
		Eigen::VectorXd db_reg = _bias * (2.0 * lambda);
		_weightsGradient = _weightsGradient + dW_reg;
		_biasGradient = _biasGradient + db_reg;
		*/
		_updateParameters();

		// Calculate the gradient w.r.t the input
		Eigen::VectorXd inputGradient = _weights.transpose() * dLoss_dPreActivation;

		return inputGradient;
	}

	std::vector<Eigen::MatrixXd> backward(Eigen::VectorXd& lossGradient, bool input3D)
	{ //Backward pass to Convolution2D layer
		Eigen::VectorXd dLoss_dPreActivation = _activation->computeGradient(
			lossGradient, _output);

		// Calculate the gradient w.r.t the parameters
		_weightsGradient = dLoss_dPreActivation * _input.transpose();
		_biasGradient = dLoss_dPreActivation;

		// L2 Regularization
		/*double lambda = 0.5;
		Eigen::MatrixXd dW_reg = _weights * 2.0 * lambda;
		Eigen::VectorXd db_reg = _bias * 2.0 * lambda;
		_weightsGradient = _weightsGradient + dW_reg;
		_biasGradient = _biasGradient + db_reg;
		*/
		_updateParameters();

		// Calculate the gradient w.r.t the input
		Eigen::VectorXd flatInputGradient = _weights.transpose() *
			dLoss_dPreActivation;
		std::vector<Eigen::MatrixXd> inputGradient =
			_unflattenInputGradient(flatInputGradient);

		return inputGradient;
	}

	void SetTestMode()
	{
		_weightsGradient.resize(0, 0);
		_biasGradient.resize(0);
	}

	void SetTrainingMode()
	{
		_weightsGradient.setConstant(_outputSize, _inputSize, 0.0);
		_biasGradient.setConstant(_outputSize, 0.0);
	}

private:
	void _initializeWeights() 
	{
		_weights.resize(_outputSize, _inputSize);
		_bias.resize(_outputSize);

		// Setup random number generator and distribution for He initialization
		std::random_device rd;
		std::mt19937 randomEngine(rd());
		std::normal_distribution<double> distribution(0, std::sqrt(2.0 /
			_inputSize));

		// Initialize weights
		for (int i = 0; i < _outputSize; ++i) {
			for (int j = 0; j < _inputSize; ++j) {
				_weights(i, j) = distribution(randomEngine);
			}
		}

		_bias.setZero(); //Initialize bias
	}

	void _updateParameters()
	{
		// Update weights and biases
		_optimizer->updateStep(_weights, _weightsGradient);
		_optimizer->updateStep(_bias, _biasGradient);
		// Reset gradients
		_weightsGradient.setZero();
		_biasGradient.setZero();
	}

	Eigen::VectorXd _flattenData(const std::vector<Eigen::MatrixXd>& data) const 
	{ //flatten 3D Tensor to a Vector
		Eigen::VectorXd flattenData(_inputSize);

		int rowIndex = 0;
		for (const auto& matrix : data) {
			Eigen::Map<const Eigen::VectorXd> matrixMap(matrix.data(),
				matrix.size());
			flattenData.block(rowIndex, 0, matrixMap.size(), 1) = matrixMap;
			rowIndex += matrixMap.size();

			if (rowIndex > _inputSize) {
				std::cerr << "Row index exceeded the limit of input size, " <<
					_inputSize << std::endl;
				break; //Error
			}
		}

		return flattenData;
	}

	Eigen::VectorXd _flattenData(const Eigen::MatrixXd& data) const 
	{
		Eigen::VectorXd flattenData(_inputSize);

		int rowIndex = 0;
		Eigen::Map<const Eigen::VectorXd> matrixMap(data.data(),
			data.size());
		flattenData.block(rowIndex, 0, matrixMap.size(), 1) = matrixMap;
		rowIndex += matrixMap.size();

		if (rowIndex > _inputSize) {
			std::cerr << "Row index exceeded the limit of input size, " <<
				_inputSize << std::endl; //Error
		}

		return flattenData;

	}

	std::vector<Eigen::MatrixXd> _unflattenInputGradient(const Eigen::VectorXd&
		flattenInputGradient) const 
	{

		std::vector<Eigen::MatrixXd> unflattenInputGradient(_inputChannels,
			Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

		for (int c = 0; c < _inputChannels; ++c) {
			for (int h = 0; h < _inputHeight; ++h) {
				for (int w = 0; w < _inputWidth; ++w) {
					unflattenInputGradient[c](h, w) =
						flattenInputGradient[c * _inputSize/_inputChannels + 
						h * _inputHeight + w];
				}
			}
		}
		return unflattenInputGradient;
	}
};