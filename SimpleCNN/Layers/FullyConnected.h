#pragma once

#include <iostream>
#include <random>
#include <memory>
#include <cassert>
#include "../Optimizer.h"
#include "../Activation.h"


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

	Eigen::VectorXd _input; //flatten
	Eigen::VectorXd _output;
	Eigen::MatrixXd _weightsGradient;
	Eigen::VectorXd _biasGradient;

	std::vector<Eigen::VectorXd> _inputBatch;
	std::vector<Eigen::VectorXd> _outputBatch;

	std::vector<Eigen::MatrixXd> _weightsGradBatch;
	std::vector<Eigen::VectorXd> _biasGradBatch;

	std::unique_ptr<AdamOptimizer> _optimizer;
	std::unique_ptr<Activation> _activation;

public:
	FullyConnected(int inputSize, int outputSize, std::unique_ptr<Activation> activationFunction, int batchSize = 1)
		:_inputSize(inputSize), _outputSize(outputSize), _inputChannels(0), _batchSize(batchSize),
		_optimizer(std::make_unique<AdamOptimizer>(-1)), _activation(std::move(activationFunction))
	{
		initializeWeights();

		_weightsGradient.resize(_outputSize, _inputSize);
		_biasGradient.resize(_outputSize);

		if (_activation == nullptr) {
			_activation.reset(new ReLU()); //Default Activation function
		}
		if (_batchSize == 1) {
			_input.resize(_inputSize);
			_output.resize(_outputSize);
		}
		else if (_batchSize > 1) {
			_inputBatch.assign(_batchSize, Eigen::VectorXd::Zero(_inputSize));
			_outputBatch.assign(_batchSize, Eigen::VectorXd::Zero(_outputSize));
		}
		else {
			//throw invalid_argument("Invalid constructor arguments. All values must be greater than zero.");
			std::cerr << "Non-positive Batch size is not valid! " << _batchSize << std::endl;
			//~FullyConnected(); //Error
		}
	}

	Eigen::VectorXd forward(const Eigen::VectorXd& input) {  //Vector to vector (from fully-connected)
		if (!_inputChannels) {
			_inputChannels = 1, _inputHeight = input.rows(),
				_inputWidth = input.cols();

			assert(_inputSize == _inputHeight * _inputWidth);
		}

		if (_batchSize == 1) {
			_input = input;
		}
		Eigen::VectorXd preActivation = _weights * input + _bias;
		_output = _activation->activate(preActivation);

		return _output;
	}

	Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input) { //3D input to vector (from convolutional)
		if (!_inputChannels) {
			_inputChannels = input.size(), _inputHeight = input[0].rows(),
				_inputWidth = input[0].cols();
		}

		assert(_inputSize == _inputChannels * _inputHeight * _inputWidth);

		_input = flattenData(input);
		Eigen::VectorXd preActivation = _weights * _input + _bias;
		_output = _activation->activate(preActivation);

		return _output;
	}

	std::vector<Eigen::VectorXd> forwardBatch(std::vector<Eigen::VectorXd>& inputBatch) { //Vectors batch to vector batch (from fully-connected)

		if (!_inputChannels) {
			_inputChannels = 1, _inputHeight = inputBatch[0].rows(),
				_inputWidth = inputBatch[0].cols();
		}

		for (int b = 0; b < _batchSize; b++) {
			_inputBatch[b] = inputBatch[b];
			_outputBatch[b] = forward(_inputBatch[b]);
		}

		return _outputBatch;
	}

	std::vector<Eigen::VectorXd> forwardBatch(std::vector<std::vector<Eigen::MatrixXd>>& inputBatch) { //3D input batch to vector batch (from convolutional)

		if (!_inputChannels) {
			_inputChannels = inputBatch[0].size(), 
				_inputHeight = inputBatch[0][0].rows(),
				_inputWidth = inputBatch[0][0].cols();
		}

		for (int b = 0; b < _batchSize; b++) {
			_inputBatch[b] = flattenData(inputBatch[b]);
			_outputBatch[b] = forward(_inputBatch[b]);
		}

		return _outputBatch;
	}

	Eigen::VectorXd backward(Eigen::VectorXd& lossGradient) { //Vector to vector
		// Calculate the gradient w.r.t the input
		Eigen::VectorXd inputGradient = _weights.transpose() * 
			_activation->computeGradient(lossGradient, _output);

		// Calculate the gradient w.r.t the parameters
		_weightsGradient = lossGradient * _input.transpose();
		_biasGradient = lossGradient;

		return inputGradient;
	}

	std::vector<Eigen::MatrixXd> backward(Eigen::VectorXd& lossGradient, bool input3D) {  //Vector to 3D output

		// Calculate the gradient w.r.t the input
		Eigen::VectorXd flatInputGradient = _weights.transpose() * 
			_activation->computeGradient(lossGradient, _output);
		std::vector<Eigen::MatrixXd> inputGradient = 
			unflattenInputGradient(flatInputGradient);

		// Calculate the gradient w.r.t the parameters
		_weightsGradient = lossGradient * _input.transpose();
		_biasGradient = lossGradient;

		return inputGradient;
	}

	std::vector<Eigen::VectorXd> backwardBatch(std::vector<Eigen::VectorXd>& lossGradientBatch) { //Vectors batch to vector batch (from fully-connected)
		std::vector<Eigen::VectorXd> inputGradBatch(_batchSize, Eigen::VectorXd::Zero(_inputSize));

		for (int b = 0; b < _batchSize; b++) {
			// Calculate the gradient w.r.t the input
			inputGradBatch[b] = _weights.transpose() * _activation->
				computeGradient(lossGradientBatch[b], _outputBatch[b]);

			// Calculate the gradient w.r.t the parameters
			_weightsGradient += lossGradientBatch[b] * _inputBatch[b].transpose();
			_biasGradient += lossGradientBatch[b];
		}

		return inputGradBatch;
	}

	std::vector<std::vector<Eigen::MatrixXd>> backwardBatch(std::vector<Eigen::VectorXd>& lossGradientBatch, bool input3D) { //Vector batch to 3D output batch
		std::vector<std::vector<Eigen::MatrixXd>> inputGradBatch(_batchSize,
			std::vector<Eigen::MatrixXd>(_inputChannels, Eigen::MatrixXd::Zero(_inputHeight, _inputWidth)));

		for (int b = 0; b < _batchSize; b++) {
			// Calculate the gradient w.r.t the input
			Eigen::VectorXd flatInputGradient = _weights.transpose() * _activation->
				computeGradient(lossGradientBatch[b], _outputBatch[b]);
			inputGradBatch[b] = unflattenInputGradient(flatInputGradient);

			// Calculate the gradient w.r.t the parameters
			_weightsGradient += lossGradientBatch[b] * _inputBatch[b].transpose();
			_biasGradient += lossGradientBatch[b];
		}

		return inputGradBatch;
	}

	void updateParameters() {
		// Update weights and biases
		_optimizer->updateStep(_weights, _weightsGradient);
		_optimizer->updateStep(_bias, _biasGradient);
		// Reset gradients
		_weightsGradient.setZero();
		_biasGradient.setZero();
	}

private:
	void initializeWeights() {
		_weights.resize(_outputSize, _inputSize);
		_bias.resize(_outputSize);

		// Setup random number generator and distribution for He initialization
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> distribution(0, std::sqrt(2.0 / _weights.cols()));

		// Initialize weights
		for (int i = 0; i < _weights.rows(); ++i) {
			for (int j = 0; j < _weights.cols(); ++j) {
				_weights(i, j) = distribution(gen);
			}
		}

		_bias.setOnes(); //Initialize bias
	}

	Eigen::VectorXd flattenData(const std::vector<Eigen::MatrixXd>& data) {
		Eigen::VectorXd flattenData(_inputSize);

		int rowIndex = 0;
		for (const auto& matrix : data) {
			Eigen::Map<const Eigen::VectorXd> matrixMap(matrix.data(), matrix.size());
			flattenData.block(rowIndex, 0, matrixMap.size(), 1) = matrixMap;
			rowIndex += matrixMap.size();

			if (rowIndex > _inputSize) {
				std::cerr << "Row index exceeded the limit of input size, " << _inputSize << std::endl;
				break; //Error
			}
		}
		return flattenData;

	}

	std::vector<Eigen::MatrixXd> unflattenInputGradient(const Eigen::VectorXd& flattenInputGradient) {
		std::vector<Eigen::MatrixXd> unflattenInputGradient(_inputChannels,
			Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

		for (int c = 0; c < _inputChannels; c++) {
			for (int h = 0; h < _inputHeight; h++) {
				for (int w = 0; w < _inputWidth; w++) {
					unflattenInputGradient[c](h, w) = flattenInputGradient[c + h + w];
				}
			}
		}

		return unflattenInputGradient;
	}

};