#ifndef FULLYCONNECTED_HPP
#define FULLYCONNECTED_HPP

#include <iostream>
#include <random>
#include <memory>
#include <cassert>
#include <Eigen/Dense>
#include "../Optimizer.hpp"
#include "../Activation.hpp"
#include "../Regularization.hpp"

#include <mutex>


class FullyConnected {
private:
	const size_t _inputSize;
	const size_t _outputSize;

	size_t _inputChannels;
	size_t _inputHeight;
	size_t _inputWidth;

	Eigen::VectorXd _flatInput;
	Eigen::VectorXd _output;

	Eigen::MatrixXd _weights;
	Eigen::VectorXd _bias;
	Eigen::MatrixXd _weightsGradient;
	Eigen::VectorXd _biasGradient;

	std::unique_ptr<Activation> _activation;
	std::unique_ptr<AdamOptimizer> _optimizer;


public:
	FullyConnected(size_t inputSize, size_t outputSize, 
	std::unique_ptr<Activation> activationFunction)
		:_inputSize(inputSize), _outputSize(outputSize), _inputChannels(0),
		_optimizer(std::make_unique<AdamOptimizer>(-1)),
		_activation(std::move(activationFunction))
	{
		_initializeWeights();

		_weightsGradient.resize(_outputSize, _inputSize);
		_biasGradient.resize(_outputSize);

		_flatInput.resize(_inputSize);
		_output.resize(_outputSize);
	}

	Eigen::VectorXd forward(const Eigen::VectorXd& input)
	{ //Vector to vector (from fully-connected)
		
		if (!_inputChannels) { // initialize input dimensions
			_inputChannels = 1;
			_inputHeight = input.rows();
			_inputWidth = input.cols();
		}

		assert(_inputSize == _inputHeight * _inputWidth);

		_flatInput = input;
		
		Eigen::VectorXd preActivationOut = _weights * _flatInput + _bias;
		_output = _activation->Activate(preActivationOut);

		return _output;
	}

	Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input)
	{ //3D input to vector (from convolutional)

		if (!_inputChannels) { // initialize input dimensions
			_inputChannels = input.size();
			_inputHeight = input[0].rows();
			_inputWidth = input[0].cols();
		}

		assert(_inputSize == _inputChannels * _inputHeight * _inputWidth);

		_flatInput = _flattenData(input);
		Eigen::VectorXd preActivationOut = _weights * _flatInput + _bias;
		_output = _activation->Activate(preActivationOut);

		return _output;
	}

	Eigen::VectorXd backward(Eigen::VectorXd& lossGradient)
	{ //Backward pass to FC layer
		Eigen::VectorXd dLoss_dPreActivation = _activation->computeGradient(
			lossGradient, _output);

		// Calculate the gradient w.r.t the parameters
		_weightsGradient = dLoss_dPreActivation * _flatInput.transpose();
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
		_weightsGradient = dLoss_dPreActivation * _flatInput.transpose();
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
		std::vector<Eigen::MatrixXd> inputGradient = _unflattenData(
			flatInputGradient);

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
		for (size_t r = 0; r < _outputSize; ++r) {
			for (size_t c = 0; c < _inputSize; ++c) {
				_weights(r, c) = distribution(randomEngine);
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

		size_t rowIndex = 0;
		for (const auto& matrix : data) {
			Eigen::Map<const Eigen::VectorXd> matrixMap(matrix.data(),
				matrix.size());
			flattenData.segment(rowIndex, matrixMap.size()) = matrixMap;
			rowIndex += matrixMap.size();

			if (rowIndex > _inputSize) {
				std::cerr << "Row index exceeded the limit of input size, " <<
					_inputSize << std::endl;
				break; //Error
			}
		}

		return flattenData;
	}

	std::vector<Eigen::MatrixXd> _unflattenData(const Eigen::VectorXd& flattenedData) 
	const 
	{ //reconstruct a 3D Tensor from a flattened Vector
        std::vector<Eigen::MatrixXd> unflattenedData;
        unflattenedData.reserve(_inputChannels);

        size_t index = 0;
        for (size_t c = 0; c < _inputChannels; ++c) {
            Eigen::Map<const Eigen::MatrixXd> matrixMap(flattenedData.data() 
				+ index, _inputHeight, _inputWidth);
            unflattenedData.emplace_back(matrixMap);
            index += _inputHeight * _inputWidth;
        
			if (index > _inputSize) {
					std::cerr << "Index exceeded the limit of flattened data size, " << 
						_inputSize << std::endl;
					break; // Error
			}
		}

        return unflattenedData;
	}
	
};

#endif // FULLYCONNECTED_HPP