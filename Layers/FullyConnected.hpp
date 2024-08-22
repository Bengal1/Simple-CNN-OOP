#ifndef FULLYCONNECTED_HPP
#define FULLYCONNECTED_HPP

#include <iostream>
#include <random>
#include <memory>
#include "../Optimizer.hpp"
#include "../Activation.hpp"
#include "../Regularization.hpp"

//#include <mutex>


class FullyConnected {
private:
	const size_t _inputSize;
	const size_t _outputSize;
	size_t _inputChannels;
	size_t _inputHeight;
	size_t _inputWidth;
	bool _initialized;

	Eigen::VectorXd _flatInput;
	Eigen::VectorXd _preActivationOutput;

	Eigen::MatrixXd _weights;
	Eigen::VectorXd _bias;
	Eigen::MatrixXd _weightsGradient;
	Eigen::VectorXd _biasGradient;

	std::unique_ptr<Activation> _activation;
	std::unique_ptr<AdamOptimizer> _optimizer;

public:
	FullyConnected(size_t inputSize, size_t outputSize, 
				   std::unique_ptr<Activation> activationFunction)
		:_inputSize(inputSize),
		 _outputSize(outputSize),
		 _initialized(false),
		 _optimizer(std::make_unique<AdamOptimizer>(FullyConnectedMode)),
		 _activation(std::move(activationFunction))
	{
		_initializeWeights();

		_weightsGradient.resize(_outputSize, _inputSize);
		_biasGradient.resize(_outputSize);

		_flatInput.resize(_inputSize);
		_preActivationOutput.resize(_outputSize);
	}

	Eigen::VectorXd forward(const Eigen::VectorXd& input)
	{ //Vector to vector (from fully-connected)
		_setupInputDimensions(input.rows(), input.cols(), 1);
		_flatInput = input;
		
		_preActivationOutput = _weights * _flatInput + _bias;
		return _activation->Activate(_preActivationOutput);
	}

	Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input)
	{ //3D input to vector (from convolutional)
		_setupInputDimensions(input[0].rows(), input[0].cols(), input.size());
		
		_flatInput = _flattenData(input);
		_preActivationOutput = _weights * _flatInput + _bias;
		return _activation->Activate(_preActivationOutput);
	}

	Eigen::VectorXd backward(Eigen::VectorXd& lossGradient)
	{ //Backward pass to FC layer
		Eigen::VectorXd dLoss_dPreActivation = _activation->computeGradient(
											lossGradient, _preActivationOutput);

		// Calculate the gradient w.r.t the parameters
		_weightsGradient = dLoss_dPreActivation * _flatInput.transpose();
		_biasGradient = dLoss_dPreActivation;

		// Calculate the gradient w.r.t the input
		Eigen::VectorXd inputGradient = _weights.transpose() * dLoss_dPreActivation;

		_updateParameters();

		return inputGradient;
	}

	std::vector<Eigen::MatrixXd> backward(Eigen::VectorXd& lossGradient, bool input3D)
	{ //Backward pass to Convolution2D layer
		Eigen::VectorXd dLoss_dPreActivation = _activation->computeGradient(
											   lossGradient, _preActivationOutput);

		// Calculate the gradient w.r.t the parameters
		_weightsGradient = dLoss_dPreActivation * _flatInput.transpose();
		_biasGradient = dLoss_dPreActivation;
		
		// Calculate the gradient w.r.t the input
		Eigen::VectorXd flatInputGradient = _weights.transpose() * dLoss_dPreActivation;

		std::vector<Eigen::MatrixXd> inputGradient = _unflattenData(flatInputGradient);

		_updateParameters();

		return inputGradient;
	}

	void setTestMode()
	{
		_weightsGradient.resize(0, 0);
		_biasGradient.resize(0);
	}

	void setTrainingMode()
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

	void _setupInputDimensions(size_t inputHeight, size_t inputWidth, size_t inputChannels = 0)
    {
        if (!_initialized) {
            _inputHeight = inputHeight;
            _inputWidth = inputWidth;
            _inputChannels = inputChannels;

			// Calculate the expected input size and validate it
			size_t expectedInputSize = _inputChannels * _inputHeight * _inputWidth;
			if (_inputSize != expectedInputSize) {
				throw std::invalid_argument("Input size does not match the expected dimensions.");
			}
			_initialized = true;
		}
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
			Eigen::Map<const Eigen::VectorXd> matrixMap(matrix.data(), matrix.size());
			flattenData.segment(rowIndex, matrixMap.size()) = matrixMap;
			rowIndex += matrixMap.size();

			if (rowIndex > _inputSize) {
                throw std::runtime_error("Row index exceeded the limit of input size.");
            }
		}

		return flattenData;
	}

	std::vector<Eigen::MatrixXd> _unflattenData(const Eigen::VectorXd& flattenedData) 
	const 
	{ //reconstruct a 3D Tensor from a flattened Vector
        std::vector<Eigen::MatrixXd> unflattenedData(_inputChannels);

        size_t index = 0;
        for (size_t c = 0; c < _inputChannels; ++c) {
            Eigen::Map<const Eigen::MatrixXd> matrixMap(flattenedData.data() 
											+ index, _inputHeight, _inputWidth);
            unflattenedData[c] = matrixMap;
            index += _inputHeight * _inputWidth;
	
			if (index > _inputSize) {
                throw std::runtime_error("Index exceeded the limit of flattened data size.");
            }
		}

        return unflattenedData;
	}
	
};

#endif // FULLYCONNECTED_HPP

// L2 Regularization
/*double lambda = 0.5;
Eigen::MatrixXd dW_reg = _weights * 2.0 * lambda;
Eigen::VectorXd db_reg = _bias * 2.0 * lambda;
_weightsGradient = _weightsGradient + dW_reg;
_biasGradient = _biasGradient + db_reg;
*/
