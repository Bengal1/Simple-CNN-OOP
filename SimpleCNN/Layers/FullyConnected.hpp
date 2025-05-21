#pragma once

#include <iostream>
#include <random>
#include <memory>
#include "../Optimizer.hpp"


template <typename T>
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
	// Input type enum
	enum class InputType { Vector, Matrix, Tensor3D };
	InputType _inputType;
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
	FullyConnected(size_t inputSize, size_t outputSize, 
		double maxGradNorm = -1.0, double weightDecay = 0.0, 
		size_t batchSize = 1)
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

	// Forward pass
	template<typename T>
	Eigen::VectorXd forward(const T& input) {
		// Handle input dimension initialization lazily
		if (_inputChannels == 0) _getInputDimensions(input);

		// Case 1: If the input is Eigen::VectorXd
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			if (input.size() != _inputSize) {
				throw std::invalid_argument("[FullyConnected]: Input size does not match.");
			}
			_flatInput = input;
		}
		// Case 2: If the input is Eigen::MatrixXd
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			if (input.rows() != _inputHeight || input.cols() != _inputWidth) {
				throw std::invalid_argument("[FullyConnected]: Input dimensions do not match.");
			}
			_flatInput = _flattenInput(input);
		}
		// Case 3: If the input is std::vector<Eigen::MatrixXd>
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			if (input.size() != _inputChannels) {
				throw std::invalid_argument("[FullyConnected]: Input channels do not match.");
			}
			_flatInput = _flattenInput(input);
		}

		// Perform the forward pass
		return _weights * _flatInput + _bias;
	}

	// Backward pass
	auto backward(const Eigen::VectorXd& dLoss_dOutput) {
		if (dLoss_dOutput.size() != _outputSize) {
			throw std::invalid_argument("[FullyConnected]: Loss gradient size does not match.");
		}

		// Compute gradients w.r.t parameters
		_weightsGradient = dLoss_dOutput * _flatInput.transpose();
		_biasGradient = dLoss_dOutput;

		// Compute flat gradient w.r.t input
		Eigen::VectorXd flatGrad = _weights.transpose() * dLoss_dOutput;

		// Restore gradient to match input shape
		return _restoreInputShape(flatGrad);
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

	template <typename T>
	void _getInputDimensions(const T& input) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			_inputChannels = 1;
			_inputHeight = input.size();
			_inputWidth = 1;
			_inputType = InputType::Vector;
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			_inputChannels = 1;
			_inputHeight = input.rows();
			_inputWidth = input.cols();
			_inputType = InputType::Matrix;
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			_inputChannels = input.size();
			_inputHeight = input[0].rows();
			_inputWidth = input[0].cols();
			_inputType = InputType::Tensor3D;
		}
		// Handle unsupported types
		else {
			throw std::invalid_argument("[FullyConnected]: Unsupported input type.");
		}
		// Check if the input size matches the expected size
		if (_inputSize != _inputChannels * _inputHeight * _inputWidth) {
			throw std::invalid_argument("[FullyConnected]: Input size does not match the expected size.");
		}
	}

	//template <typename T>
	Eigen::VectorXd _flattenInput(const T& input) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			return input;
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			return _flattenMatrixToVector(input);
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			return _flatten3DToVector(input);
		}
		// Handle unsupported types
		else {
			throw std::invalid_argument("[FullyConnected]: Unsupported input type.");
		}
	}

	Eigen::VectorXd _flattenMatrixToVector(const Eigen::MatrixXd& data) {
		// Flatten the input data
		Eigen::Map<const Eigen::VectorXd> map(data.data(), data.size());
		return map;
	}

	Eigen::VectorXd _flatten3DToVector(const std::vector<Eigen::MatrixXd>& data) {
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

	//template <typename T>
	auto _restoreInputShape(const Eigen::VectorXd& flat) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			if (flat.size() != _inputSize) {
				throw std::invalid_argument("[FullyConnected]: Size mismatch in unflattenVector.");
			}
			return flat;  // No reshape needed
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {// Restore to matrix
			if (flat.size() != _inputHeight * _inputWidth) {
				throw std::invalid_argument("Size mismatch in unflattening Matrix.");
			}
			return Eigen::Map<const Eigen::MatrixXd>(flat.data(), _inputHeight, _inputWidth);
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {// Restore to 3D tensor
			if (flat.size() != _inputChannels * _inputHeight * _inputWidth) {
				throw std::invalid_argument("Size mismatch in unflattening 3D Tensor.");
			}
			std::vector<Eigen::MatrixXd> unflat(_inputChannels, 
				Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
			size_t index = 0;
			// Unflatten input into 3D tensor
			for (size_t c = 0; c < _inputChannels; ++c)
				for (size_t h = 0; h < _inputHeight; ++h)
					for (size_t w = 0; w < _inputWidth; ++w) {
						unflat[c](h, w) = flat(index++);
					}
			return unflat;
		}
		else {
			throw std::invalid_argument("[FullyConnected]: Unsupported type for unflattening.");
		}
	}
};

	/*auto _restoreInputShape(const Eigen::VectorXd& flat) {
		if (_inputType == InputType::Vector) {
			if (flat.size() != _inputSize) {
				throw std::invalid_argument("[FullyConnected]: Size mismatch in unflattenVector.");
			}
			return flat;  // No reshape needed
		}
		else if (_inputType == InputType::Matrix) {// Restore to matrix
			if (flat.size() != _inputHeight * _inputWidth) {
				throw std::invalid_argument("Size mismatch in unflattening Matrix.");
			}
			return Eigen::Map<const Eigen::MatrixXd>(flat.data(), _inputHeight, _inputWidth);
		}
		else if (_inputType == InputType::Tensor3D) {// Restore to 3D tensor
			if (flat.size() != _inputChannels * _inputHeight * _inputWidth) {
				throw std::invalid_argument("Size mismatch in unflattening 3D Tensor.");
			}
			std::vector<Eigen::MatrixXd> unflat(_inputChannels, 
				Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
			size_t index = 0;
			// Unflatten input into 3D tensor
			for (size_t c = 0; c < _inputChannels; ++c)
				for (size_t h = 0; h < _inputHeight; ++h)
					for (size_t w = 0; w < _inputWidth; ++w) {
						unflat[c](h, w) = flat(index++);
					}
			return unflat;
		}
		else {
			throw std::invalid_argument("[FullyConnected]: Unsupported type for unflattening.");
		}
	}*/


/*#include <variant>
std::variant<Eigen::VectorXd, Eigen::MatrixXd, std::vector<Eigen::MatrixXd>>
_restoreInputShape(const Eigen::VectorXd& flat) {
	switch (_inputType) {
	case InputType::Vector:
		if (flat.size() != _inputSize)
			throw std::invalid_argument("Mismatch restoring Vector.");
		return flat;

	case InputType::Matrix:
		if (flat.size() != _inputHeight * _inputWidth)
			throw std::invalid_argument("Mismatch restoring Matrix.");
		return Eigen::Map<const Eigen::MatrixXd>(flat.data(), _inputHeight, _inputWidth);

	case InputType::Tensor3D: {
		if (flat.size() != _inputChannels * _inputHeight * _inputWidth)
			throw std::invalid_argument("Mismatch restoring 3D tensor.");
		std::vector<Eigen::MatrixXd> restored(_inputChannels, Eigen::MatrixXd(_inputHeight, _inputWidth));
		size_t index = 0;
		for (size_t c = 0; c < _inputChannels; ++c) {
			for (size_t h = 0; h < _inputHeight; ++h) {
				for (size_t w = 0; w < _inputWidth; ++w) {
					restored[c](h, w) = flat(index++);
				}
			}
		}
		return restored;
	}

	default:
		throw std::logic_error("Unknown input type during shape restore.");
	}
}*/


