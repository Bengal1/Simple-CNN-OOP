#pragma once

#include <iostream>
#include <random>
#include <memory>
#include "../Optimizer.hpp"
#include <variant>


class Convolution2D {
private:
	// Input dimensions
	const size_t _inputHeight;
	const size_t _inputWidth;
	const size_t _inputChannels;
	const size_t _batchSize;
	// Filter parameters
	const size_t _numFilters;
	const size_t _kernelSize;
	const size_t _stride;
	const size_t _padding;
	// Output dimensions
	size_t _outputHeight;
	size_t _outputWidth;
	// Biases
	Eigen::VectorXd _biases;
	Eigen::VectorXd _biasesGradient;
	// Input data
	Eigen::MatrixXd _input;
	std::vector<Eigen::MatrixXd> _input3D;
	// Filters and gradients
	std::vector<std::vector<Eigen::MatrixXd>> _filters;
	std::vector<std::vector<Eigen::MatrixXd>> _filtersGradient;
	// Optimizer
	std::unique_ptr<Optimizer> _optimizer;
public:
	Convolution2D(size_t inputHeight, size_t inputWidth, size_t inputChannels,
		size_t numFilters, size_t kernelSize, double maxGradNorm = -1.0,
		double weightDecay = 0.0, size_t batchSize = 1,
		size_t stride = 1, size_t padding = 0)
		: _inputHeight(inputHeight),
		_inputWidth(inputWidth),
		_inputChannels(inputChannels),
		_numFilters(numFilters),
		_kernelSize(kernelSize),
		_batchSize(batchSize),
		_stride(stride),
		_padding(padding),
		_optimizer(std::make_unique<AdamOptimizer>(
			static_cast<int>(numFilters), maxGradNorm, weightDecay))
	{
		// Validate input
		_validateInputParameters();
		// Initialize filters, variables and gradients
		_initialize();
	}

	template<typename T>
	std::vector<Eigen::MatrixXd> forward(const T& input) {

		std::vector<Eigen::MatrixXd> convOutput(_numFilters,
			Eigen::MatrixXd::Zero(_outputHeight, _outputWidth));

		if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			// Single input, single channel
			if (input.rows() != _inputHeight || input.cols() != _inputWidth) {
				throw std::invalid_argument("[Convolution2D]: Input dimensions do not match.");
			}
			_input = input;
			for (size_t f = 0; f < _numFilters; ++f) {
				convOutput[f] += _Convolve2D(input, _filters[f][0], _padding);

				convOutput[f].array() += _biases[f];
			}
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			// Multi-channel input
			if (input.size() != _inputChannels) {
				throw std::invalid_argument("[Convolution2D]: Input channels do not match.");
			}
			if (input[0].rows() != _inputHeight || input[0].cols() != _inputWidth) {
				throw std::invalid_argument("[Convolution2D]: Input dimensions do not match.");
			}
			_input3D = input;
			for (size_t f = 0; f < _numFilters; ++f) {
				for (size_t c = 0; c < _inputChannels; ++c) {
					convOutput[f] += _Convolve2D(input[c],
						_filters[f][c], _padding);
				}
				convOutput[f].array() += _biases[f];
			}

		}
		else {
			throw std::invalid_argument("[Convolution2D]: Unsupported input type.");
		}

		return convOutput;
	}


	auto backward(const std::vector<Eigen::MatrixXd>& dLoss_dOutput) {
		if (dLoss_dOutput.size() != _numFilters) {
			throw std::invalid_argument("[Convolution2D]: Loss gradient size must match number of filters.");
		}

		std::vector<Eigen::MatrixXd> dLoss_dInput(_inputChannels,
			Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

		for (size_t f = 0; f < _numFilters; ++f) {
			// Gradient w.r.t biases
			_biasesGradient(f) += dLoss_dOutput[f].sum();

			for (size_t c = 0; c < _inputChannels; ++c) {
				Eigen::MatrixXd dLoss_dOutput_reversed = dLoss_dOutput[f].colwise().reverse().rowwise().reverse();
				// Gradient w.r.t. filters
				if (_inputChannels == 1) {
					_filtersGradient[f][c] += _Convolve2D(_input, dLoss_dOutput_reversed);
				}
				else if (_inputChannels > 1) {
					_filtersGradient[f][c] += _Convolve2D(_input3D[c], dLoss_dOutput_reversed);
				}
				else {
					throw std::invalid_argument("[Convolution2D]: Invalid number of input channels.");
				}
				// Reverse filter for dL/dIn[f] computation
				Eigen::MatrixXd reversedFilter = _filters[f][c].colwise().reverse().rowwise().reverse();

				// Gradient w.r.t input 
				dLoss_dInput[c] += _Convolve2D(dLoss_dOutput[f], reversedFilter, _kernelSize - 1);
			}
		}
		

		return dLoss_dInput;
	}


	void updateParameters()
	{
		// Update filters
		for (int f = 0; f < _numFilters; ++f) {
			//for (size_t c = 0; c < _inputChannels; ++c) {

			_optimizer->updateStep(_filters[f], _filtersGradient[f], f);
			// Reset filter's gradient
			for (size_t c = 0; c < _inputChannels; ++c)
				_filtersGradient[f][c].setZero();
			//}
		}
		// Update biases
		_optimizer->updateStep(_biases, _biasesGradient);
		_biasesGradient.setZero();
	}

	std::vector<std::vector<Eigen::MatrixXd>> getFilters() {
		return _filters;
	}

	Eigen::VectorXd getBiases() {
		return _biases;
	}

private:
	void _initializeFilters() {
		_filters.assign(_numFilters,std::vector<Eigen::MatrixXd>(
			_inputChannels, Eigen::MatrixXd::Zero(_kernelSize, _kernelSize)));

		// Setup random number generator and distribution for He initialization
		std::random_device rd;
		std::mt19937 randomEngine(rd());
		std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 /
			(_kernelSize * _kernelSize * _inputChannels)));

		for (size_t f = 0; f < _numFilters; ++f) {
			for (size_t c = 0; c < _inputChannels; ++c) {
				_filters[f][c] = Eigen::MatrixXd::NullaryExpr(_kernelSize, _kernelSize, [&]() {
					return distribution(randomEngine);
					});
			}
		}

		_biases = Eigen::VectorXd::Zero(_numFilters);
	}

	void _initialize() {
		// Calculate output dimensions
		size_t outHeight = (_inputHeight - _kernelSize + 2 * _padding) / _stride + 1;
		size_t outWidth = (_inputWidth - _kernelSize + 2 * _padding) / _stride + 1;

		if (outHeight == 0 || outWidth == 0) {
			throw std::invalid_argument("[Convolution2D]: Output dimensions must be positive.");
		}
		_outputHeight = outHeight;
		_outputWidth = outWidth;
		// Initialize input structure
		if (_batchSize == 1 && _inputChannels == 1) {
			// Single input, single channel
			_input.resize(_inputHeight, _inputWidth);
		}
		else if (_inputChannels == 1 || _batchSize == 1) {
			// Single-channel batch OR single multi-channel input
			_input3D.assign(std::max(_batchSize, _inputChannels),
				Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
		}
		else {
			// Batch of multi-channel inputs
		}
		// Initialize gradients
		if (_batchSize == 1) {
			_filtersGradient.assign(_numFilters, std::vector<Eigen::MatrixXd>(
				_inputChannels,Eigen::MatrixXd::Zero(_kernelSize, _kernelSize)));
			_biasesGradient = Eigen::VectorXd::Zero(_numFilters);
		}
		else {
			// init batch parmeters
		}
		// Initialize filters and biases
		_initializeFilters();
	}

	void _validateInputParameters() const {
		if (_inputHeight == 0 || _inputWidth == 0) {
			throw std::invalid_argument("[Convolution2D]: Input dimensions must be positive.");
		}
		if (_inputChannels == 0) {
			throw std::invalid_argument("[Convolution2D]: Input channels must be positive.");
		}
		if (_numFilters == 0) {
			throw std::invalid_argument("[Convolution2D]: Number of filters must be positive.");
		}
		if (_kernelSize == 0) {
			throw std::invalid_argument("[Convolution2D]: Kernel size must be positive.");
		}
		if (_stride == 0) {
			throw std::invalid_argument("[Convolution2D]: Stride must be positive.");
		}
		if (_batchSize == 0) {
			throw std::invalid_argument("[Convolution2D]: Batch size must be positive.");
		}
		if (_inputHeight < _kernelSize || _inputWidth < _kernelSize) {
			throw std::invalid_argument("[Convolution2D]: Input dimensions must be larger than kernel size.");
		}
	}

	const Eigen::MatrixXd _padWithZeros(const Eigen::MatrixXd& input,
		const size_t promptPadding = 0) const
	{
		const size_t inputHeight = input.rows();
		const size_t inputWidth = input.cols();

		size_t padding = (promptPadding != 0) ? promptPadding : _padding;

		const size_t padInputHeight = inputHeight + 2 * padding;
		const size_t padInputWidth = inputWidth + 2 * padding;

		Eigen::MatrixXd paddedInput = Eigen::MatrixXd::Zero(padInputHeight,
			padInputWidth);

		paddedInput.block(padding, padding, inputHeight, inputWidth) = input;

		return paddedInput;
	}

	const std::vector<Eigen::MatrixXd> _padWithZeros(const std::vector<Eigen::MatrixXd>& input,
		const size_t promptPadding = 0) const
	{// Overloaded function for 3D input
		std::vector<Eigen::MatrixXd> paddedInput;
		paddedInput.reserve(input.size());

		for (const auto& channelMatrix : input) {
			paddedInput.emplace_back(_padWithZeros(channelMatrix, promptPadding));
		}

		return paddedInput;
	}

	Eigen::MatrixXd _Convolve2D(const Eigen::MatrixXd& input,
		const Eigen::MatrixXd& kernel,
		size_t padding = 0) const
	{
		const size_t inputHeight = input.rows();
		const size_t inputWidth = input.cols();
		const size_t filterHeight = kernel.rows();
		const size_t filterWidth = kernel.cols();

		// Check if kernel size is valid
		if (filterHeight > inputHeight || filterWidth > inputWidth) {
			throw std::invalid_argument("[Convolution2D]: Kernel size must be smaller than input size.");
		}

		// Pad input with zeros if padding is specified
		Eigen::MatrixXd paddedInput = (padding > 0) ? _padWithZeros(input, padding) : input;

		const size_t paddedHeight = inputHeight + 2 * padding;
		const size_t paddedWidth = inputWidth + 2 * padding;

		const size_t outputHeight = (paddedHeight - filterHeight) / _stride + 1;
		const size_t outputWidth = (paddedWidth - filterWidth) / _stride + 1;

		Eigen::MatrixXd ConvolutionResult = Eigen::MatrixXd::Zero(outputHeight, outputWidth);
		// Perform convolution
		for (size_t h = 0; h < outputHeight; ++h) {
			for (size_t w = 0; w < outputWidth; ++w) {
				ConvolutionResult(h, w) = (paddedInput.block(h * _stride, w * _stride,
					filterHeight, filterWidth).cwiseProduct(kernel)).sum();
			}
		}

		return ConvolutionResult;
	}
};
