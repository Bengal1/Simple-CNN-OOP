#pragma once

#include <iostream>
#include <random>
#include <memory>
#include <cassert>
#include "../Activation.h"
#include "../Optimizer.h"
#include "../Regularization.h"


class Convolution2D {
private:
	const int _inputHeight;
	const int _inputWidth;
	const int _inputChannels;
	const int _batchSize;

	const int _numFilters;
	const int _kernelSize;
	const int _stride;
	const int _padding;
	

	int _outputHeight;
	int _outputWidth;

	Eigen::MatrixXd _input;
	std::vector<Eigen::MatrixXd> _input3D;
	std::vector<Eigen::MatrixXd> _filters;
	std::vector<Eigen::MatrixXd> _filtersGradient;

	std::vector<std::vector<Eigen::MatrixXd>> _preActivationOutput;
	std::vector<std::vector<Eigen::MatrixXd>> _input3DBatch;
	std::vector<std::vector<Eigen::MatrixXd>> _filtersGradBatch;

	std::unique_ptr<AdamOptimizer> _optimizer;
	std::unique_ptr<Activation> _activation;
	//std::unique_ptr<ReLU> _activation;
	BatchNormalization _bn;

public:
	Convolution2D(int inputHeight, int inputWidth, int inputChannels,
		int numFilters, int kernelSize, 
		std::unique_ptr<Activation> activationFunction,
		int batchSize = 1, int stride = 1, int padding = 0)
		: _inputHeight(inputHeight), _inputWidth(inputWidth),
		_inputChannels(inputChannels), _numFilters(numFilters),
		_kernelSize(kernelSize), _batchSize(batchSize), _stride(stride),
		_padding(padding),
		_optimizer(std::make_unique<AdamOptimizer>(numFilters)),
		_activation(std::move(activationFunction))
	{
		if (_inputHeight <= 0 || _inputWidth <= 0) {
			throw std::invalid_argument("Input dimensions must be positive.");
		}
		if (_inputChannels <= 0) {
			throw std::invalid_argument("Input channels must be positive.");
		}
		if (_numFilters <= 0) {
			throw std::invalid_argument("Number of filters must be positive.");
		}
		if (_kernelSize <= 0) {
			throw std::invalid_argument("Kernel size must be positive.");
		}
		if (_stride <= 0) {
			throw std::invalid_argument("Stride must be positive.");
		}
		if (_batchSize <= 0) {
			throw std::invalid_argument("Batch size must be positive.");
		}
		if (_padding < 0) {
			throw std::invalid_argument("Padding must be non-negative.");
		}
		if (_inputHeight < _kernelSize || _inputWidth < _kernelSize) {
			throw std::invalid_argument("Input dimensions must be larger than kernel size.");
		}
		// Initialize filters and gradients
		_initializeFilters();
		
		
		_outputHeight = (_inputHeight - _kernelSize + 2 * _padding) / _stride + 1;
		_outputWidth = (_inputWidth - _kernelSize + 2 * _padding) / _stride + 1;
		
		if (_outputHeight <= 0 || _outputWidth <= 0) {
			throw std::invalid_argument("Output dimensions must be positive.");
		}

		if (_batchSize == 1) {
			if (_inputChannels == 1) {
				_input.resize(_inputHeight, _inputWidth);
			}
			else {
				_input3D.assign(_batchSize, Eigen::MatrixXd::Zero(_inputHeight,
					_inputWidth));
			}
			_filtersGradient.assign(_numFilters, Eigen::MatrixXd::Zero(_kernelSize,
				_kernelSize));
			_preActivationOutput.assign(1, std::vector<Eigen::MatrixXd>(_numFilters,
				Eigen::MatrixXd::Zero(_outputHeight, _outputWidth)));
		}
		else if (_batchSize > 1) {
			if (_inputChannels == 1) {
				_input3D.assign(_batchSize, Eigen::MatrixXd::Zero(_inputHeight,
					_inputWidth));
			}
			else {
				_input3DBatch.assign(_batchSize, std::vector<Eigen::MatrixXd>(
					_inputChannels, Eigen::MatrixXd::Zero(_inputHeight, _inputWidth)));
			}
			_filtersGradBatch.assign(_batchSize, std::vector<Eigen::MatrixXd>(
				_numFilters, Eigen::MatrixXd::Zero(_kernelSize, _kernelSize)));
			_preActivationOutput.assign(_batchSize, std::vector<Eigen::MatrixXd>(
				_numFilters, Eigen::MatrixXd::Zero(_outputHeight, _outputWidth)));
		}
	}

	std::vector<Eigen::MatrixXd> forward(const Eigen::MatrixXd& input,
										 const int batchNum = 0) 
	{
		std::vector<Eigen::MatrixXd> layerOutput(_numFilters,
			Eigen::MatrixXd::Zero(_outputHeight, _outputWidth));
		
		_input = input;
		for (int f = 0; f < _numFilters; f++) {
			_preActivationOutput[batchNum][f] += _Convolve2D(input, _filters[f], _padding);
		}

		std::vector<Eigen::MatrixXd> nornalizedOutput = _bn.forward(
			_preActivationOutput[batchNum]);
		layerOutput = _activation->Activate(nornalizedOutput);
		
		return layerOutput;
	}

	std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& multiInput,
										 const int batchNum = 0) 
	{ //Overload for multi-channel input
		std::vector<Eigen::MatrixXd> layerOutput(_numFilters,
			Eigen::MatrixXd::Zero(_outputHeight, _outputWidth));

		_input3D = multiInput;
		for (int f = 0; f < _numFilters; f++) {
			for (int c = 0; c < _inputChannels; c++) {
				_preActivationOutput[batchNum][f] += _Convolve2D(multiInput[c],
					_filters[f], _padding);
			}
		}

		std::vector<Eigen::MatrixXd> nornalizedOutput = _bn.forward(
			_preActivationOutput[batchNum]);
		layerOutput = _activation->Activate(nornalizedOutput);

		return layerOutput;
	}

	std::vector<std::vector<Eigen::MatrixXd>> forwardBatch(
					std::vector<Eigen::MatrixXd>&inputBatch) 
	{
		int batchSize = inputBatch.size();
		std::vector<std::vector<Eigen::MatrixXd>> outputBatch(batchSize,
			std::vector<Eigen::MatrixXd>(_numFilters, Eigen::MatrixXd::Zero(
				_outputHeight, _outputWidth)));

		_input3D = inputBatch;

		for (int b = 0; b < batchSize; b++) {
			outputBatch[b] = forward(_input3D[b], b);
		}

		return outputBatch;
	}

	std::vector<std::vector<Eigen::MatrixXd>> forwardBatch(std::vector<std::
										vector<Eigen::MatrixXd>>&inputBatch) 
	{
		int batchSize = inputBatch.size();
		std::vector<std::vector<Eigen::MatrixXd>> outputBatch(batchSize,
			std::vector<Eigen::MatrixXd>(_numFilters, Eigen::MatrixXd::Zero(
				_outputHeight, _outputWidth)));

		_input3DBatch = inputBatch;

		for (int b = 0; b < batchSize; b++) {
			outputBatch[b] = forward(_input3DBatch[b], b);
		}

		return outputBatch;
	}

	std::vector<Eigen::MatrixXd> backward(std::vector<Eigen::MatrixXd>& lossGradient,
										  const int batchNum = 0) 
	{
		std::vector<Eigen::MatrixXd> inputGradient(_inputChannels,
			Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

		if (lossGradient.size() != _numFilters) {
			throw std::invalid_argument("Loss gradient size must match number of filters.");
		}

		std::vector<Eigen::MatrixXd> dLoss_dPreActivation = 
			_activation->computeGradient(lossGradient, _preActivationOutput[batchNum]);
		std::vector<Eigen::MatrixXd> dLoss_dBN = _bn.backward(dLoss_dPreActivation);

		for (int c = 0; c < _inputChannels; c++) {
			for (int f = 0; f < _numFilters; f++) {

				// Calculate the gradient w.r.t the parameters
				if (_inputChannels == 1) {
					_filtersGradient[f] += _Convolve2D(_input, dLoss_dBN[f]);
				}
				else if (_inputChannels > 1) {
					/*std::cout << "input3D[" << c << "]: " << _input3D[c].rows() << "x" << _input3D[c].cols() << std::endl;
					std::cout << "dLoss_dBN[" << f << "]: " << dLoss_dBN[f].rows() << "x" << dLoss_dBN[f].cols() << std::endl;*/

					_filtersGradient[f] += _Convolve2D(_input3D[c],
													   dLoss_dBN[f]);
				}
				// Calculate the gradient w.r.t the input
				_calculateInputGradient(dLoss_dPreActivation[f], _filters[f],
						_preActivationOutput[batchNum][f], inputGradient[c]);
				_preActivationOutput[batchNum][f].setZero();
			}
		}

		updateParameters();

		return inputGradient;
	}

	std::vector<std::vector<Eigen::MatrixXd>> backwardBatch(
		std::vector<std::vector<Eigen::MatrixXd>>& lossGradientBatch) 
	{
		int batchSize = lossGradientBatch.size();
		std::vector<std::vector<Eigen::MatrixXd>> inputGradBatch(batchSize,
			std::vector<Eigen::MatrixXd>(_inputChannels, Eigen::MatrixXd::Zero(
				_outputHeight, _outputWidth)));

		for (int b = 0; b < batchSize; b++) {
			inputGradBatch[b] = backward(lossGradientBatch[b], b);
		}

		return inputGradBatch;
	}


	void updateParameters() 
	{
		for (int f = 0; f < _numFilters; f++) {
			_optimizer->updateStep(_filters[f], _filtersGradient[f], f);
			// Reset gradients
			_filtersGradient[f].setZero();
		}
		_bn.updateParameters();
	}

	void updateBatch() 
	{
		int batchSize = _filtersGradBatch.size();
		for (int b = 0; b < batchSize; b++) {
			for (int f = 0; f < _numFilters; f++) {
				_optimizer->updateStep(_filters[f], _filtersGradBatch[b][f], f);
				// Reset gradients
				_filtersGradBatch[b][f].setZero();
			}
		}
		_bn.updateParameters();
	}

	void SetTestMode() 
	{
		_filtersGradient.clear();
		_filtersGradBatch.clear();
		_bn.SetTestMode();
	}

	void SetTrainingMode() 
	{
		if (_batchSize == 1) {
			_filtersGradient.assign(_numFilters, Eigen::MatrixXd::Zero(_kernelSize,
				_kernelSize));
		}
		else if (_batchSize > 1) {
			_filtersGradBatch.assign(_batchSize, std::vector<Eigen::MatrixXd>(
				_numFilters, Eigen::MatrixXd::Zero(_kernelSize, _kernelSize)));
		}
		_bn.SetTrainingMode();
	}

private:
	void _initializeFilters() {
		_filters.assign(_numFilters, Eigen::MatrixXd::Zero(_kernelSize, _kernelSize));

		// Setup random number generator and distribution for He initialization
		std::random_device rd;
		std::mt19937 randomEngine(rd());
		std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 /
			(_inputHeight * _inputWidth * _inputChannels)));

		for (int f = 0; f < _numFilters; f++) {
			Eigen::MatrixXd filter(_kernelSize, _kernelSize);

			// Initialize filters
			for (int row = 0; row < _kernelSize; row++) {
				for (int col = 0; col < _kernelSize; col++) {
					filter(row, col) = distribution(randomEngine);
				}
			}
			_filters[f] = filter;
		}
	}

	Eigen::MatrixXd _padWithZeros(const Eigen::MatrixXd& input,
								  int promptPadding = 0) 
	{
		if (promptPadding < 0) {
			std::cerr << "Padding cannot be negative! pad = " << promptPadding 
															  << std::endl;
			exit(-1);
		}
		const int inputHeight = input.rows();
		const int inputWidth = input.cols();

		int padding = (promptPadding != 0) ? promptPadding : _padding;

		const int padInputHeight = inputHeight + 2 * padding;
		const int padInputWidth = inputWidth + 2 * padding;

		Eigen::MatrixXd paddedInput = Eigen::MatrixXd::Zero(padInputHeight,
															padInputWidth);

		paddedInput.block(padding, padding, inputHeight, inputWidth) = input;

		return paddedInput;
	}

	std::vector<Eigen::MatrixXd> _padWithZeros(const std::vector<Eigen::MatrixXd>& input,
											   int promptPadding = 0)
	{

		std::vector<Eigen::MatrixXd> paddedInput;
		paddedInput.reserve(input.size());

		for (const auto& channelMatrix : input) {
			paddedInput.emplace_back(_padWithZeros(channelMatrix, promptPadding));
		}

		return paddedInput;
	}

	Eigen::MatrixXd _Convolve2D(const Eigen::MatrixXd& input,
								const Eigen::MatrixXd& kernel,
								int padding = 0) 
	{
		const int inputHeight = input.rows();
		const int inputWidth = input.cols();
		const int filterHeight = kernel.rows();
		const int filterWidth = kernel.cols();

		Eigen::MatrixXd paddedInput = (padding > 0) ? _padWithZeros(input, padding) : input;

		const int paddedHeight = paddedInput.rows();
		const int paddedWidth = paddedInput.cols();

		const int outputHeight = (paddedHeight - filterHeight) / _stride + 1;
		const int outputWidth = (paddedWidth - filterWidth) / _stride + 1;

		Eigen::MatrixXd ConvolutionResult(outputHeight, outputWidth);

		for (int h = 0; h < outputHeight; ++h) {
			for (int w = 0; w < outputWidth; ++w) {
				ConvolutionResult(h, w) = (paddedInput.block(h * _stride, w * _stride,
								filterHeight, filterWidth).cwiseProduct(kernel)).sum();
			}
		}

		return ConvolutionResult;
	}


	void _calculateInputGradient(const Eigen::MatrixXd& lossGradientChannel,
								 const Eigen::MatrixXd& filter,
								 const Eigen::MatrixXd& preActivationOut,
								 Eigen::MatrixXd& inputGradChannel)
	{
		Eigen::MatrixXd dActivation = _activation->computeGradient(lossGradientChannel, preActivationOut);

		Eigen::MatrixXd reversedFilter = filter.colwise().reverse().rowwise().reverse();

		// Use the new generalized convolution function
		Eigen::MatrixXd inputGradient = _Convolve2D(dActivation, reversedFilter, _kernelSize - 1);

		inputGradChannel += inputGradient;
	}
};


//Eigen::MatrixXd _Convolve2D(const Eigen::MatrixXd& input,
//	const Eigen::MatrixXd& kernel) {
//	const int inputHeight = input.rows();
//	const int inputWidth = input.cols();
//	const int filterHeight = kernel.rows();
//	const int filterWidth = kernel.cols();
//	const int outputHeight = (inputHeight - filterHeight) / _stride + 1;
//	const int outputWidth = (inputWidth - filterWidth) / _stride + 1;

//	Eigen::MatrixXd ConvolutionResult(outputHeight, outputWidth);

//	for (int h = 0; h < outputHeight; h++) {
//		for (int w = 0; w < outputWidth; w++) {
//			ConvolutionResult(h, w) = (input.block(h * _stride, w * _stride,
//				filterHeight, filterWidth).cwiseProduct(kernel)).sum();
//		}
//	}

//	return ConvolutionResult;
//}

//void _calculateInputGradient(const Eigen::MatrixXd& lossGradientChannel,
//	const Eigen::MatrixXd& filter, Eigen::MatrixXd preActivationOut,
//	Eigen::MatrixXd& inputGradChannel) {
//	Eigen::MatrixXd dOutput_dInput = _activation->computeGradient(
//		lossGradientChannel, preActivationOut);
//	Eigen::MatrixXd reversedFilter = filter.reverse();
//
//	// Iterate through positions and calculate the input gradient
//	for (int i = 0; i < _outputHeight; i++) {
//		for (int j = 0; j < _outputWidth; j++) {
//			for (int k = 0; k < _kernelSize; k++) {
//				for (int l = 0; l < _kernelSize; l++) {
//					int inputRow = i + k;
//					int inputCol = j + l;
//
//					if (inputRow > 0 && inputRow < _inputHeight &&
//						inputCol > 0 && inputCol < _inputWidth) {
//						inputGradChannel(inputRow, inputCol) +=
//							dOutput_dInput(i, j) * filter(k, l);
//					}
//					else if (inputRow == 0 || inputRow == _inputHeight - 1 &&
//						inputCol == 0 && inputCol == _inputWidth - 1) {
//						inputGradChannel(inputRow, inputCol) +=
//							dOutput_dInput(i, j) * reversedFilter(k, l);
//					}
//				}
//			}
//		}
//	}
//}