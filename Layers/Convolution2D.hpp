#pragma once

#include <iostream>
#include <random>
#include <memory>
#include <cassert>
#include "../Activation.hpp"
#include "../Optimizer.hpp"
#include "../Regularization.hpp"


class Convolution2D {
private:
	const int _inputHeight;
	const int _inputWidth;
	const int _inputChannels;
	const int _numFilters;
	const int _kernelSize;
	const int _stride;
	const int _padding;
	const int _batchSize;

	int _outputHeight;
	int _outputWidth;

	Eigen::MatrixXd _input;
	std::vector<Eigen::MatrixXd> _input3D;
	std::vector<Eigen::MatrixXd> _filters;
	std::vector<Eigen::MatrixXd> _filtersGradient;

	std::vector<std::vector<Eigen::MatrixXd>> _preActivationOutput;

	std::unique_ptr<AdamOptimizer> _optimizer;
	std::unique_ptr<ReLU> _activation;
	BatchNormalization _bn;

public:
	Convolution2D(int inputHeight, int inputWidth, int inputChannels,
		int numFilters, int kernelSize, int batchSize = 1, int stride = 1,
		int padding = 0)
		: _inputHeight(inputHeight), _inputWidth(inputWidth),
		_inputChannels(inputChannels), _numFilters(numFilters),
		_kernelSize(kernelSize), _batchSize(batchSize), _stride(stride),
		_padding(padding),
		_optimizer(std::make_unique<AdamOptimizer>(numFilters)),
		_activation(std::make_unique<ReLU>())
	{
		_initializeFilters();

		_outputHeight = (_inputHeight - _kernelSize + 2 * _padding) / _stride + 1;
		_outputWidth = (_inputWidth - _kernelSize + 2 * _padding) / _stride + 1;

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
		else {
			std::cerr << "Non-positive Batch size is not valid! " << _batchSize
				<< std::endl;
			exit(-1);
		}
	}

	std::vector<Eigen::MatrixXd> forward(const Eigen::MatrixXd& input,
		const int batchNum = 0)
	{
		std::vector<Eigen::MatrixXd> layerOutput(_numFilters,
			Eigen::MatrixXd::Zero(_outputHeight, _outputWidth));

		if (!_padding) {
			_input = input;
		}
		else if (_padding > 0) {
			_input = _padWithZeros(input);
		}
		else {
			std::cerr << "Non-positive padding value is not valid! " << _padding
				<< std::endl;
			exit(-1);
		}
		for (int f = 0; f < _numFilters; f++) {
			_preActivationOutput[batchNum][f] += _Convolve2D(input, _filters[f]);
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

		if (!_padding) {
			_input3D = multiInput;
		}
		else if (_padding > 0) {
			_input3D = _padWithZeros(multiInput);
		}
		else {
			std::cerr << "Non-positive padding value is not valid! " << _padding
				<< std::endl;
			exit(1);
		}

		for (int f = 0; f < _numFilters; f++) {
			for (int c = 0; c < _inputChannels; c++) {
				_preActivationOutput[batchNum][f] += _Convolve2D(multiInput[c],
					_filters[f]);
			}
		}

		std::vector<Eigen::MatrixXd> nornalizedOutput = _bn.forward(
			_preActivationOutput[batchNum]);
		layerOutput = _activation->Activate(nornalizedOutput);

		return layerOutput;
	}

	std::vector<Eigen::MatrixXd> backward(std::vector<Eigen::MatrixXd>& lossGradient,
		const int batchNum = 0)
	{
		std::vector<Eigen::MatrixXd> inputGradient(_inputChannels,
			Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

		assert(lossGradient.size() == _numFilters);

		std::vector<Eigen::MatrixXd> dLoss_dPreActivation = _activation->computeGradient(
			lossGradient, _preActivationOutput[batchNum]);
		std::vector<Eigen::MatrixXd> dLoss_dBN = _bn.backward(dLoss_dPreActivation);

		for (int c = 0; c < _inputChannels; c++) {
			for (int f = 0; f < _numFilters; f++) {

				// Calculate the gradient w.r.t the parameters
				if (_inputChannels == 1) {
					_filtersGradient[f] += _Convolve2D(_input, dLoss_dBN[f]);
				}
				else if (_inputChannels > 1) {
					_filtersGradient[f] += _Convolve2D(_input3D[c],
						dLoss_dBN[f]);
				}
				// Calculate the gradient w.r.t the input
				_calculateInputGradient(dLoss_dPreActivation[f], _filters[f],
					_preActivationOutput[batchNum][f], inputGradient[c]);
				_preActivationOutput[batchNum][f].setZero();
			}
		}
		_updateParameters();

		return inputGradient;
	}

	void SetTestMode()
	{
		_filtersGradient.clear();
		_bn.SetTestMode();
	}

	void SetTrainingMode()
	{
		_filtersGradient.assign(_numFilters, Eigen::MatrixXd::Zero(_kernelSize,
				_kernelSize));
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

	Eigen::MatrixXd _Convolve2D(const Eigen::MatrixXd& input,
		const Eigen::MatrixXd& kernel) const{
		const int inputHeight = input.rows();
		const int inputWidth = input.cols();
		const int filterHeight = kernel.rows();
		const int filterWidth = kernel.cols();
		const int outputHeight = (inputHeight - filterHeight) / _stride + 1;
		const int outputWidth = (inputWidth - filterWidth) / _stride + 1;

		Eigen::MatrixXd ConvolutionResult(outputHeight, outputWidth);

		for (int h = 0; h < outputHeight; h++) {
			for (int w = 0; w < outputWidth; w++) {
				ConvolutionResult(h, w) = (input.block(h * _stride, w * _stride,
					filterHeight, filterWidth).cwiseProduct(kernel)).sum();
			}
		}

		return ConvolutionResult;
	}


	void _updateParameters()
	{
		for (int f = 0; f < _numFilters; f++) {
			_optimizer->updateStep(_filters[f], _filtersGradient[f], f);
			// Reset gradients
			_filtersGradient[f].setZero();
		}
		_bn.updateParameters();
	}

	void _calculateInputGradient(const Eigen::MatrixXd& lossGradientChannel,
		const Eigen::MatrixXd& filter, Eigen::MatrixXd preActivationOut,
		Eigen::MatrixXd& inputGradChannel) {
		Eigen::MatrixXd dOutput_dInput = _activation->computeGradient(
			lossGradientChannel, preActivationOut);
		Eigen::MatrixXd reversedFilter = filter.reverse();

		// Iterate through positions and calculate the input gradient
		for (int i = 0; i < _outputHeight; i++) {
			for (int j = 0; j < _outputWidth; j++) {
				for (int k = 0; k < _kernelSize; k++) {
					for (int l = 0; l < _kernelSize; l++) {
						int inputRow = i + k;
						int inputCol = j + l;

						if (inputRow > 0 && inputRow < _inputHeight &&
							inputCol > 0 && inputCol < _inputWidth) {
							inputGradChannel(inputRow, inputCol) +=
								dOutput_dInput(i, j) * filter(k, l);
						}
						else if (inputRow == 0 || inputRow == _inputHeight - 1 &&
							inputCol == 0 && inputCol == _inputWidth - 1) {
							inputGradChannel(inputRow, inputCol) +=
								dOutput_dInput(i, j) * reversedFilter(k, l);
						}
					}
				}
			}
		}
	}

	Eigen::MatrixXd _padWithZeros(const Eigen::MatrixXd& input,
		int promptPadding = 0) const {
		int padding;
		if (promptPadding)
			padding = promptPadding;
		else
			padding = _padding;

		int padInputHeight = _inputHeight + 2 * padding;
		int padInputWidth = _inputWidth + 2 * padding;
		Eigen::MatrixXd paddedInput = Eigen::MatrixXd::Zero(padInputHeight,
			padInputWidth);

		paddedInput.block(_padding, _padding, padInputHeight,
			padInputWidth) = input;


		return paddedInput;
	}

	std::vector<Eigen::MatrixXd> _padWithZeros(const std::vector<
		Eigen::MatrixXd>& input, int promptPadding = 0) const
	{
		int padding = 0;

		if (promptPadding)
			padding = promptPadding;
		else
			padding = _padding;

		int padInputHeight = _inputHeight + 2 * padding;
		int padInputWidth = _inputWidth + 2 * padding;
		std::vector<Eigen::MatrixXd> paddedInput(_inputChannels,
			Eigen::MatrixXd::Zero(padInputHeight, padInputWidth));

		for (int c = 0; c < _inputChannels; c++) {
			paddedInput[c].block(_padding, _padding, padInputHeight,
				padInputWidth) = input[c];
		}

		return paddedInput;
	}

};
