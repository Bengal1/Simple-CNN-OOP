#pragma once

#include <iostream>
#include <random>
#include <memory>
#include <cassert>
#include "../Activation.h"
#include "../Optimizer.h"


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
	std::vector<std::vector<Eigen::MatrixXd>> _input3DBatch;
	std::vector<std::vector<Eigen::MatrixXd>> _filtersGradBatch;

	std::unique_ptr<AdamOptimizer> _optimizer;
	std::unique_ptr<ReLU> _activation;

public:
	Convolution2D(int inputHeight, int inputWidth, int inputChannels, 
		int numFilters, int kernelSize,int batchSize = 1, int stride = 1, 
		int padding = 0)
		: _inputHeight(inputHeight), _inputWidth(inputWidth), _inputChannels(inputChannels), 
		_numFilters(numFilters), _kernelSize(kernelSize), _batchSize(batchSize), 
		_stride(stride), _padding(padding),
		_optimizer(std::make_unique<AdamOptimizer>(numFilters)),
		_activation(std::make_unique<ReLU>())
	{
		initializeFilters();

		_outputHeight = (_inputHeight - _kernelSize + 2 * _padding) / _stride + 1;
		_outputWidth = (_inputWidth - _kernelSize + 2 * _padding) / _stride + 1;

		if (_batchSize == 1) {
			if (_inputChannels == 1) {
				_input.resize(_inputHeight, _inputWidth);
			}
			else {
				_input3D.assign(_batchSize, Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
			}
			_filtersGradient.assign(_numFilters, Eigen::MatrixXd::Zero(_kernelSize, _kernelSize));
			_preActivationOutput.assign(1, std::vector<Eigen::MatrixXd>(_numFilters,
				Eigen::MatrixXd::Zero(_outputHeight, _outputWidth)));
		}
		else if (_batchSize > 1) {
			if (_inputChannels == 1) {
				_input3D.assign(_batchSize, Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
			}
			else {
				_input3DBatch.assign(_batchSize, std::vector<Eigen::MatrixXd>(_inputChannels,
					Eigen::MatrixXd::Zero(_inputHeight, _inputWidth)));
			}
			_filtersGradBatch.assign(_batchSize, std::vector<Eigen::MatrixXd>(_numFilters,
				Eigen::MatrixXd::Zero(_kernelSize, _kernelSize)));
			_preActivationOutput.assign(_batchSize, std::vector<Eigen::MatrixXd>(_numFilters,
				Eigen::MatrixXd::Zero(_outputHeight, _outputWidth)));
		}
		else {
			std::cerr << "Non-positive Batch size is not valid! " << _batchSize << std::endl;
			//std::cerr << "Setting Batch size to 1. " << _batchSize << std::endl;
			//_batchSize = 1;
		}
	}

	std::vector<Eigen::MatrixXd> forward(Eigen::MatrixXd& input, int batchNum = 0) {
		std::vector<Eigen::MatrixXd> output(_numFilters,
			Eigen::MatrixXd::Zero(_outputHeight, _outputWidth));
		Eigen::MatrixXd preActivationOut = Eigen::MatrixXd::Zero
		(_outputHeight, _outputWidth);

		_input = input;

		for (int f = 0; f < _numFilters; ++f) {
			for (int c = 0; c < _inputChannels; ++c) {
				preActivationOut += Convolve2D(input, _filters[f]);
			}
			_preActivationOutput[batchNum][f] = preActivationOut;
			output[f] = _activation->activate(preActivationOut);
			preActivationOut.setZero();
		}

		return output;
	}

	std::vector<Eigen::MatrixXd> forward(std::vector<Eigen::MatrixXd>& multiInput, int batchNum = 0) {  //Overload for multi-channel input
		std::vector<Eigen::MatrixXd> output(_numFilters,
			Eigen::MatrixXd::Zero(_outputHeight, _outputWidth));
		Eigen::MatrixXd preActivationOut = Eigen::MatrixXd::Zero
		(_outputHeight, _outputWidth);
		_input3D = multiInput;
		_input.resize(0, 0);

		for (int f = 0; f < _numFilters; ++f) {
			for (int c = 0; c < _inputChannels; ++c) {
				preActivationOut += Convolve2D(multiInput[c], _filters[f]);
			}
			_preActivationOutput[batchNum][f] = preActivationOut;
			output[f] = _activation->activate(preActivationOut);
			preActivationOut.setZero();
		}

		return output;
	}

	std::vector<std::vector<Eigen::MatrixXd>> forwardBatch(std::vector<Eigen::MatrixXd>& inputBatch) {
		int batchSize = inputBatch.size();
		std::vector<std::vector<Eigen::MatrixXd>> outputBatch(batchSize,
			std::vector<Eigen::MatrixXd>(_numFilters, Eigen::MatrixXd::Zero(_outputHeight,
				_outputWidth)));

		_input3D = inputBatch;

		for (int b = 0; b < batchSize; b++) {
			outputBatch[b] = forward(_input3D[b], b);
		}

		return outputBatch;
	}

	std::vector<std::vector<Eigen::MatrixXd>> forwardBatch(std::vector<std::
		vector<Eigen::MatrixXd>>& inputBatch) {
		int batchSize = inputBatch.size();
		std::vector<std::vector<Eigen::MatrixXd>> outputBatch(batchSize, std::vector<Eigen::MatrixXd>(
			_numFilters,Eigen::MatrixXd::Zero(_outputHeight, _outputWidth)));

		_input3DBatch = inputBatch;

		for (int b = 0; b < batchSize; b++) {
			outputBatch[b] = forward(_input3DBatch[b], b);
		}

		return outputBatch;
	}

	std::vector<Eigen::MatrixXd> backward(std::vector<Eigen::MatrixXd>& lossGradient) {
		std::vector<Eigen::MatrixXd> inputGradient(_inputChannels,
			Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

		assert(lossGradient.size() == _numFilters);

		
		for (int c = 0; c < _inputChannels; c++) {
			for (int f = 0; f < _numFilters; f++) {
				Eigen::MatrixXd dLoss_dPreActivation = _activation->computeGradient(lossGradient[f], 
					_preActivationOutput[0][f]);
				// Calculate the gradient w.r.t the parameters
				if (_inputChannels == 1) {
					_filtersGradient[f] = Convolve2D(_input, dLoss_dPreActivation);
				}
				else if (_inputChannels > 1) {
					_filtersGradient[f] += Convolve2D(_input3D[c], dLoss_dPreActivation);
				}
				// Calculate the gradient w.r.t the input
				calculateInputGradient(lossGradient[f], _filters[f],
					_preActivationOutput[0][f], inputGradient[c]);
			}
		}

		return inputGradient;
	}

	std::vector<std::vector<Eigen::MatrixXd>> backwardBatch(std::vector<std::vector<Eigen::MatrixXd>>& lossGradientBatch) {
		int batchSize = lossGradientBatch.size();
		std::vector<std::vector<Eigen::MatrixXd>> inputGradBatch(batchSize, std::vector<Eigen::MatrixXd>(
			_inputChannels, Eigen::MatrixXd::Zero(_outputHeight, _outputWidth)));

		for (int b = 0; b < batchSize; b++) {
			for (int c = 0; c < _inputChannels; c++) {
				for (int f = 0; f < _numFilters; f++) {
					// Calculate the gradient w.r.t the parameters
					if (_inputChannels == 1) {
						_filtersGradient[f] += Convolve2D(_input, lossGradientBatch[b][f]);
					}
					else if (_inputChannels > 1) {
						_filtersGradient[f] += Convolve2D(_input3DBatch[b][c], lossGradientBatch[b][f]);
					}
					// Calculate the gradient w.r.t the input
					calculateInputGradient(lossGradientBatch[b][f], _filters[f],
						_preActivationOutput[b][f], inputGradBatch[b][c]);
				}
			}
		}

		return inputGradBatch;
	}


	void updateParameters() {
		for (int f = 0; f < _numFilters; f++) {
			_optimizer->updateStep(_filters[f], _filtersGradient[f], f);
			// Reset gradients
			_filtersGradient[f].setZero();
		}
	}

	void updateBatch() {
		int batchSize = _filtersGradBatch.size();
		for (int b = 0; b < batchSize; b++) {
			for (int f = 0; f < _numFilters; f++) {
				_optimizer->updateStep(_filters[f], _filtersGradBatch[b][f], f);
				// Reset gradients
				_filtersGradBatch[b][f].setZero();
			}
		}
	}

private:
	void initializeFilters() {
		_filters.assign(_numFilters, Eigen::MatrixXd::Zero(_kernelSize, _kernelSize));

		// Setup random number generator and distribution for He initialization
		std::random_device rd;
		std::mt19937 generator(rd());
		std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 / (_kernelSize * _kernelSize)));

		for (int i = 0; i < _numFilters; ++i) {
			Eigen::MatrixXd filter(_kernelSize, _kernelSize);

			// Initialize filters
			for (int row = 0; row < _kernelSize; row++) {
				for (int col = 0; col < _kernelSize; col++) {
					filter(row, col) = distribution(generator);
				}
			}
			_filters[i] = filter;
		}
	}

	Eigen::MatrixXd Convolve2D(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel) {
		const int inputHeight = input.rows();
		const int inputWidth = input.cols();
		const int filterHeight = kernel.rows();
		const int filterWidth = kernel.cols();
		const int outputHeight = (inputHeight - filterHeight) / _stride + 1;
		const int outputWidth = (inputWidth - filterWidth) / _stride + 1;

		Eigen::MatrixXd output(outputHeight, outputWidth);

		for (int h = 0; h < outputHeight; h++) {
			for (int w = 0; w < outputWidth; w++) {
				output(h, w) = (input.block(h * _stride, w * _stride, filterHeight, 
					filterWidth).cwiseProduct(kernel)).sum();
			}
		}

		return output;
	}

	void calculateInputGradient(const Eigen::MatrixXd& lossGradientChannel, const Eigen::MatrixXd& filter,
		Eigen::MatrixXd preActivationOut, Eigen::MatrixXd& inputGradChannel) {
		Eigen::MatrixXd dOutput_dInput = _activation->computeGradient(lossGradientChannel, preActivationOut);

		// Iterate through positions and calculate the input gradient
		for (int i = 0; i < _outputHeight; ++i) {
			for (int j = 0; j < _outputWidth; ++j) {
				for (int k = 0; k < _kernelSize; ++k) {
					for (int l = 0; l < _kernelSize; ++l) {
						int inputRow = i + k;
						int inputCol = j + l;

						if (inputRow >= 0 && inputRow < _inputHeight && inputCol >= 0 && inputCol < _inputWidth) {
							inputGradChannel(inputRow, inputCol) += dOutput_dInput(i, j) * filter(k, l);
						}
					}
				}
			}
		}
	}

	Eigen::MatrixXd padWithZeros(const Eigen::MatrixXd& input, int irregularPad = 0) {
		int padding;
		if (irregularPad)
			padding = irregularPad;
		else
			padding = _padding;

		int padInputHeight = _inputHeight + 2 * padding;
		int padInputWidth = _inputWidth + 2 * padding;
		Eigen::MatrixXd paddedInput = Eigen::MatrixXd::Zero(padInputHeight, padInputWidth);

		paddedInput.block(_padding, _padding, padInputHeight, padInputWidth) = input;


		return paddedInput;
	}

};
