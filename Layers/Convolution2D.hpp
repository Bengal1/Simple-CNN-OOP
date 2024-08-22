#ifndef CONVOLUTION2D_HPP
#define CONVOLUTION2D_HPP

#include <iostream>
#include <random>
#include <memory>
#include "../Activation.hpp"
#include "../Optimizer.hpp"
#include "../Regularization.hpp"
#include <optional>


class Convolution2D {
private:
	const size_t _inputHeight;
	const size_t _inputWidth;
	const size_t _inputChannels;
	const size_t _numFilters;
	const size_t _kernelSize;

	const size_t _stride;
	const size_t _padding;

	size_t _outputHeight;
	size_t _outputWidth;

	std::vector<Eigen::MatrixXd> _input;
	std::vector<Eigen::MatrixXd> _normalizedConvOutput;

	std::vector<Eigen::MatrixXd> _filters;
	std::vector<Eigen::MatrixXd> _filtersGradient;
	Eigen::VectorXd _biases;
	Eigen::VectorXd _biasesGradient;

	std::unique_ptr<AdamOptimizer> _optimizer;
	std::unique_ptr<ReLU> _activation;
	std::optional<BatchNormalization> _bn;

public:
	Convolution2D(size_t inputHeight, size_t inputWidth, size_t inputChannels,
				  size_t numFilters, size_t kernelSize,  size_t stride = 1, 
				  size_t padding = 0)
		: _inputHeight(inputHeight), 
		  _inputWidth(inputWidth),
		  _inputChannels(inputChannels), 
		  _numFilters(numFilters),
		  _kernelSize(kernelSize),
		  _stride(stride), 
		  _padding(padding),
		  _optimizer(std::make_unique<AdamOptimizer>(numFilters)),
		  _activation(std::make_unique<ReLU>())
	{
		_initializeFilters();

		_outputHeight = (_inputHeight - _kernelSize + 2 * _padding) / _stride + 1;
		_outputWidth = (_inputWidth - _kernelSize + 2 * _padding) / _stride + 1;

		_input.assign(_inputChannels, 
					  Eigen::MatrixXd::Zero(_inputHeight,_inputWidth));
		_filtersGradient.assign(_numFilters, 
						Eigen::MatrixXd::Zero(_kernelSize, _kernelSize));
		_normalizedConvOutput.assign(_numFilters,
						Eigen::MatrixXd::Zero(_outputHeight, _outputWidth));

		_biases = Eigen::VectorXd::Zero(_numFilters);
        _biasesGradient = Eigen::VectorXd::Zero(_numFilters);

		_bn.emplace(_numFilters,_outputHeight, _outputWidth);
	}

	std::vector<Eigen::MatrixXd> forward(const Eigen::MatrixXd& input)
	{
		std::vector<Eigen::MatrixXd> convolutionOutput(_numFilters,
			Eigen::MatrixXd::Zero(_outputHeight, _outputWidth));
		
		_input[0] = _padWithZeros(input);
		
		for (size_t f = 0; f < _numFilters; ++f) {
			convolutionOutput[f] += _convolve2D(input, _filters[f]);
			(convolutionOutput[f].array() += _biases(f)).matrix();
		}
		
		_normalizedConvOutput = _bn->forward(convolutionOutput);
		return _activation->Activate(_normalizedConvOutput);
	}

	std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input3D) 
	{ //Overload for multi-channel input
		std::vector<Eigen::MatrixXd> convolutionOutput(_numFilters,
					Eigen::MatrixXd::Zero(_outputHeight, _outputWidth));

		_input = _padWithZeros(input3D);
		
		for (size_t f = 0; f < _numFilters; ++f) {
			for (size_t c = 0; c < _inputChannels; ++c) {
				convolutionOutput[f] += _convolve2D(input3D[c], _filters[f]);
			}
			(convolutionOutput[f].array() += _biases(f)).matrix();
		}

		_normalizedConvOutput = _bn->forward(convolutionOutput);
		return _activation->Activate(_normalizedConvOutput);
	}

	std::vector<Eigen::MatrixXd> backward(std::vector<Eigen::MatrixXd>& lossGradient)
	{
		std::vector<Eigen::MatrixXd> inputGradient(_inputChannels,
					Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

		assert(lossGradient.size() == _numFilters);
		
		std::vector<Eigen::MatrixXd> dLoss_dNormalizedOutput = _activation->computeGradient(
														lossGradient, _normalizedConvOutput);

		std::vector<Eigen::MatrixXd> dLoss_dConvOutput = _bn->backward(dLoss_dNormalizedOutput);
		
		for (size_t c = 0; c < _inputChannels; ++c) {
			for (size_t f = 0; f < _numFilters; ++f) {
				// Calculate the gradient w.r.t the parameters
				_filtersGradient[f] += _convolve2D(_input[c], dLoss_dConvOutput[f]);
				// Calculate the gradient w.r.t the input
				inputGradient[c] = _calculateInputGradient(dLoss_dConvOutput[f], _filters[f]); 
				// Compute bias gradients
				_biasesGradient(f) += dLoss_dConvOutput[f].sum();
				_normalizedConvOutput[f].setZero();
			}
		}
		_updateParameters();
			
		return inputGradient;
	}

	void setTestMode()
	{
		_filtersGradient.clear();
		_bn->SetTestMode();
	}

	void setTrainingMode()
	{
		_filtersGradient.assign(_numFilters, 
			Eigen::MatrixXd::Zero(_kernelSize, _kernelSize));
		_bn->SetTrainingMode();
	}

private:
	void _initializeFilters() 
	{
		_filters.assign(_numFilters, 
				 Eigen::MatrixXd::Zero(_kernelSize, _kernelSize));

		// Setup random number generator and distribution for He initialization
		std::random_device rd;
		std::mt19937 randomEngine(rd());
		std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 /
			(_inputHeight * _inputWidth * _inputChannels)));

		for (size_t f = 0; f < _numFilters; ++f) {
			Eigen::MatrixXd initializedKernel(_kernelSize, _kernelSize);

			// Initialize filters
			for (size_t row = 0; row < _kernelSize; ++row) {
				for (size_t col = 0; col < _kernelSize; ++col) {
					initializedKernel(row, col) = distribution(randomEngine);
				}
			}
			_filters[f] = initializedKernel;
		}
	}

	Eigen::MatrixXd _convolve2D(const Eigen::MatrixXd& input,
		const Eigen::MatrixXd& kernel) const
	{
		const size_t inputHeight = input.rows();
		const size_t inputWidth = input.cols();
		const size_t filterHeight = kernel.rows();
		const size_t filterWidth = kernel.cols();

		assert(inputHeight > filterHeight and inputWidth > filterWidth);
		
		const size_t outputHeight = (inputHeight - filterHeight) / _stride + 1;
		const size_t outputWidth = (inputWidth - filterWidth) / _stride + 1;

		Eigen::MatrixXd ConvolutionResult(outputHeight, outputWidth);
		ConvolutionResult.setZero();
		Eigen::MatrixXd reversedKernel = kernel.reverse();

		for (size_t h = 0; h < outputHeight; ++h) {
			for (size_t w = 0; w < outputWidth; ++w) {
				ConvolutionResult(h, w) = (input.block(h * _stride, w * _stride,
					filterHeight, filterWidth).cwiseProduct(reversedKernel)).sum();
			}
		}

		return ConvolutionResult;
	}

	void _updateParameters()
	{
		_bn->updateParameters();

		for (size_t f = 0; f < _numFilters; ++f) {
			_optimizer->updateStep(_filters[f], _filtersGradient[f], f);
			// Reset gradients
			_filtersGradient[f].setZero();
		}
		_optimizer->updateStep(_biases, _biasesGradient);
		_biasesGradient.setZero();
	}

	Eigen::MatrixXd _calculateInputGradient(const Eigen::MatrixXd& dLoss_dPreActivation,
		const Eigen::MatrixXd& filter) 
	{
		Eigen::MatrixXd dLoss_dInput(_inputHeight, _inputWidth);
		Eigen::MatrixXd reversedKernel = filter.reverse();

		// Zero-pad dLoss_dPreActivation to match dLoss_dInput size
    	size_t paddedHeight = dLoss_dPreActivation.rows() + _kernelSize - 1;
    	size_t paddedWidth = dLoss_dPreActivation.cols() + _kernelSize - 1;
    	
		Eigen::MatrixXd padded_dLoss = Eigen::MatrixXd::Zero(paddedHeight, paddedWidth);
    	padded_dLoss.block((_kernelSize - 1) / 2, (_kernelSize - 1) / 2, 
			dLoss_dPreActivation.rows(), dLoss_dPreActivation.cols()) = dLoss_dPreActivation;

		// convolve2D(dL/dPre-activation, rot180(kernel))
		for (size_t h = 0; h < _outputHeight; ++h) {
            for (size_t w = 0; w < _outputWidth; ++w) {
				dLoss_dInput(h, w) += (padded_dLoss.block(h, w, _kernelSize, _kernelSize).cwiseProduct(reversedKernel)).sum();
        	}
        }

		return dLoss_dInput;
	}

	Eigen::MatrixXd _padWithZeros(const Eigen::MatrixXd& input) const 
	{	
		if(_padding == 0){
			return input;
		}
		if( _padding < 0){
			throw std::invalid_argument("Padding must be non-negative");
		}

		size_t padInputHeight = _inputHeight + 2 * _padding;
		size_t padInputWidth = _inputWidth + 2 * _padding;
		Eigen::MatrixXd paddedInput = Eigen::MatrixXd::Zero(padInputHeight,
			padInputWidth);

		paddedInput.block(_padding, _padding, padInputHeight, padInputWidth) 
						= input;


		return paddedInput;
	}

	std::vector<Eigen::MatrixXd> _padWithZeros(const std::vector<
		Eigen::MatrixXd>& input) const
	{
		if(_padding == 0){
			return input;
		}
		if( _padding < 0){
			throw std::invalid_argument("Padding must be non-negative");
		}

		size_t padInputHeight = _inputHeight + 2 * _padding;
		size_t padInputWidth = _inputWidth + 2 * _padding;
		std::vector<Eigen::MatrixXd> paddedInput(_inputChannels,
			Eigen::MatrixXd::Zero(padInputHeight, padInputWidth));

		for (size_t c = 0; c < _inputChannels; ++c) {
			paddedInput[c].block(_padding, _padding, padInputHeight,
						padInputWidth) = input[c];
		}

		return paddedInput;
	}

};

#endif // CONVOLUTION2D_HPP