#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>


class MaxPooling {
private:
    const int _inputHeight;
    const int _inputWidth;
    const int _inputChannels;

    const int _kernelSize;
    const int _stride;
    const int _batchSize;

    int _outputHeight;
    int _outputWidth;

    int _mapIndex;

    std::vector<Eigen::MatrixXd> _output;
    std::vector<std::vector<Eigen::MatrixXd>> _outputBatch;

    std::vector<std::tuple<int, int, int>> _inputGradientMap;
    std::vector<std::tuple<int, int, int, int>> _inputGradientMapBatch;

public:
    MaxPooling(int inputHeight, int inputWidth, int inputChannels, 
        int poolSize, int batchSize = 1, int stride = 2)
        : _inputHeight(inputHeight), _inputWidth(inputWidth), 
        _inputChannels(inputChannels), _kernelSize(poolSize),
		_stride(stride), _batchSize(batchSize), _mapIndex(0)
    {
		if (_kernelSize <= 0) {
			throw std::invalid_argument("Kernel size must be positive.");
		}
		if (_stride <= 0) {
			throw std::invalid_argument("Stride must be positive.");
		}
		if (_batchSize <= 0) {
			throw std::invalid_argument("Batch size must be positive.");
		}
		if (_inputHeight <= 0 || _inputWidth <= 0) {
			throw std::invalid_argument("Input dimensions must be positive.");
		}
		if (_inputChannels <= 0) {
			throw std::invalid_argument("Input channels must be positive.");
		}
		if (_inputHeight < _kernelSize || _inputWidth < _kernelSize) {
			throw std::invalid_argument("Input dimensions must be larger than kernel size.");
		}

        _outputHeight = (_inputHeight - _kernelSize) / _stride + 1;
        _outputWidth = (_inputWidth - _kernelSize) / _stride + 1;

        if (_batchSize == 1) {
            _output.assign(_inputChannels, Eigen::MatrixXd::Zero(
                _outputHeight, _outputWidth));
            _inputGradientMap.reserve(_inputChannels * _outputHeight * 
                _outputWidth);
        }
        else if (_batchSize > 1) {
            _outputBatch.assign(_batchSize, std::vector<Eigen::MatrixXd>(_inputChannels,
                Eigen::MatrixXd::Zero(_outputHeight, _outputWidth)));
            _inputGradientMapBatch.reserve(_batchSize * _inputChannels * 
                _outputHeight * _outputWidth);
        }
    }

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) 
    {
        _inputGradientMap.clear();
        _mapIndex = 0;

        Eigen::Index row, col;

        for (int c = 0; c < _inputChannels; c++) {
            for (int h = 0; h < _outputHeight; h++) {
                for (int w = 0; w < _outputWidth; w++) {
                    _output[c](h, w) = (input[c].block(h * _stride, w * _stride, 
                                        _kernelSize, _kernelSize)).maxCoeff(&row, &col);
                    _inputGradientMap.emplace_back( c, h * _stride + row, w * _stride + col );
                }
            }
        }

        return _output;
    }

    std::vector<std::vector<Eigen::MatrixXd>> forwardBatch(const std::vector<
        std::vector<Eigen::MatrixXd>>& inputBatch) 
    {
        _inputGradientMapBatch.clear();
        _mapIndex = 0;

        Eigen::Index row, col;

        int batchSize = inputBatch.size();
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < _inputChannels; c++) {
                for (int h = 0; h < _outputHeight; h++) {
                    for (int w = 0; w < _outputWidth; w++) {
                        _outputBatch[b][c](h, w) = (inputBatch[b][c].block(h * _stride, 
                                            w * _stride, _kernelSize, _kernelSize)).maxCoeff(&row, &col);
                        _inputGradientMapBatch.emplace_back( b, c, h * _stride + row, w * _stride + col );

                    }
                }
            }
        }

        return _outputBatch;
    }

    std::vector<Eigen::MatrixXd> backward(const std::vector<
        Eigen::MatrixXd>& lossGradient) 
    {
        std::vector<Eigen::MatrixXd> inputGradient(_inputChannels, 
            Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
        
        if (_inputGradientMap.size() != lossGradient.size() * lossGradient[0].rows() * lossGradient[0].cols()) {
            throw std::runtime_error("Gradient map size mismatch");
        }

        for (int c = 0; c < _inputChannels; c++) {
            for (int h = 0; h < _outputHeight; h++) {
                for (int w = 0; w < _outputWidth; w++) {
                    int channel, row, col;
                    _getDataLocation(channel, row, col);

                    inputGradient[channel](row, col) = lossGradient[c](h, w);
                }
            }
        }

        return inputGradient;
    }

    std::vector<std::vector<Eigen::MatrixXd>> backwardBatch(const std::vector<
        std::vector<Eigen::MatrixXd>>& lossGradientBatch) 
    {
        std::vector<std::vector<Eigen::MatrixXd>> inputGradientBatch(_batchSize, 
            std::vector<Eigen::MatrixXd>(_inputChannels,
            Eigen::MatrixXd::Zero(_inputHeight, _inputWidth)));

        if(_inputGradientMapBatch.size() != lossGradientBatch.size() * 
            lossGradientBatch[0].size() * lossGradientBatch[0][0].rows() * 
            lossGradientBatch[0][0].cols()){
            throw std::runtime_error("Gradient map size mismatch");
        }

        int batchSize = lossGradientBatch.size();
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < _inputChannels; c++) {
                for (int h = 0; h < _outputHeight; h++) {
                    for (int w = 0; w < _outputWidth; w++) {
                        int batch, channel, row, col;
                        _getDataLocation(batch, channel, row, col);

                        inputGradientBatch[batch][channel](row, col) = 
                            lossGradientBatch[b][c](h, w);
                    }
                }
            }
        }

        return inputGradientBatch;
    }

private:
    void _getDataLocation(int& dataChannel, int& dataRow, int& dataColumn) {
        if (_mapIndex >= _inputGradientMap.size()) {
            throw std::out_of_range("MaxPooling: Gradient map index out of range.");
        }
        const auto& currentLocation = _inputGradientMap[_mapIndex++];
        dataChannel = std::get<0>(currentLocation);
        dataRow = std::get<1>(currentLocation);
        dataColumn = std::get<2>(currentLocation);
    }


    void _getDataLocation(int& dataBatch, int& dataChannel, int& dataRow, int& dataColumn) {
        if (_mapIndex >= _inputGradientMapBatch.size()) {
            throw std::out_of_range("MaxPooling: Gradient map index out of range.");
        }
        const auto& currentLocation = _inputGradientMapBatch[_mapIndex++];
        dataBatch = std::get<0>(currentLocation);
        dataChannel = std::get<1>(currentLocation);
        dataRow = std::get<2>(currentLocation);
        dataColumn = std::get<3>(currentLocation);
    }
};
