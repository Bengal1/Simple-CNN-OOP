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

    std::vector<Eigen::MatrixXd> _output;
    
    std::vector<std::tuple<int, int, int>> _inputGradientMap;

public:
    MaxPooling(int inputHeight, int inputWidth, int inputChannels,
        int poolSize, int batchSize = 1, int stride = 2)
        : _inputHeight(inputHeight), _inputWidth(inputWidth),
        _inputChannels(inputChannels), _kernelSize(poolSize),
        _stride(stride), _batchSize(batchSize)
    {
        // Calculate output dimensions
        _outputHeight = (_inputHeight - _kernelSize) / _stride + 1;
        _outputWidth = (_inputWidth - _kernelSize) / _stride + 1;
        if (_batchSize == 1) {
            _output.assign(_inputChannels, Eigen::MatrixXd::Zero(
                _outputHeight, _outputWidth));
            _inputGradientMap.reserve(_inputChannels * _outputHeight *
                _outputWidth);
        }
        else {
            exit(-1);
        }
    }

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input)
    {
        Eigen::Index row, col;

        for (int c = 0; c < _inputChannels; ++c) {
            for (int h = 0; h < _outputHeight; ++h) {
                for (int w = 0; w < _outputWidth; ++w) {
                    _output[c](h, w) = (input[c].block(h * _stride, w * _stride,
                        _kernelSize, _kernelSize)).maxCoeff(&row, &col);
                    _inputGradientMap.push_back({ c, row, col });
                }
            }
        }
        return _output;
    }

   
    std::vector<Eigen::MatrixXd> backward(const std::vector<
        Eigen::MatrixXd>& lossGradient)
    {
        std::vector<Eigen::MatrixXd> inputGradient(_inputChannels,
            Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

        assert(_inputGradientMap.size() == lossGradient.size() *
            lossGradient[0].rows() * lossGradient[0].cols());

        for (int c = 0; c < _inputChannels; ++c) {
            for (int h = 0; h < _outputHeight; ++h) {
                for (int w = 0; w < _outputWidth; ++w) {
                    int channel, row, col;
                    _getDataLocation(channel, row, col);

                    inputGradient[channel](row, col) = lossGradient[c](h, w);
                }
            }
        }

        return inputGradient;
    }

private:
    void _getDataLocation(int& dataChannel, int& dataRow, int& dataColumn)
    {
        std::tuple<int, int, int> currentLocation = _inputGradientMap.front();
        dataChannel = std::get<0>(currentLocation);
        dataRow = std::get<1>(currentLocation);
        dataColumn = std::get<2>(currentLocation);
        _inputGradientMap.erase(_inputGradientMap.begin());
    }

};
