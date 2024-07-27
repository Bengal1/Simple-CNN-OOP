#ifndef MAXPOOLING_HPP
#define MAXPOOLING_HPP

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

    int _outputHeight;
    int _outputWidth;

    std::vector<Eigen::MatrixXd> _output;
    std::vector<std::tuple<int, int, int>> _inputGradientMap;

public:
    MaxPooling(int inputHeight, int inputWidth, int inputChannels,
        int poolSize, int stride = 2)
        : _inputHeight(inputHeight), _inputWidth(inputWidth),
        _inputChannels(inputChannels), _kernelSize(poolSize),
        _stride(stride)
    {
        // Calculate output dimensions
        _outputHeight = (_inputHeight - _kernelSize) / _stride + 1;
        _outputWidth = (_inputWidth - _kernelSize) / _stride + 1;
        _output.assign(_inputChannels, Eigen::MatrixXd::Zero(
            _outputHeight, _outputWidth));
        _inputGradientMap.reserve(_inputChannels * _outputHeight *
            _outputWidth);
        
    }

    std::vector<Eigen::MatrixXd> forward(std::vector<Eigen::MatrixXd>& input)
    {
        Eigen::Index row, col;
        
        assert(input.size() == _inputChannels);

        for (int c = 0; c < _inputChannels; ++c) {
            
            _output[c] = _maxPoolChannel(input[c], c);
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
    Eigen::MatrixXd _maxPoolChannel(Eigen::MatrixXd& inputChannel, int channel)
    {
        Eigen::Index row = 0, col = 0;
        Eigen::MatrixXd outputChannel = Eigen::MatrixXd::
            Zero(_outputHeight, _outputWidth);
    
        for (int h = 0; h < _outputHeight; ++h) {
            for (int w = 0; w < _outputWidth; ++w) {
                outputChannel(h, w) = (inputChannel.block(h * _stride, w * 
                    _stride, _kernelSize, _kernelSize)).maxCoeff(&row, &col);
                _inputGradientMap.push_back({ channel, row, col });
            }
        }

        return outputChannel;
    }

    void _getDataLocation(int& dataChannel, int& dataRow, int& dataColumn)
    {
        std::tuple<int, int, int> currentLocation = _inputGradientMap.front();
        dataChannel = std::get<0>(currentLocation);
        dataRow = std::get<1>(currentLocation);
        dataColumn = std::get<2>(currentLocation);
        _inputGradientMap.erase(_inputGradientMap.begin());
    }

};

#endif // MAXPOOLING_HPP