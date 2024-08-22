#ifndef MAXPOOLING_HPP
#define MAXPOOLING_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>


class MaxPooling {
private:
    const size_t _inputHeight;
    const size_t _inputWidth;
    const size_t _inputChannels;
    
    const size_t _kernelSize;
    const size_t _stride;

    size_t _outputHeight;
    size_t _outputWidth;

    std::vector<std::tuple<size_t, size_t, size_t>> _inputGradientMap;

public:
    MaxPooling(size_t inputHeight, size_t inputWidth, size_t inputChannels,
               size_t poolSize, size_t stride = 2)
        : _inputHeight(inputHeight),
          _inputWidth(inputWidth),
          _inputChannels(inputChannels),
          _kernelSize(poolSize),
          _stride(stride)
    {
        _outputHeight = (_inputHeight - _kernelSize) / _stride + 1;
        _outputWidth = (_inputWidth - _kernelSize) / _stride + 1;
        _inputGradientMap.reserve(_inputChannels * _outputHeight 
                                                 * _outputWidth);
    }

    std::vector<Eigen::MatrixXd> forward(std::vector<Eigen::MatrixXd>& input)
    {
        if (input.size() != _inputChannels) {
            throw std::invalid_argument("Number of input channels must match the initialized value.");
        }
        
        std::vector<Eigen::MatrixXd> output(_inputChannels);
        for (size_t c = 0; c < _inputChannels; ++c) {

            output[c] = _maxPoolChannel(input[c], c);
        } 
           
        return output;
    }

   
    std::vector<Eigen::MatrixXd> backward(const std::vector<
        Eigen::MatrixXd>& lossGradient)
    {
        if (_inputGradientMap.size() != lossGradient.size() * lossGradient[0].rows() 
                                        * lossGradient[0].cols()) {
            throw std::logic_error("Gradient map size does not match the expected size.");
        }

        std::vector<Eigen::MatrixXd> inputGradient(_inputChannels,
                    Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
        for (size_t c = 0; c < _inputChannels; ++c) {
            for (size_t h = 0; h < _outputHeight; ++h) {
                for (size_t w = 0; w < _outputWidth; ++w) {
                    size_t channel, row, col;
                    _getDataLocation(channel, row, col);

                    inputGradient[channel](row, col) = lossGradient[c](h, w);
                }
            }
        }

        return inputGradient;
    }

private:
    /*Eigen::MatrixXd _maxPoolChannel(Eigen::MatrixXd& inputChannel, size_t channel)
    {
        Eigen::Index row = 0, col = 0;
        Eigen::MatrixXd outputChannel = Eigen::MatrixXd::Zero(
                                        _outputHeight, _outputWidth);
    
        for (size_t h = 0; h < _outputHeight; ++h) {
            for (size_t w = 0; w < _outputWidth; ++w) {
                outputChannel(h, w) = (inputChannel.block(h * _stride, w * 
                    _stride, _kernelSize, _kernelSize)).maxCoeff(&row, &col);
                _inputGradientMap.push_back({ channel, row, col });
            }
        }

        return outputChannel;
    }*/

    /*void _getDataLocation(size_t& dataChannel, size_t& dataRow, size_t& dataColumn)
    {
        std::tuple<size_t, size_t, size_t> currentLocation = _inputGradientMap.front();
        dataChannel = std::get<0>(currentLocation);
        dataRow = std::get<1>(currentLocation);
        dataColumn = std::get<2>(currentLocation);
        _inputGradientMap.erase(_inputGradientMap.begin());
    }*/
    
    Eigen::MatrixXd _maxPoolChannel(const Eigen::MatrixXd& inputChannel, size_t channel)
    {
        Eigen::Index row = 0, col = 0;
        Eigen::MatrixXd outputChannel = Eigen::MatrixXd::Zero(_outputHeight, _outputWidth);

        for (size_t h = 0; h < _outputHeight; ++h) {
            for (size_t w = 0; w < _outputWidth; ++w) {
                outputChannel(h, w) = inputChannel.block(h * _stride, w * _stride, _kernelSize, _kernelSize).maxCoeff(&row, &col);
                _inputGradientMap.emplace_back(channel, h * _stride + row, w * _stride + col);
            }
        }

        return outputChannel;
    }
    
    void _getDataLocation(size_t& dataChannel, size_t& dataRow, size_t& dataColumn)
    {
        if (_inputGradientMap.empty()) {
            throw std::logic_error("Gradient map is empty.");
        }
        std::tie(dataChannel, dataRow, dataColumn) = _inputGradientMap.front();
        _inputGradientMap.erase(_inputGradientMap.begin());
    }
};

#endif // MAXPOOLING_HPP