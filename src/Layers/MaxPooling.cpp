/**
 * @file MaxPooling.cpp
 * @brief Implementation file for the MaxPooling layer.
 *
 * This file contains the implementation of the MaxPooling class, including the
 * constructor, forward and backward passes, and all helper methods.
 */
#include "../../include/Layers/MaxPooling.hpp"

#include <stdexcept>

/**
 * @brief Constructs a new MaxPooling object.
 * @copydoc MaxPooling::MaxPooling
 */
MaxPooling::MaxPooling(size_t inputHeight, size_t inputWidth, size_t inputChannels, size_t poolSize,
                       size_t batchSize, size_t stride)
    : _inputHeight(inputHeight),
      _inputWidth(inputWidth),
      _inputChannels(inputChannels),
      _kernelSize(poolSize),
      _stride(stride),
      _batchSize(batchSize)
{
    _validate();
    _initialize();
}

std::vector<Eigen::MatrixXd> MaxPooling::forward(const std::vector<Eigen::MatrixXd>& input)
{
    _inputGradientMap.clear();
    _inputGradientMap.reserve(_inputChannels * _outputHeight * _outputWidth);

    Eigen::Index row, col;

    for (size_t c = 0; c < _inputChannels; ++c)
    {
        for (size_t h = 0; h < _outputHeight; ++h)
        {
            for (size_t w = 0; w < _outputWidth; ++w)
            {
                _output[c](h, w) = input[c]
                                       .block(h * _stride, w * _stride, _kernelSize, _kernelSize)
                                       .maxCoeff(&row, &col);
                _inputGradientMap.emplace_back(c, h * _stride + row, w * _stride + col);
            }
        }
    }

    return _output;
}

std::vector<std::vector<Eigen::MatrixXd>> MaxPooling::forwardBatch(
    const std::vector<std::vector<Eigen::MatrixXd>>& inputBatch)
{
    _inputGradientMapBatch.clear();
    _inputGradientMapBatch.reserve(_batchSize * _inputChannels * _outputHeight * _outputWidth);

    Eigen::Index row, col;

    for (size_t b = 0; b < _batchSize; ++b)
    {
        for (size_t c = 0; c < _inputChannels; ++c)
        {
            for (size_t h = 0; h < _outputHeight; ++h)
            {
                for (size_t w = 0; w < _outputWidth; ++w)
                {
                    _outputBatch[b][c](h, w) =
                        inputBatch[b][c]
                            .block(h * _stride, w * _stride, _kernelSize, _kernelSize)
                            .maxCoeff(&row, &col);
                    _inputGradientMapBatch.emplace_back(b, c, h * _stride + row, w * _stride + col);
                }
            }
        }
    }

    return _outputBatch;
}

std::vector<Eigen::MatrixXd> MaxPooling::backward(const std::vector<Eigen::MatrixXd>& gradient)
{
    std::vector<Eigen::MatrixXd> inputGradient(_inputChannels,
                                               Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

    if (_inputGradientMap.size() != gradient.size() * gradient[0].rows() * gradient[0].cols())
    {
        throw std::runtime_error("[MaxPooling]: Gradient map size mismatch");
    }

    size_t mapIndex = 0;
    for (size_t c = 0; c < _inputChannels; ++c)
    {
        for (size_t h = 0; h < _outputHeight; ++h)
        {
            for (size_t w = 0; w < _outputWidth; ++w)
            {
                size_t channel, row, col;
                _getDataLocation(channel, row, col, mapIndex);
                inputGradient[channel](row, col) = gradient[c](h, w);
            }
        }
    }

    return inputGradient;
}

std::vector<std::vector<Eigen::MatrixXd>> MaxPooling::backwardBatch(
    const std::vector<std::vector<Eigen::MatrixXd>>& gradientBatch)
{
    std::vector<std::vector<Eigen::MatrixXd>> inputGradientBatch(
        _batchSize, std::vector<Eigen::MatrixXd>(_inputChannels,
                                                 Eigen::MatrixXd::Zero(_inputHeight, _inputWidth)));

    if (_inputGradientMapBatch.size() != gradientBatch.size() * gradientBatch[0].size() *
                                             gradientBatch[0][0].rows() *
                                             gradientBatch[0][0].cols())
    {
        throw std::runtime_error("[MaxPooling]: Gradient map size mismatch");
    }

    size_t mapIndex = 0;
    for (size_t b = 0; b < _batchSize; ++b)
    {
        for (size_t c = 0; c < _inputChannels; ++c)
        {
            for (size_t h = 0; h < _outputHeight; ++h)
            {
                for (size_t w = 0; w < _outputWidth; ++w)
                {
                    size_t batch, channel, row, col;
                    _getDataLocation(batch, channel, row, col, mapIndex);
                    inputGradientBatch[batch][channel](row, col) = gradientBatch[b][c](h, w);
                }
            }
        }
    }

    return inputGradientBatch;
}

void MaxPooling::_initialize()
{
    _outputHeight = (_inputHeight - _kernelSize) / _stride + 1;
    _outputWidth = (_inputWidth - _kernelSize) / _stride + 1;

    if (_outputHeight == 0 || _outputWidth == 0)
    {
        throw std::invalid_argument("[MaxPooling]: Output dimensions must be positive.");
    }

    if (_batchSize == 1)
    {
        _output.assign(_inputChannels, Eigen::MatrixXd::Zero(_outputHeight, _outputWidth));
    }
    else
    {
        _outputBatch.assign(
            _batchSize, std::vector<Eigen::MatrixXd>(
                            _inputChannels, Eigen::MatrixXd::Zero(_outputHeight, _outputWidth)));
    }
}

void MaxPooling::_validate() const
{
    if (_kernelSize == 0)
    {
        throw std::invalid_argument("[MaxPooling]: Kernel size must be positive.");
    }
    if (_stride == 0)
    {
        throw std::invalid_argument("[MaxPooling]: Stride must be positive.");
    }
    if (_batchSize == 0)
    {
        throw std::invalid_argument("[MaxPooling]: Batch size must be positive.");
    }
    if (_inputHeight == 0 || _inputWidth == 0)
    {
        throw std::invalid_argument("[MaxPooling]: Input dimensions must be positive.");
    }
    if (_inputChannels == 0)
    {
        throw std::invalid_argument("[MaxPooling]: Input channels must be positive.");
    }
    if (_inputHeight < _kernelSize || _inputWidth < _kernelSize)
    {
        throw std::invalid_argument(
            "[MaxPooling]: Input dimensions must be larger than kernel size.");
    }
}

void MaxPooling::_getDataLocation(size_t& dataChannel, size_t& dataRow, size_t& dataColumn,
                                  size_t& mapIndex)
{
    if (mapIndex >= _inputGradientMap.size())
    {
        throw std::out_of_range("[MaxPooling]: Gradient map index out of range.");
    }

    const auto& currentLocation = _inputGradientMap[mapIndex++];
    dataChannel = std::get<0>(currentLocation);
    dataRow = std::get<1>(currentLocation);
    dataColumn = std::get<2>(currentLocation);
}

void MaxPooling::_getDataLocation(size_t& dataBatch, size_t& dataChannel, size_t& dataRow,
                                  size_t& dataColumn, size_t& mapIndex)
{
    if (mapIndex >= _inputGradientMapBatch.size())
    {
        throw std::out_of_range("[MaxPooling]: Gradient map index out of range.");
    }

    const auto& currentLocation = _inputGradientMapBatch[mapIndex++];
    dataBatch = std::get<0>(currentLocation);
    dataChannel = std::get<1>(currentLocation);
    dataRow = std::get<2>(currentLocation);
    dataColumn = std::get<3>(currentLocation);
}
