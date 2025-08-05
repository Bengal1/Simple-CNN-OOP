/**
 * @file Convolution2D.tpp
 * @brief Template implementation file for the Convolution2D layer's methods.
 *
 * This file contains the implementation details for the template-based
 * methods of the Convolution2D class, specifically the forward pass.
 */
#pragma once

#include <type_traits>

#include "Convolution2D.hpp"

template <typename T>
std::vector<Eigen::MatrixXd> Convolution2D::forward(const T& input)
{
    std::vector<Eigen::MatrixXd> output(_numFilters,
                                        Eigen::MatrixXd::Zero(_outputHeight, _outputWidth));

    if constexpr (std::is_same_v<T, Eigen::MatrixXd>)
    {
        if (input.rows() != _inputHeight || input.cols() != _inputWidth)
            throw std::invalid_argument("[Convolution2D]: Input dimensions do not match.");
        _input = input;

        for (size_t f = 0; f < _numFilters; ++f)
        {
            output[f] += _Convolve2D(input, _filters[f][0], _padding);
            output[f].array() += _biases[f];
        }
    }
    else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>)
    {
        if (input.size() != _inputChannels || input[0].rows() != _inputHeight ||
            input[0].cols() != _inputWidth)
            throw std::invalid_argument("[Convolution2D]: Input dimensions do not match.");
        _input3D = input;

        for (size_t f = 0; f < _numFilters; ++f)
        {
            for (size_t c = 0; c < _inputChannels; ++c)
                output[f] += _Convolve2D(input[c], _filters[f][c], _padding);
            output[f].array() += _biases[f];
        }
    }
    else
    {
        throw std::invalid_argument("[Convolution2D]: Unsupported input type.");
    }

    return output;
}
