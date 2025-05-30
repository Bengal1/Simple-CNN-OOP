#pragma once

#include "FullyConnected.hpp"
#include <type_traits>


template <typename T>
Eigen::VectorXd FullyConnected::forward(const T& input) {
    if (_inputChannels == 0)
        _getInputDimensions(input);

    if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
        if (input.size() != _inputSize)
            throw std::invalid_argument("[FullyConnected]: Input size does not match.");
        _flatInput = input;
    }
    else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
        if (input.rows() != _inputHeight || input.cols() != _inputWidth)
            throw std::invalid_argument("[FullyConnected]: Input dimensions do not match.");
        _flatInput = _flattenInput(input);
    }
    else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
        if (input.size() != _inputChannels)
            throw std::invalid_argument("[FullyConnected]: Input channels do not match.");
        _flatInput = _flattenInput(input);
    }

    return _weights * _flatInput + _bias;
}

template <typename T>
T FullyConnected::backward(const Eigen::VectorXd& dLoss_dOutput) {
    if (dLoss_dOutput.size() != _outputSize)
        throw std::invalid_argument("[FullyConnected]: Loss gradient size does not match.");

    _weightsGradient = dLoss_dOutput * _flatInput.transpose();
    _biasGradient = dLoss_dOutput;

    Eigen::VectorXd flatGrad = _weights.transpose() * dLoss_dOutput;
    return _restoreInputShape<T>(flatGrad);
}

template <typename T>
void FullyConnected::_getInputDimensions(const T& input) {
    if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
        _inputChannels = 1;
        _inputHeight = input.size();
        _inputWidth = 1;
        _inputType = InputType::Vector;
    }
    else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
        _inputChannels = 1;
        _inputHeight = input.rows();
        _inputWidth = input.cols();
        _inputType = InputType::Matrix;
    }
    else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
        _inputChannels = input.size();
        _inputHeight = input[0].rows();
        _inputWidth = input[0].cols();
        _inputType = InputType::Tensor3D;
    }
    else {
        throw std::invalid_argument("[FullyConnected]: Unsupported input type.");
    }

    if (_inputSize != _inputChannels * _inputHeight * _inputWidth)
        throw std::invalid_argument("[FullyConnected]: Input size does not match the expected size.");
}

template <typename T>
Eigen::VectorXd FullyConnected::_flattenInput(const T& input) {
    if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
        return input;
    }
    else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
        return Eigen::Map<const Eigen::VectorXd>(input.data(), input.size());
    }
    else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
        Eigen::VectorXd flat(_inputSize);
        Eigen::Index offset = 0;
        for (const auto& mat : input) {
            Eigen::Map<const Eigen::VectorXd> map(mat.data(), mat.size());
            flat.segment(offset, map.size()) = map;
            offset += map.size();
        }
        return flat;
    }
    else {
        throw std::invalid_argument("[FullyConnected]: Unsupported input type.");
    }
}

template <typename T>
T FullyConnected::_restoreInputShape(const Eigen::VectorXd& flat) {
    if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
        if (flat.size() != _inputSize)
            throw std::invalid_argument("[FullyConnected]: Size mismatch in unflattenVector.");
        return flat;
    }
    else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
        if (flat.size() != _inputHeight * _inputWidth)
            throw std::invalid_argument("Size mismatch in unflattening Matrix.");
        return Eigen::Map<const Eigen::MatrixXd>(flat.data(), _inputHeight, _inputWidth);
    }
    else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
        if (flat.size() != _inputChannels * _inputHeight * _inputWidth)
            throw std::invalid_argument("Size mismatch in unflattening 3D Tensor.");
        std::vector<Eigen::MatrixXd> result(_inputChannels,
            Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
        size_t idx = 0;
        for (size_t c = 0; c < _inputChannels; ++c)
            for (size_t h = 0; h < _inputHeight; ++h)
                for (size_t w = 0; w < _inputWidth; ++w)
                    result[c](h, w) = flat(idx++);
        return result;
    }
    else {
        throw std::invalid_argument("[FullyConnected]: Unsupported type for unflattening.");
    }
}
