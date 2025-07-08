// Dropout.tpp
#pragma once

#include <type_traits>

template <typename T>
T Dropout::forward(const T& input)
{
    if (_numChannels == 0)
    {
        _getInputDimensions(input);
    }

    if (!_isTraining || _dropoutRate == 0.0)
    {
        return input;
    }

    if constexpr (std::is_same_v<T, Eigen::MatrixXd>)
    {
        if (_inputHeight != input.rows() || _inputWidth != input.cols())
        {
            throw std::invalid_argument("[Dropout]: Input dimensions do not match.");
        }

        Eigen::MatrixXd randomMask = _createRandomMask();
        _dropoutMask = (randomMask.array() > _dropoutRate).cast<double>();
        return input.array() * _dropoutMask.array() * _dropoutScale;
    }
    else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>)
    {
        if (_numChannels != input.size() || _inputHeight != input[0].rows() ||
            _inputWidth != input[0].cols())
        {
            throw std::invalid_argument("[Dropout]: Input dimensions do not match.");
        }

        std::vector<Eigen::MatrixXd> droppedOutInput(_numChannels);
        _dropoutMask3D.clear();
        _dropoutMask3D.reserve(_numChannels);

        for (size_t c = 0; c < _numChannels; ++c)
        {
            Eigen::MatrixXd mask = (_createRandomMask().array() > _dropoutRate).cast<double>();
            _dropoutMask3D.push_back(mask);
            droppedOutInput[c] = input[c].array() * mask.array() * _dropoutScale;
        }
        return droppedOutInput;
    }
    else
    {
        throw std::invalid_argument("[Dropout]: Unsupported input type.");
    }
}

template <typename T>
T Dropout::backward(const T& dOutput)
{
    if (!_isTraining || _dropoutRate == 0.0)
    {
        return dOutput;
    }

    if constexpr (std::is_same_v<T, Eigen::MatrixXd>)
    {
        if (_inputHeight != dOutput.rows() || _inputWidth != dOutput.cols())
        {
            throw std::invalid_argument("[Dropout]: Gradient dimensions do not match.");
        }
        return dOutput.array() * _dropoutMask.array() * _dropoutScale;
    }
    else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>)
    {
        if (_numChannels != dOutput.size() || _inputHeight != dOutput[0].rows() ||
            _inputWidth != dOutput[0].cols())
        {
            throw std::invalid_argument("[Dropout]: Gradient dimensions do not match.");
        }

        std::vector<Eigen::MatrixXd> dInput(_numChannels);
        for (size_t c = 0; c < _numChannels; ++c)
        {
            dInput[c] = dOutput[c].array() * _dropoutMask3D[c].array() * _dropoutScale;
        }
        return dInput;
    }
    else
    {
        throw std::invalid_argument("[Dropout]: Unsupported input type.");
    }
}

template <typename T>
void Dropout::_getInputDimensions(const T& input)
{
    if constexpr (std::is_same_v<T, Eigen::VectorXd>)
    {
        _numChannels = 1;
        _inputHeight = input.size();
        _inputWidth = 1;
        _dropoutMask = Eigen::MatrixXd::Zero(_inputHeight, _inputWidth);
    }
    else if constexpr (std::is_same_v<T, Eigen::MatrixXd>)
    {
        _numChannels = 1;
        _inputHeight = input.rows();
        _inputWidth = input.cols();
        _dropoutMask = Eigen::MatrixXd::Zero(_inputHeight, _inputWidth);
    }
    else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>)
    {
        _numChannels = input.size();
        _inputHeight = input[0].rows();
        _inputWidth = input[0].cols();
        _dropoutMask = Eigen::MatrixXd::Zero(_inputHeight, _inputWidth);
    }
    else
    {
        throw std::invalid_argument("[Dropout]: Unsupported input type.");
    }
}
