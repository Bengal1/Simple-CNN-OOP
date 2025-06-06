#include "../../include/Layers/Convolution2D.hpp"

#include <random>
#include <stdexcept>

// Constructor
Convolution2D::Convolution2D(size_t inputHeight, size_t inputWidth, size_t inputChannels,
                             size_t numFilters, size_t kernelSize, size_t batchSize, size_t stride,
                             size_t padding)
    : _inputHeight(inputHeight),
      _inputWidth(inputWidth),
      _inputChannels(inputChannels),
      _numFilters(numFilters),
      _kernelSize(kernelSize),
      _batchSize(batchSize),
      _stride(stride),
      _padding(padding),
      _optimizer(std::make_unique<Adam>(Adam::OptimizerMode::Convolution2D, static_cast<int>(numFilters)))
{
    _validateInputParameters();
    _initialize();
}

std::vector<Eigen::MatrixXd> Convolution2D::backward(
    const std::vector<Eigen::MatrixXd>& dLoss_dOutput)
{
    if (dLoss_dOutput.size() != _numFilters)
        throw std::invalid_argument(
            "[Convolution2D]: Loss gradient size must match number of filters.");

    std::vector<Eigen::MatrixXd> dLoss_dInput(_inputChannels,
                                              Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));

    for (size_t f = 0; f < _numFilters; ++f)
    {
        _biasesGradient(f) += dLoss_dOutput[f].sum();

        for (size_t c = 0; c < _inputChannels; ++c)
        {
            Eigen::MatrixXd reversed = dLoss_dOutput[f].colwise().reverse().rowwise().reverse();

            if (_inputChannels == 1)
                _filtersGradient[f][c] += _Convolve2D(_input, reversed);
            else
                _filtersGradient[f][c] += _Convolve2D(_input3D[c], reversed);

            Eigen::MatrixXd reversedFilter = _filters[f][c].colwise().reverse().rowwise().reverse();
            dLoss_dInput[c] += _Convolve2D(dLoss_dOutput[f], reversedFilter, _kernelSize - 1);
        }
    }

    return dLoss_dInput;
}

void Convolution2D::updateParameters()
{
    for (int f = 0; f < _numFilters; ++f)
    {
        _optimizer->updateStep(_filters[f], _filtersGradient[f], f);
        for (size_t c = 0; c < _inputChannels; ++c) _filtersGradient[f][c].setZero();
    }

    _optimizer->updateStep(_biases, _biasesGradient);
    _biasesGradient.setZero();
}

std::vector<std::vector<Eigen::MatrixXd>> Convolution2D::getFilters()
{
    return _filters;
}

Eigen::VectorXd Convolution2D::getBiases()
{
    return _biases;
}

void Convolution2D::_initialize()
{
    _outputHeight = (_inputHeight - _kernelSize + 2 * _padding) / _stride + 1;
    _outputWidth = (_inputWidth - _kernelSize + 2 * _padding) / _stride + 1;

    if (_outputHeight == 0 || _outputWidth == 0)
        throw std::invalid_argument("[Convolution2D]: Output dimensions must be positive.");

    if (_batchSize == 1 && _inputChannels == 1)
    {
        _input.resize(_inputHeight, _inputWidth);
    }
    else
    {
        _input3D.assign(std::max(_batchSize, _inputChannels),
                        Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
    }

    _filtersGradient.assign(
        _numFilters, std::vector<Eigen::MatrixXd>(_inputChannels,
                                                  Eigen::MatrixXd::Zero(_kernelSize, _kernelSize)));
    _biasesGradient = Eigen::VectorXd::Zero(_numFilters);

    _initializeFilters();
}

void Convolution2D::_initializeFilters()
{
    _filters.assign(_numFilters,
                    std::vector<Eigen::MatrixXd>(_inputChannels,
                                                 Eigen::MatrixXd::Zero(_kernelSize, _kernelSize)));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(
        0.0, std::sqrt(2.0 / (_kernelSize * _kernelSize * _inputChannels)));

    for (size_t f = 0; f < _numFilters; ++f)
        for (size_t c = 0; c < _inputChannels; ++c)
            _filters[f][c] =
                Eigen::MatrixXd::NullaryExpr(_kernelSize, _kernelSize, [&]() { return dist(gen); });

    _biases = Eigen::VectorXd::Zero(_numFilters);
}

void Convolution2D::_validateInputParameters() const
{
    if (_inputHeight == 0 || _inputWidth == 0 || _inputChannels == 0 || _numFilters == 0 ||
        _kernelSize == 0 || _stride == 0 || _batchSize == 0)
        throw std::invalid_argument("[Convolution2D]: All dimensions must be positive.");

    if (_inputHeight < _kernelSize || _inputWidth < _kernelSize)
        throw std::invalid_argument("[Convolution2D]: Input must be larger than kernel.");
}

const Eigen::MatrixXd Convolution2D::_padWithZeros(const Eigen::MatrixXd& input,
                                                   size_t promptPadding) const
{
    size_t pad = (promptPadding != 0) ? promptPadding : _padding;
    Eigen::MatrixXd padded = Eigen::MatrixXd::Zero(input.rows() + 2 * pad, input.cols() + 2 * pad);
    padded.block(pad, pad, input.rows(), input.cols()) = input;
    return padded;
}

const std::vector<Eigen::MatrixXd> Convolution2D::_padWithZeros(
    const std::vector<Eigen::MatrixXd>& input, size_t promptPadding) const
{
    std::vector<Eigen::MatrixXd> padded;
    padded.reserve(input.size());
    for (const auto& mat : input) padded.push_back(_padWithZeros(mat, promptPadding));
    return padded;
}

Eigen::MatrixXd Convolution2D::_Convolve2D(const Eigen::MatrixXd& input,
                                           const Eigen::MatrixXd& kernel, size_t padding) const
{
    Eigen::MatrixXd paddedInput = (padding > 0) ? _padWithZeros(input, padding) : input;

    size_t outH = (paddedInput.rows() - kernel.rows()) / _stride + 1;
    size_t outW = (paddedInput.cols() - kernel.cols()) / _stride + 1;
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(outH, outW);

    for (size_t i = 0; i < outH; ++i)
        for (size_t j = 0; j < outW; ++j)
            result(i, j) =
                (paddedInput.block(i * _stride, j * _stride, kernel.rows(), kernel.cols())
                     .cwiseProduct(kernel))
                    .sum();

    return result;
}
