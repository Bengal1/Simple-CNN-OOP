/**
 * @file Convolution2D.hpp
 * @brief Header file for the Convolution2D layer.
 *
 * This header defines the Convolution2D class, which implements a 2D
 * convolutional layer for a neural network. It handles the forward and
 * backward passes, parameter updates, and manages filters and biases.
 */
#pragma once

#include <Eigen/Dense>
#include <memory>

#include "../Optimizer/Adam.hpp"

/**
 * @class Convolution2D
 * @brief Implements a 2D convolutional layer.
 *
 * This class provides the core functionality of a convolutional layer,
 * including convolution operations, forward and backward passes, and
 * parameter updates via an optimizer.
 */
class Convolution2D
{
   private:
    // Default parameters
    static constexpr int DefaultStride = 1;
    static constexpr int DefaultPadding = 0;
    // Input dimensions
    const size_t _inputHeight;
    const size_t _inputWidth;
    const size_t _inputChannels;
    const size_t _batchSize;

    // Filter parameters
    const size_t _numFilters;
    const size_t _kernelSize;
    const size_t _stride;
    const size_t _padding;

    // Output dimensions
    size_t _outputHeight;
    size_t _outputWidth;

    // Biases
    Eigen::VectorXd _biases;
    Eigen::VectorXd _biasesGradient;

    // Input data
    Eigen::MatrixXd _input;
    std::vector<Eigen::MatrixXd> _input3D;

    // Filters and gradients
    std::vector<std::vector<Eigen::MatrixXd>> _filters;
    std::vector<std::vector<Eigen::MatrixXd>> _filtersGradient;

    // Optimizer
    std::unique_ptr<Optimizer> _optimizer;

   public:
    Convolution2D(size_t inputHeight, size_t inputWidth, size_t inputChannels, size_t numFilters,
                  size_t kernelSize, size_t batchSize = 1, size_t stride = DefaultStride,
                  size_t padding = DefaultPadding);

    template <typename T>
    std::vector<Eigen::MatrixXd> forward(const T& input);

    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& dLoss_dOutput);
    void updateParameters();

    std::vector<std::vector<Eigen::MatrixXd>> getFilters();
    Eigen::VectorXd getBiases();

   private:
    void _initialize();
    void _initializeFilters();
    void _validateInputParameters() const;

    const Eigen::MatrixXd _padWithZeros(const Eigen::MatrixXd& input,
                                        size_t promptPadding = 0) const;
    const std::vector<Eigen::MatrixXd> _padWithZeros(const std::vector<Eigen::MatrixXd>& input,
                                                     size_t promptPadding = 0) const;

    Eigen::MatrixXd _Convolve2D(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel,
                                size_t padding = 0) const;
};

#include "Convolution2D.tpp"
