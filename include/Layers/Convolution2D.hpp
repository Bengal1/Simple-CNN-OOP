#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "../Optimizer/Adam.hpp"

class Convolution2D
{
   private:
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
                  size_t kernelSize, double maxGradNorm = -1.0, double weightDecay = 0.0,
                  size_t batchSize = 1, size_t stride = 1, size_t padding = 0);

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
