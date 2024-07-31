#ifndef REGULARIZATION_HPP
#define REGULARIZATION_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <Eigen/Dense>
#include "Optimizer.hpp"


class BatchNormalization {
private:
    double _epsilon;
    bool _initialized;
    bool _isTraining;

    const size_t _numChannels;
    const size_t _channelHeight;
    const size_t _channelWidth;
    const size_t _featureSize;

    Eigen::VectorXd _mean;
    Eigen::VectorXd _variance;

    Eigen::VectorXd _gamma;
    Eigen::VectorXd _beta;
    Eigen::VectorXd _dGamma;
    Eigen::VectorXd _dBeta;

    Eigen::MatrixXd _batchedInput;               // X - input in batches
    Eigen::MatrixXd _normalizedBatch;            // X_hat - normalized batches
    std::vector<Eigen::MatrixXd> _input;

    std::unique_ptr<AdamOptimizer> _optimizer;

public:
    BatchNormalization(size_t numChannels, size_t channelHeight, size_t channelWidth)
        : _numChannels(numChannels),
          _channelHeight(channelHeight),
          _channelWidth(channelWidth), 
          _featureSize(channelHeight * channelWidth), 
          _epsilon(1e-8), _initialized(false), _isTraining(true),
          _optimizer(std::make_unique<AdamOptimizer>(BatchNormalizationMode)) 
    {}

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>&
        input) 
    {
        if (!_initialized) {
            _InitializeParameters(input);
        }
        _batchedInput = _createBatchesFromChannels(input);

        // Compute mean and variance
        _mean = _batchedInput.colwise().mean();
        _variance = ((_batchedInput.rowwise() - _mean.transpose()).array().square()
                    .colwise().sum() / _numChannels).matrix();
        // Normalize
        _normalizedBatch = (_batchedInput.rowwise() - _mean.transpose()).array()
                        .rowwise() / (_variance.array().transpose() + _epsilon)
                        .sqrt();
        // Apply scale (gamma) and shift (beta) - Y = gamma*X + beta
        Eigen::MatrixXd scaledShiftedBatch = ((_normalizedBatch.array().rowwise() 
                                          * _gamma.transpose().array()).rowwise() 
                                          + _beta.transpose().array()).matrix();
        // Map and set output variable                  
        std::vector<Eigen::MatrixXd> outputBN = _remapBatchesToChannels(scaledShiftedBatch);
        
        return outputBN;
    }

    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& dLoss_dOutput) 
    {
        Eigen::MatrixXd dLoss_dY(_numChannels, _featureSize);

        dLoss_dY = _createBatchesFromChannels(dLoss_dOutput);
        
        // Gradients w.r.t. gamma and beta
        _dGamma = (_normalizedBatch.array() * dLoss_dY.array()).colwise().sum();
        _dBeta = dLoss_dY.colwise().sum();
        // Gradients w.r.t. normalized batch
        Eigen::MatrixXd dLoss_dXHat = (dLoss_dY.array().rowwise() * _gamma.transpose()
        .array()).matrix();
        // Gradients w.r.t. variance
        Eigen::VectorXd dLoss_dVar = ((dLoss_dXHat.array() * (_batchedInput.rowwise() 
                                    - _mean.transpose()).array()).colwise().sum()
                                    .transpose() * -0.5 * (_variance.array() + 
                                    _epsilon).pow(-1.5));
        // Gradients w.r.t. mean
        Eigen::VectorXd dLoss_dMean = (dLoss_dXHat.array().rowwise() * -(_variance.array() 
                                    + _epsilon).sqrt().cwiseInverse().transpose()).matrix()
                                    .colwise().sum().transpose() + (dLoss_dVar.array() * (-2.0 
                                    / _numChannels)).matrix().asDiagonal() * (_batchedInput.rowwise() 
                                    - _mean.transpose()).colwise().sum().transpose();
        // Gradients w.r.t. input batch
        Eigen::MatrixXd dLoss_dBatches = (dLoss_dXHat.array().rowwise() * (_variance.array() 
                                        + _epsilon).sqrt().cwiseInverse().transpose()).matrix() 
                                        + ((_batchedInput.rowwise() - _mean.transpose()) * (dLoss_dVar
                                        .transpose() * 2.0 / _numChannels).asDiagonal()) 
                                        + dLoss_dMean.replicate(1, _numChannels).transpose();
        // set dLoss_dInput
        std::vector<Eigen::MatrixXd> dLoss_dInput(_numChannels, 
            Eigen::MatrixXd::Zero(_channelHeight,_channelWidth));

        dLoss_dInput = _remapBatchesToChannels(dLoss_dBatches);
        
        return dLoss_dInput;
    }

    void updateParameters() 
    {
        // Update learnable parameters 
        _optimizer->updateStep(_gamma, _dGamma, 0);
        _optimizer->updateStep(_beta, _dBeta, 1);
        // Reset gradients
        _dGamma.setZero();
        _dBeta.setZero();
    }

    void SetTestMode() {
        _isTraining = false;
        _dGamma.resize(0);
        _dBeta.resize(0);
        //_input.clear();
    }

    void SetTrainingMode() {
        _isTraining = true;
        _dGamma.setConstant(_featureSize, 0.0);
        _dBeta.setConstant(_featureSize, 0.0);
    }

private:
    void _InitializeParameters(const std::vector<Eigen::MatrixXd>& input) 
    {
        //Initialize parameters
        _gamma.setConstant(_featureSize, 1.0);
        _beta.setConstant(_featureSize, 0.0);
        _dGamma.setConstant(_featureSize, 0.0);
        _dBeta.setConstant(_featureSize, 0.0);
        //Initialize variables
        _mean.setConstant(_featureSize, 0.0);
        _variance.setConstant(_featureSize, 0.0);
        _batchedInput.resize(_numChannels, _featureSize);
        _normalizedBatch.resize(_numChannels, _featureSize);
        
        _initialized = true;
    }

    Eigen::MatrixXd _createBatchesFromChannels(const std::vector<Eigen::MatrixXd>& input)
    {
        Eigen::MatrixXd batches(_numChannels, _featureSize);
        
        //reshpe every channel to a row and store it _batchedInput
        for (size_t c = 0; c < _numChannels; ++c) {
            for (size_t h = 0; h < _channelHeight; ++h) {
                for (size_t w = 0; w < _channelWidth; ++w) {
                    batches(c, h * _channelHeight + w) = input[c](h, w);
                }
            }
        }
        return batches;
    }

    std::vector<Eigen::MatrixXd> _remapBatchesToChannels(Eigen::MatrixXd& batches)
    {
        std::vector<Eigen::MatrixXd> remapedInput(_numChannels,
            Eigen::MatrixXd::Zero(_channelHeight, _channelWidth));

        // Reshape each row of batch back into a matrix 
        for (size_t c = 0; c < _numChannels; ++c) {
            for (size_t h = 0; h < _channelHeight; ++h) {
                for (size_t w = 0; w < _channelWidth; ++w) {
                        remapedInput[c](h, w) = batches(c, h * _channelWidth + w);
                }
            }
        }
        return remapedInput;
    }

};

class Dropout {
private:
    size_t _inputHeight;
    size_t _inputWidth;
    size_t _numChannels;
    double _dropoutRate;
    bool _isTraining;

public:
    Dropout(double dropoutRate = 0.5)
        : _dropoutRate(dropoutRate),
          _inputHeight(0),
          _inputWidth(0),
          _numChannels(0), _isTraining(true) 
        {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) 
    {
        if (!_inputHeight) {
            _inputHeight = input.rows();
            _inputWidth = input.cols();
        }
        if (!_isTraining || _dropoutRate == 0.0) {
            // No dropout
            return input;
        }

        // Create a random mask with values between -1 and 1
        Eigen::MatrixXd dropoutMask = CreateRandomMask();
        double randomThreshold = 2 * _dropoutRate - 1;

        // Apply dropout
        dropoutMask = (dropoutMask.array() > randomThreshold).cast<double>();
        return input.array() * dropoutMask.array() / (1.0 - _dropoutRate);
    }

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) 
    {
        if (!_numChannels) {
            _numChannels = input.size();
            _inputHeight = input[0].rows();
            _inputWidth = input[0].cols();
        }
        if (!_isTraining || _dropoutRate == 0.0) {
            // No dropout
            return input;
        }
        std::vector<Eigen::MatrixXd> dropedOutInput(_numChannels,
            Eigen::MatrixXd::Zero(_inputHeight, _inputWidth));
        for (size_t c = 0; c < _numChannels; ++c) {
            // Create a random mask with values between -1 and 1
            Eigen::MatrixXd dropoutMask = CreateRandomMask();
            double randomThreshold = 2 * _dropoutRate - 1;

            // Apply dropout
            dropoutMask = (dropoutMask.array() > randomThreshold).cast<double>();
            dropedOutInput[c] = input[c].array() * dropoutMask.array() / (1.0 -
                _dropoutRate);
        }
        return dropedOutInput;
    }

    void SetTestMode()
    {
        _isTraining = false;
    }

    void SetTrainingMode()
    {
        _isTraining = true;
    }

private:
    Eigen::MatrixXd CreateRandomMask() 
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        Eigen::MatrixXd randomMask = Eigen::MatrixXd::NullaryExpr(_inputHeight,
            _inputWidth, [&dist, &gen]() { return dist(gen); });

        return randomMask;
    }
};

#endif // REGULARIZATION_HPP
