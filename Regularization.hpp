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
    const size_t _numChannels;
    const size_t _channelHeight;
    const size_t _channelWidth;
    const size_t _featureSize;
    double _epsilon;
    double _momentum;
    bool _initialized;
    bool _isTraining;

    Eigen::VectorXd _runningMean;
    Eigen::VectorXd _runningVariance;

    Eigen::VectorXd _gamma;
    Eigen::VectorXd _beta;
    Eigen::VectorXd _dGamma;
    Eigen::VectorXd _dBeta;

    Eigen::MatrixXd _batchedInput;               // X - input in batches
    Eigen::MatrixXd _normalizedBatch;            // X_hat - normalized batches
    std::vector<Eigen::MatrixXd> _input;

    std::unique_ptr<AdamOptimizer> _optimizer;

public:
    BatchNormalization(size_t numChannels, size_t channelHeight, 
                       size_t channelWidth, double momentum = 0.1)
        : _numChannels(numChannels),
          _channelHeight(channelHeight),
          _channelWidth(channelWidth), 
          _featureSize(channelHeight * channelWidth), 
          _epsilon(1e-8), 
          _momentum(momentum),
          _initialized(false), _isTraining(true),
          _optimizer(std::make_unique<AdamOptimizer>(BatchNormalizationMode)) 
    {}

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>&
        input) 
    {
        if (!_initialized) {
            _InitializeParameters();
        }
        _batchedInput = _createBatchesFromChannels(input);
        
        // Compute mean and variance
        Eigen::VectorXd Mean = _batchedInput.colwise().mean();
        Eigen::VectorXd Variance = ((_batchedInput.rowwise() - Mean.transpose())
                        .array().square().colwise().sum() / _numChannels).matrix();

        // Update running mean and variance - exponential moving average
        if(_isTraining){
            _runningMean = (1 - _momentum) * _runningMean + _momentum * Mean;
            _runningVariance = (1 - _momentum) * _runningVariance + _momentum * Variance;
        }
        Eigen::VectorXd meanToUse = _isTraining ? Mean : _runningMean;
        Eigen::VectorXd varianceToUse = _isTraining ? Variance :
                _runningVariance;
        // Normalize
        _normalizedBatch = (_batchedInput.rowwise() - meanToUse.transpose()).array()
                        .rowwise() / (varianceToUse.array().transpose() + _epsilon)
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
                                    - _runningMean.transpose()).array()).colwise().sum()
                                    .transpose() * -0.5 * (_runningVariance.array() + 
                                    _epsilon).pow(-1.5));
        // Gradients w.r.t. mean
        Eigen::VectorXd dLoss_dMean = (dLoss_dXHat.array().rowwise() * -(_runningVariance.array() 
                                    + _epsilon).sqrt().cwiseInverse().transpose()).matrix()
                                    .colwise().sum().transpose() + (dLoss_dVar.array() * (-2.0 
                                    / _numChannels)).matrix().asDiagonal() * (_batchedInput.rowwise() 
                                    - _runningMean.transpose()).colwise().sum().transpose();
        // Gradients w.r.t. input batch
        Eigen::MatrixXd dLoss_dBatches = (dLoss_dXHat.array().rowwise() * (_runningVariance.array() 
                                        + _epsilon).sqrt().cwiseInverse().transpose()).matrix() 
                                        + ((_batchedInput.rowwise() - _runningMean.transpose()) * 
                                        (dLoss_dVar.transpose() * 2.0 / _numChannels).asDiagonal()) 
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

    void SetTestMode() { //add assertion
        _isTraining = false;
        _dGamma.resize(0);
        _dBeta.resize(0);
    }

    void SetTrainingMode() { //add assertion
        _isTraining = true;
        _dGamma.setConstant(_featureSize, 0.0);
        _dBeta.setConstant(_featureSize, 0.0);
    }

private:
    void _InitializeParameters() 
    {
        //Initialize parameters
        _gamma.setConstant(_featureSize, 1.0);
        _beta.setConstant(_featureSize, 0.0);
        _dGamma.setConstant(_featureSize, 0.0);
        _dBeta.setConstant(_featureSize, 0.0);
        //Initialize variables
        _runningMean.setConstant(_featureSize, 0.0);
        _runningVariance.setConstant(_featureSize, 0.0);
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
        /*for (size_t c = 0; c < _numChannels; ++c) {
            batches.row(c) = Eigen::Map<const Eigen::RowVectorXd>(input[c].data(), _featureSize);
        }*/
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
                        remapedInput[c](h, w) = batches(c, h * _channelHeight + w);
                }
            }
        }
        /*for (size_t c = 0; c < _numChannels; ++c) {
            remapedInput[c] = Eigen::Map<const Eigen::MatrixXd>(batches.row(c).data(), _channelHeight, _channelWidth);
        }*/
        return remapedInput;
    }

};

class Dropout {
private:
    size_t _inputHeight;
    size_t _inputWidth;
    size_t _numChannels;

    double _dropoutRate;
    double _scaleFactor;
    bool _isTraining;
    bool _initialized;

    std::mt19937 _gen;
    std::uniform_real_distribution<double> _dist;
public:
    Dropout(double dropoutRate = 0.5)
        : _dropoutRate(dropoutRate),
          _scaleFactor(1/(1-dropoutRate)),
          _inputHeight(0),
          _inputWidth(0),
          _numChannels(0), 
          _isTraining(true), 
          _initialized(false),
          _gen(std::random_device{}()), 
          _dist(0.0, 1.0) 
    {
        if (dropoutRate < 0.0 || dropoutRate >= 1.0) {
            throw std::invalid_argument("Dropout rate must be between 0 and 1.");
        }
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) 
    {
        setupDimensions(input.rows(), input.cols());

        if (!_isTraining || _dropoutRate == 0.0) {
            return input;
        }
        
        Eigen::MatrixXd dropoutMask = createRandomMask();
        dropoutMask = (dropoutMask.array() > _dropoutRate).cast<double>();
        Eigen::MatrixXd dropoutOutput = (input.array() * dropoutMask.array()) 
                                        / (1.0 - _dropoutRate);

        return dropoutOutput * _scaleFactor;
    }

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) 
    {
        if (input.empty()) {
            throw std::invalid_argument("Input batch must not be empty.");
        }

        setupDimensions(input[0].rows(), input[0].cols(), input.size());

        if (!_isTraining || _dropoutRate == 0.0) {
            return input;
        }

        std::vector<Eigen::MatrixXd> dropoutOutputs(_numChannels);
        for (size_t c = 0; c < _numChannels; ++c) {
            Eigen::MatrixXd dropoutMask = createRandomMask();
            dropoutMask = (dropoutMask.array() > _dropoutRate).cast<double>();

            dropoutOutputs[c] = (input[c].array() * dropoutMask.array()) / (1.0 - _dropoutRate);
            dropoutOutputs[c] *= _scaleFactor;
        }

        return dropoutOutputs;
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
    Eigen::MatrixXd createRandomMask() 
    {
        return Eigen::MatrixXd::NullaryExpr(_inputHeight, _inputWidth, [this]() {
            return _dist(_gen);
        });
    }

    void setupDimensions(size_t inputHeight, size_t inputWidth, size_t numChannels = 0)
    {
        if (!_initialized) {
            _inputHeight = inputHeight;
            _inputWidth = inputWidth;
            _numChannels = numChannels;
            _initialized = true;
        }
    }


};

#endif // REGULARIZATION_HPP
