// BatchNormalization.hpp
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "../Optimizer/Optimizer.hpp"

class BatchNormalization
{
   private:
    // Default parameters
    static constexpr double DefaultMomentum = 0.1;
    // Input dimensions
    size_t _numChannels = 0;
    size_t _channelHeight = 0;
    size_t _channelWidth = 0;

    // Operational flags
    bool _initialized = false;
    bool _isTraining = true;

    // Hyperparameters
    double _epsilon = 1e-6;
    double _momentum;

    // Statistics
    std::vector<double> _channelMean;
    std::vector<double> _channelVariance;
    std::vector<double> _runningMean;
    std::vector<double> _runningVariance;

    // Learnable parameters
    Eigen::VectorXd _gamma;
    Eigen::VectorXd _beta;
    Eigen::VectorXd _dGamma;
    Eigen::VectorXd _dBeta;

    // Cached input
    std::vector<Eigen::MatrixXd> _input;

    // Optimizer
    std::unique_ptr<Optimizer> _optimizer;

   public:
    BatchNormalization(double momentum = DefaultMomentum);
    ~BatchNormalization() = default;

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input);
    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& dOutput);
    void updateParameters();
    void setTrainingMode(bool isTraining);

   private:
    void _InitializeParameters(const std::vector<Eigen::MatrixXd>& input);
};
