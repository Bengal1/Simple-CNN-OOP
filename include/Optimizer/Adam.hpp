#pragma once
#include "Optimizer.hpp"

class Adam : public Optimizer
{
   public:
    // Optimizer Modes
    enum class OptimizerMode
    {
        FullyConnected,
        BatchNormalization,
        Convolution2D
    };

   private:
    // Default parameters
    static constexpr double DefaultLR = 7e-5;
    static constexpr double DefaultBeta1 = 0.9;
    static constexpr double DefaultBeta2 = 0.999;
    static constexpr double DefaultEpsilon = 1.0e-8;
    static constexpr size_t DefaultParamNum = 1;
    // Adam optimizer parameters
    const double _beta1;
    const double _beta2;
    const double _epsilon;
    const size_t _numParams;
    size_t _numChannels = 0;
    // Time step
    size_t _timeStep = 0;
    // Initialization & config
    bool _isInitialized = false;
    bool _isScheduled = false;
    OptimizerMode _mode;
    // Bias correction
    double _biasCorrection1 = 1.0;
    double _biasCorrection2 = 1.0;
    // Learning rate & Scheduler
    double _effectiveLearningRate = 0.0;
    double _initialLearningRate = 0.0;
    const size_t _warmupSteps = 4000;
    // Moments
    std::vector<Eigen::VectorXd> _firstMomentEstimateVector;
    std::vector<Eigen::VectorXd> _secondMomentEstimateVector;
    std::vector<Eigen::MatrixXd> _firstMomentEstimateMatrix;
    std::vector<Eigen::MatrixXd> _secondMomentEstimateMatrix;
    std::vector<std::vector<Eigen::MatrixXd>> _firstMomentEstimateTensor;
    std::vector<std::vector<Eigen::MatrixXd>> _secondMomentEstimateTensor;

   public:
    Adam(OptimizerMode mode, size_t numParams = DefaultParamNum, double learningRate = DefaultLR,
         double beta1 = DefaultBeta1, double beta2 = DefaultBeta2, double epsilon = DefaultEpsilon);


    void updateStep(Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients,
                    const int paramIndex = 0) override;

    void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients,
                    const int paramIndex = 0) override;

    void updateStep(std::vector<Eigen::MatrixXd>& parameters,
                    const std::vector<Eigen::MatrixXd>& gradients,
                    const int paramIndex = 0) override;

   private:
    bool validateOptimizerMode() const;

    void _validateInputParameters() const;

    void _initializeMoments(size_t rows, size_t cols = 0, size_t channels = 0);

    void _updateEffectiveLearningRate();

    void _applyBiasCorrection();

    void _learningRateScheduler();
};