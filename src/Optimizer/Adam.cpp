#include "../../include/Optimizer/Adam.hpp"

Adam::Adam(OptimizerMode mode, size_t numParams, double learningRate, double beta1, double beta2, double epsilon)
    : Optimizer(learningRate),
      _mode(mode),
      _numParams(numParams),
      _beta1(beta1),
      _beta2(beta2),
      _epsilon(epsilon)
{
    _validateInputParameters();
    _initialLearningRate = learningRate;
}

void Adam::updateStep(Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients,
                      const int paramIndex)
{
    if (parameters.size() != gradients.size())
    {
        throw std::invalid_argument(
            "[Optimizer]: Parameters and gradients must have the same size.");
    }
    if (!_isInitialized)
    {
        _initializeMoments(parameters.size());
    }
    if (paramIndex == 0 && _mode == Adam::OptimizerMode::BatchNormalization)
    { // only for BN - Gamma
        _timeStep++;
        _updateEffectiveLearningRate();
    }

    // calculate moments - m_t, v_t
    _firstMomentEstimateVector[paramIndex] =
        _beta1 * _firstMomentEstimateVector[paramIndex].array() + (1 - _beta1) * gradients.array();
    _secondMomentEstimateVector[paramIndex] =
        _beta2 * _secondMomentEstimateVector[paramIndex].array() +
        (1 - _beta2) * gradients.array().square();

    Eigen::VectorXd firstMomentEstimateHat =
        _firstMomentEstimateVector[paramIndex] / _biasCorrection1;
    Eigen::VectorXd secondMomentEstimateHat =
        _secondMomentEstimateVector[paramIndex] / _biasCorrection2;

    parameters -= (_effectiveLearningRate * firstMomentEstimateHat.array() /
                   (secondMomentEstimateHat.array().sqrt() + _epsilon))
                      .matrix();
}

void Adam::updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients,
                      const int paramIndex)
{
    if (parameters.rows() != gradients.rows() || parameters.cols() != gradients.cols())
    {
        throw std::invalid_argument(
            "[Optimizer]: Parameters and gradients must have the same size.");
    }
    if (!_isInitialized)
    {
        _initializeMoments(parameters.rows(), parameters.cols());
    }
    if (!paramIndex)
    {
        _timeStep++;
        _updateEffectiveLearningRate();
    }
    // calculate moments - m_t, v_t
    _firstMomentEstimateMatrix[paramIndex] =
        _beta1 * _firstMomentEstimateMatrix[paramIndex].array() + (1 - _beta1) * gradients.array();
    _secondMomentEstimateMatrix[paramIndex] =
        _beta2 * _secondMomentEstimateMatrix[paramIndex].array() +
        (1 - _beta2) * gradients.array().square();

    Eigen::MatrixXd firstMomentEstimateHat =
        _firstMomentEstimateMatrix[paramIndex] / _biasCorrection1;
    Eigen::MatrixXd secondMomentEstimateHat =
        _secondMomentEstimateMatrix[paramIndex] / _biasCorrection2;

    parameters -= (_effectiveLearningRate * firstMomentEstimateHat.array() /
                   (secondMomentEstimateHat.array().sqrt() + _epsilon))
                      .matrix();
}

void Adam::updateStep(std::vector<Eigen::MatrixXd>& parameters,
                      const std::vector<Eigen::MatrixXd>& gradients, const int paramIndex)
{
    if (parameters.size() != gradients.size())
    {
        throw std::invalid_argument(
            "[Optimizer]: Parameters and gradients must have the same size.");
    }
    if (parameters[0].rows() != gradients[0].rows() || parameters[0].cols() != gradients[0].cols())
    {
        throw std::invalid_argument(
            "[Optimizer]: Parameters and gradients must have the same size.");
    }
    if (!_isInitialized)
    {
        _initializeMoments(parameters[paramIndex].rows(), parameters[paramIndex].cols(),
                           parameters.size());
    }
    if (!paramIndex)
    {
        _timeStep++;
        _updateEffectiveLearningRate();
    }
    for (size_t c = 0; c < _numChannels; ++c)
    {
        // calculate moments - m_t, v_t
        _firstMomentEstimateTensor[paramIndex][c] =
            _beta1 * _firstMomentEstimateTensor[paramIndex][c].array() +
            (1 - _beta1) * gradients[c].array();
        _secondMomentEstimateTensor[paramIndex][c] =
            _beta2 * _secondMomentEstimateTensor[paramIndex][c].array() +
            (1 - _beta2) * gradients[c].array().square();

        Eigen::MatrixXd firstMomentEstimateHat =
            _firstMomentEstimateTensor[paramIndex][c] / _biasCorrection1;
        Eigen::MatrixXd secondMomentEstimateHat =
            _secondMomentEstimateTensor[paramIndex][c] / _biasCorrection2;

        parameters[c] -= (_effectiveLearningRate * firstMomentEstimateHat.array() /
                          (secondMomentEstimateHat.array().sqrt() + _epsilon))
                             .matrix();
    }
}

bool Adam::validateOptimizerMode() const
{
    switch (_mode)
    {
        case OptimizerMode::FullyConnected:
        case OptimizerMode::BatchNormalization:
        case OptimizerMode::Convolution2D:
            return true;
        default:
            return false;
    }
}

void Adam::_validateInputParameters() const
{
    if(!Adam::validateOptimizerMode())
    {
        throw std::invalid_argument("[Optimizer]: Invalid Optimizer Mode.");
    }
    if (_numParams == 0)
    {
        throw std::invalid_argument("[Optimizer]: Invalid number of parameters.");
    }
    if (_learningRate <= 0)
    {
        throw std::invalid_argument("[Optimizer]: Learning rate must be positive.");
    }
    if (_beta1 <= 0 || _beta1 >= 1)
    {
        throw std::invalid_argument("[Optimizer]: Beta1 must be in the range (0, 1).");
    }
    if (_beta2 <= 0 || _beta2 >= 1)
    {
        throw std::invalid_argument("[Optimizer]: Beta2 must be in the range (0, 1).");
    }
    if (_epsilon <= 0)
    {
        throw std::invalid_argument("[Optimizer]: Epsilon must be positive.");
    }
}

void Adam::_initializeMoments(size_t rows, size_t cols, size_t channels)
{
    _numChannels = (channels > 0) ? channels : 1;
    if (_mode == OptimizerMode::FullyConnected)
    { // Weights and Bias
        _firstMomentEstimateMatrix.assign(1, Eigen::MatrixXd::Zero(rows, cols));
        _secondMomentEstimateMatrix.assign(1, Eigen::MatrixXd::Zero(rows, cols));
        _firstMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(rows));
        _secondMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(rows));
    }
    else if (_mode == OptimizerMode::BatchNormalization)
    { // Gamma and Beta
        _firstMomentEstimateVector.assign(2, Eigen::VectorXd::Zero(rows));
        _secondMomentEstimateVector.assign(2, Eigen::VectorXd::Zero(rows));
    }
    else if(_mode == OptimizerMode::Convolution2D)
    { // Convolution2D - Filters and Bias
        _firstMomentEstimateTensor.assign(
            _numParams,
            std::vector<Eigen::MatrixXd>(_numChannels, Eigen::MatrixXd::Zero(rows, cols)));
        _secondMomentEstimateTensor.assign(
            _numParams,
            std::vector<Eigen::MatrixXd>(_numChannels, Eigen::MatrixXd::Zero(rows, cols)));
        _firstMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(_numParams));
        _secondMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(_numParams));
    }
    else
    {
        throw std::invalid_argument("[Optimizer]: Invalid Optimizer Mode.");
    }
    _isInitialized = true;
}

void Adam::_updateEffectiveLearningRate()
{
    if (_isScheduled)
    {
        _learningRateScheduler();
    }
    _applyBiasCorrection();
}

void Adam::_applyBiasCorrection()
{
    _biasCorrection1 = 1.0 - std::pow(_beta1, _timeStep);
    _biasCorrection2 = 1.0 - std::pow(_beta2, _timeStep);
    _effectiveLearningRate = _learningRate * std::sqrt(_biasCorrection2) / _biasCorrection1;
}

void Adam::_learningRateScheduler()
{
    if (!_isScheduled) return;
    if (_timeStep < _warmupSteps)
    {
        // Linear warmup
        double warmupRatio = static_cast<double>(_timeStep) / _warmupSteps;
        //_learningRate = _learningRate + warmupRatio * (_initialLearningRate - _learningRate);
        _learningRate = _initialLearningRate * warmupRatio;
    }
    else if (_timeStep % 10000 == 0)
    {
        _learningRate *= 0.5;
    }
}