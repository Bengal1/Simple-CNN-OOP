#pragma once

#include <vector>
#include <Eigen/Dense>


class Optimizer {
protected:
	double _learningRate;
	double _maxGradNorm;
	double _weightDecay;

public:
	Optimizer(double learningRate = 0.001, 
		double maxGradNorm = -1.0, double weightDecay = 0.0) 
		: _learningRate(learningRate), 
		_maxGradNorm(maxGradNorm), 
		_weightDecay(weightDecay)
	{}
	virtual ~Optimizer() = default;

	virtual void updateStep(Eigen::VectorXd& parameters, 
							const Eigen::VectorXd& gradients,
							const int paramIndex = 0) = 0;
	virtual void updateStep(Eigen::MatrixXd& parameters,
							const Eigen::MatrixXd& gradients,
							const int paramIndex = 0) = 0;
	virtual void updateStep(std::vector<Eigen::MatrixXd>& parameters,
							const std::vector<Eigen::MatrixXd>& gradients,
							const int paramIndex = 0) = 0;

	void setGradientClipping(double maxNorm) {
		if (maxNorm <= 0) {
			_maxGradNorm = -1.0; // disable clipping
		}
		else {
			_maxGradNorm = maxNorm;
		}
	}
	void setWeightDecay(double lambda) {
		_weightDecay = lambda;
	}
protected:
	template <typename Derived>
	void _clipGradient(Eigen::MatrixBase<Derived>& gradient) const {
		if (_maxGradNorm <= 0.0) return; // No clipping

		double norm = gradient.norm();
		if (norm == 0.0) return; // No clipping - zero gradient

		if (norm > _maxGradNorm) {
			double scale = _maxGradNorm / std::max(norm, 1e-8);
			gradient *= scale;
		}
	}

	template <typename Derived>
	void _applyWeightDecay(Eigen::MatrixBase<Derived>& grad,
						   const Eigen::MatrixBase<Derived>& weights) const {
		if (_weightDecay > 0) {
			grad += _weightDecay * weights;
		}
	}

};

// Stochastic Gradient Descent optimizer
class SGD : public Optimizer {
public:
	SGD(double learningRate = 0.001, double maxGradNorm = -1.0, double weightDecay = 0.0)
		: Optimizer(learningRate, maxGradNorm, weightDecay)
	{
		if (_learningRate <= 0) {
			throw std::invalid_argument("[Optimizer]: Learning rate must be positive.");
		}
	}

	void updateStep(Eigen::VectorXd& parameters, 
					const Eigen::VectorXd& gradients, 
					const int paramIndex = 0) override
	{ // Vectors version

		if (parameters.size() != gradients.size()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		Eigen::VectorXd grad = gradients;
		_applyWeightDecay(grad, parameters);
		_clipGradient(grad);

		parameters -= _learningRate * grad;
	}

	void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients,
		const int paramIndex = 0) override
	{ // Matrices version

		if (parameters.rows() != gradients.rows() ||
			parameters.cols() != gradients.cols()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		Eigen::MatrixXd grad = gradients;
		_applyWeightDecay(grad, parameters);
		_clipGradient(grad);

		parameters -= _learningRate * grad;
	}

	void updateStep(std::vector<Eigen::MatrixXd>& parameters,
		const std::vector<Eigen::MatrixXd>& gradients,
		const int paramIndex = 0) override 
	{
		if (parameters.size() != gradients.size()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		Eigen::MatrixXd grad = gradients[paramIndex];
		if (parameters[paramIndex].rows() != grad.rows() ||
			parameters[paramIndex].cols() != grad.cols()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		_applyWeightDecay(grad, parameters[paramIndex]);
		_clipGradient(grad);
		parameters[paramIndex] -= _learningRate * grad;
	}
};

// Adam optimizer
class AdamOptimizer :public Optimizer {
private:
	// **Optimizer Modes**
	enum OptimizerMode {
		FullyConnected = -1,
		BatchNormalization = -2
	};
	// Adam optimizer parameters
	const double _beta1;
	const double _beta2;
	const double _epsilon;
	const int _numParams;
	size_t _numChannels = 0;
	// Time step
	size_t _timeStep = 0;
	// Initialization flag
	bool _isInitialized = false;
	bool _isScheduled = false;
	// Bias correction
	double _biasCorrection1 = 1.0;
	double _biasCorrection2 = 1.0;
	// Learning rate & Scheduler
	double _effectiveLearningRate;
	double _initialLearningRate;
	const size_t _warmupSteps = 4000;
	// Moments
	std::vector<Eigen::VectorXd> _firstMomentEstimateVector;
	std::vector<Eigen::VectorXd> _secondMomentEstimateVector;
	std::vector<Eigen::MatrixXd> _firstMomentEstimateMatrix;
	std::vector<Eigen::MatrixXd> _secondMomentEstimateMatrix;
	std::vector<std::vector<Eigen::MatrixXd>> _firstMomentEstimateTensor;
	std::vector<std::vector<Eigen::MatrixXd>> _secondMomentEstimateTensor;

public:
	AdamOptimizer(int numParams, 
		double maxGradNorm = -1.0, double weightDecay = 0.0,
		double learningRate = 0.001, double beta1 = 0.9,
		double beta2 = 0.999, double epsilon = 1.0e-8)
		: Optimizer(learningRate, maxGradNorm, weightDecay),
		_numParams(numParams),
		_beta1(beta1),
		_beta2(beta2),
		_epsilon(epsilon),
		_effectiveLearningRate(learningRate)
	{
		_validateInputParameters();
		_initialLearningRate = learningRate;
	}

	void updateStep(Eigen::VectorXd& parameters, 
					const Eigen::VectorXd& gradients, 
					const int paramIndex = 0) override
	{
		if (parameters.size() != gradients.size()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		if (!_isInitialized) {
			_initializeMoments(parameters.size());
		}
		if (paramIndex == 0 && _numParams == BatchNormalization) { //only for BN - Gamma
			_timeStep++;
			_updateEffectiveLearningRate();
		}

		// calculate moments - m_t, v_t
		_firstMomentEstimateVector[paramIndex] = _beta1 * _firstMomentEstimateVector[paramIndex].
												 array() + (1 - _beta1) * gradients.array();
		_secondMomentEstimateVector[paramIndex] = _beta2 * _secondMomentEstimateVector[paramIndex].
												 array() + (1 - _beta2) * gradients.array().square();

		Eigen::VectorXd firstMomentEstimateHat = _firstMomentEstimateVector[paramIndex] 
												 / _biasCorrection1;
		Eigen::VectorXd secondMomentEstimateHat = _secondMomentEstimateVector[paramIndex] 
												  / _biasCorrection2;

		parameters -= (_effectiveLearningRate * firstMomentEstimateHat.array() /
					  (secondMomentEstimateHat.array().sqrt() + _epsilon)).matrix();
	}

	void updateStep(Eigen::MatrixXd& parameters,
					const Eigen::MatrixXd& gradients,
					const int paramIndex = 0) override
	{
		if (parameters.rows() != gradients.rows() ||
			parameters.cols() != gradients.cols()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		if (!_isInitialized) {
			_initializeMoments(parameters.rows(), 
							   parameters.cols());
		}
		if (!paramIndex) {
			_timeStep++;
			_updateEffectiveLearningRate();
		}
		// calculate moments - m_t, v_t
		_firstMomentEstimateMatrix[paramIndex] = _beta1 * _firstMomentEstimateMatrix[paramIndex].
			array() + (1 - _beta1) * gradients.array();
		_secondMomentEstimateMatrix[paramIndex] = _beta2 * _secondMomentEstimateMatrix[paramIndex].
			array() + (1 - _beta2) * gradients.array().square();

		Eigen::MatrixXd firstMomentEstimateHat = _firstMomentEstimateMatrix[paramIndex]
			/ _biasCorrection1;
		Eigen::MatrixXd secondMomentEstimateHat = _secondMomentEstimateMatrix[paramIndex]
			/ _biasCorrection2;

		parameters -= (_effectiveLearningRate * firstMomentEstimateHat.array() /
			(secondMomentEstimateHat.array().sqrt() + _epsilon)).matrix();
	}

	void updateStep(std::vector<Eigen::MatrixXd>& parameters,
					const std::vector<Eigen::MatrixXd>& gradients,
					const int paramIndex = 0) override
	{
		if (parameters.size() != gradients.size()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		if (parameters[0].rows() != gradients[0].rows() ||
			parameters[0].cols() != gradients[0].cols()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		if (!_isInitialized) {
			_initializeMoments(parameters[paramIndex].rows(), 
							   parameters[paramIndex].cols(), 
							   parameters.size());
		}
		if (!paramIndex) {
			_timeStep++;
			_updateEffectiveLearningRate();
		}
		for (size_t c = 0; c < _numChannels; ++c) {
			// calculate moments - m_t, v_t
			_firstMomentEstimateTensor[paramIndex][c] = _beta1 * _firstMomentEstimateTensor[paramIndex][c].
				array() + (1 - _beta1) * gradients[c].array();
			_secondMomentEstimateTensor[paramIndex][c] = _beta2 * _secondMomentEstimateTensor[paramIndex][c].
				array() + (1 - _beta2) * gradients[c].array().square();

			Eigen::MatrixXd firstMomentEstimateHat = _firstMomentEstimateTensor[paramIndex][c]
				/ _biasCorrection1;
			Eigen::MatrixXd secondMomentEstimateHat = _secondMomentEstimateTensor[paramIndex][c]
				/ _biasCorrection2;
			
			parameters[c] -= (_effectiveLearningRate * firstMomentEstimateHat.array() /
				(secondMomentEstimateHat.array().sqrt() + _epsilon)).matrix();
		}
	}

private:
	void _validateInputParameters() const {
		if (_numParams != FullyConnected && _numParams != BatchNormalization &&
			_numParams <= 0) {
			throw std::invalid_argument("[Optimizer]: Invalid number of parameters.");
		}
		if (_learningRate <= 0) {
			throw std::invalid_argument("[Optimizer]: Learning rate must be positive.");
		}
		if (_beta1 <= 0 || _beta1 >= 1) {
			throw std::invalid_argument("[Optimizer]: Beta1 must be in the range (0, 1).");
		}
		if (_beta2 <= 0 || _beta2 >= 1) {
			throw std::invalid_argument("[Optimizer]: Beta2 must be in the range (0, 1).");
		}
		if (_epsilon <= 0) {
			throw std::invalid_argument("[Optimizer]: Epsilon must be positive.");
		}
	}

	void _initializeMoments(size_t rows, size_t cols = 0, size_t channels = 0)
	{
		_numChannels = (channels > 0) ? channels : 1;
		if (_numParams == FullyConnected) { //Weights and Bias
			_firstMomentEstimateMatrix.assign(1, Eigen::MatrixXd::Zero(rows, cols));
			_secondMomentEstimateMatrix.assign(1, Eigen::MatrixXd::Zero(rows, cols));
			_firstMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(rows));
			_secondMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(rows));
		}
		else if (_numParams == BatchNormalization) { //Gamma and Beta
			_firstMomentEstimateVector.assign(2, Eigen::VectorXd::Zero(rows));
			_secondMomentEstimateVector.assign(2, Eigen::VectorXd::Zero(rows));
		}
		else { //Convolution2D - Filters and Bias
			_firstMomentEstimateTensor.assign(_numParams, std::vector<Eigen::MatrixXd>(
				_numChannels, Eigen::MatrixXd::Zero(rows, cols)));
			_secondMomentEstimateTensor.assign(_numParams, std::vector<Eigen::MatrixXd>(
				_numChannels, Eigen::MatrixXd::Zero(rows, cols)));
			_firstMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(_numParams));
			_secondMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(_numParams));

		}
		_isInitialized = true;
	}

	void _updateEffectiveLearningRate() {
		if (_isScheduled) {
			_learningRateScheduler();
		}
		_applyBiasCorrection();
	}

	void _applyBiasCorrection() {
		_biasCorrection1 = 1.0 - std::pow(_beta1, _timeStep);
		_biasCorrection2 = 1.0 - std::pow(_beta2, _timeStep);
		_effectiveLearningRate = _learningRate * std::sqrt(_biasCorrection2) / _biasCorrection1;
	}

	void _learningRateScheduler()
	{
		if (_isScheduled ) {
			if (_timeStep < _warmupSteps) {
				// Linear warmup
				double warmupRatio = static_cast<double>(_timeStep) / _warmupSteps;
				_learningRate = _learningRate + warmupRatio * (_initialLearningRate - _learningRate);
			}
			else if (_timeStep % 1000 == 0) {
				// Step decay after warmup
				_learningRate *= 0.5;
			}
		}
	}
};