#pragma once

#include <vector>
#include <Eigen/Dense>


class Optimizer {
protected:
	double _maxGradNorm;
	double _weightDecay;

public:
	Optimizer(double maxGradNorm = -1.0, double weightDecay = 0.0) 
		: _maxGradNorm(maxGradNorm), _weightDecay(weightDecay){}

	virtual void updateStep(Eigen::MatrixXd& parameters, 
							const Eigen::MatrixXd& gradients, 
							const int paramIndex = 0) = 0;
	virtual void updateStep(Eigen::VectorXd& parameters, 
							const Eigen::VectorXd& gradients,
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
	// Clip the gradient to prevent exploding gradients
	void _clipGradient(Eigen::MatrixXd& gradient) {
		if (_maxGradNorm > 0) {
			double norm = gradient.norm();
			if (norm > _maxGradNorm) {
				gradient *= (_maxGradNorm / norm);
			}
		}
	}

	void _clipGradient(Eigen::VectorXd& gradient) {
		if (_maxGradNorm > 0) {
			double norm = gradient.norm();
			if (norm > _maxGradNorm) {
				gradient *= (_maxGradNorm / norm);
			}
		}
	}

	// Apply weight decay to the gradients
	void _applyWeightDecay(Eigen::MatrixXd& grad, const Eigen::MatrixXd& weights) {
		if (_weightDecay > 0) {
			grad += _weightDecay * weights;
		}
	}

	void _applyWeightDecay(Eigen::VectorXd& grad, const Eigen::VectorXd& weights) {
		if (_weightDecay > 0) {
			grad += _weightDecay * weights;
		}
	}
};

// Stochastic Gradient Descent optimizer
class SGD : public Optimizer {
private:
	double _learningRate;
public:
	SGD(double learningRate = 0.001, double maxGradNorm = -1.0, double weightDecay = 0.0)
		: Optimizer(maxGradNorm, weightDecay),
		_learningRate(learningRate) 
	{
		if (_learningRate <= 0) {
			throw std::invalid_argument("[Optimizer]: Learning rate must be positive.");
		}
	}

	void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients, 
					const int paramIndex = 0) override 
	{ // Matrices version

		if (parameters.size() != gradients.size()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		Eigen::MatrixXd grad = gradients;
		_applyWeightDecay(grad, parameters);
		_clipGradient(grad);

		parameters -= _learningRate * grad;
	}

	void updateStep(Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients, 
					const int paramIndex = 0) override
	{ // Vectors version

		if (parameters.size() != gradients.size()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		Eigen::MatrixXd grad = gradients;
		_applyWeightDecay(grad, parameters);
		_clipGradient(grad);

		parameters -= _learningRate * grad;
	}

	void updateStep(std::vector<Eigen::MatrixXd>& parameters, 
					const std::vector<Eigen::MatrixXd>& gradients)
	{ // 3D Matrices version

		if (parameters.size() != gradients.size()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		
		size_t channels = parameters.size();
		for (size_t c = 0; c < channels; ++c) {
			Eigen::MatrixXd grad = gradients[c];
			_applyWeightDecay(grad, parameters[c]);
			_clipGradient(grad);

			updateStep(parameters[c], grad);
		}
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
	const double _learningRate;
	const double _beta1;
	const double _beta2;
	const double _epsilon;
	const int _numParams;
	// Time step
	size_t _timeStep;
	// Initialization flag
	bool _isInitialized;
	// Bias correction
	double _biasCorrection1;
	double _biasCorrection2;
	double _effectiveLearningRate;
	// Moments
	std::vector<Eigen::MatrixXd> _firstMomentEstimate;
	std::vector<Eigen::MatrixXd> _secondMomentEstimate;
	std::vector<Eigen::VectorXd> _firstMomentEstimateVector;
	std::vector<Eigen::VectorXd> _secondMomentEstimateVector;

public:
	AdamOptimizer(int numParams, double maxGradNorm = -1.0, double weightDecay = 0.0, 
		double learningRate = 0.001, double beta1 = 0.9,
		double beta2 = 0.999, double epsilon = 1.0e-8)
		: Optimizer(maxGradNorm, weightDecay),
		  _numParams(numParams), 
		  _learningRate(learningRate), 
		  _beta1(beta1), 
		  _beta2(beta2), 
		  _epsilon(epsilon), 
		  _timeStep(0), 
		  _isInitialized(false),
		  _biasCorrection1(1.0),
		  _biasCorrection2(1.0),
		  _effectiveLearningRate(learningRate)
	{
		_validateInputParameters();
	}

	void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients, 
					const int paramIndex = 0) override 
	{
		if(parameters.size() != gradients.size()){
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		if (!_isInitialized) {
			_initializeMoments(parameters.rows(), parameters.cols());
			_isInitialized = true;
		}

		Eigen::MatrixXd grad = gradients;
		// Apply weight decay and gradient clipping
		_applyWeightDecay(grad, parameters);
		_clipGradient(grad);

		if (!paramIndex) {
			_timeStep++;
			_updateEffectiveLearningRate();
		}
		// calculate moments - m_t, v_t
		_firstMomentEstimate[paramIndex] = _beta1 * _firstMomentEstimate[paramIndex].
										   array() + (1 - _beta1) * grad.array();
		_secondMomentEstimate[paramIndex] = _beta2 * _secondMomentEstimate[paramIndex].
									array() + (1 - _beta2) * grad.array().square();

		Eigen::MatrixXd firstMomentEstimateHat = _firstMomentEstimate[paramIndex] 
												 / _biasCorrection1;
		Eigen::MatrixXd secondMomentEstimateHat = _secondMomentEstimate[paramIndex] 
												  / _biasCorrection2;

		parameters -= (_effectiveLearningRate * firstMomentEstimateHat.array() /
					  (secondMomentEstimateHat.array().sqrt() + _epsilon)).matrix();
	}

	void updateStep(Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients, 
					const int paramIndex = 0)  override
	{
		if (parameters.size() != gradients.size()) {
			throw std::invalid_argument("[Optimizer]: Parameters and gradients must have the same size.");
		}
		if (!_isInitialized) {
			_initializeMoments(parameters.size(), parameters.size());
			_isInitialized = true;
		}

		Eigen::VectorXd grad = gradients;
		// Apply weight decay and gradient clipping
		if(_numParams == BatchNormalization && paramIndex == 0)
			_applyWeightDecay(grad, parameters);
		_clipGradient(grad);

		if (!paramIndex && _numParams == BatchNormalization) {
			_timeStep++;
			_updateEffectiveLearningRate();
		}
		// calculate moments - m_t, v_t
		_firstMomentEstimateVector[paramIndex] = _beta1 * _firstMomentEstimateVector[paramIndex].
												 array() + (1 - _beta1) * grad.array();
		_secondMomentEstimateVector[paramIndex] = _beta2 * _secondMomentEstimateVector[paramIndex].
												 array() + (1 - _beta2) * grad.array().square();

		Eigen::VectorXd firstMomentEstimateHat = _firstMomentEstimateVector[paramIndex] 
												 / _biasCorrection1;
		Eigen::VectorXd secondMomentEstimateHat = _secondMomentEstimateVector[paramIndex] 
												  / _biasCorrection2;

		parameters -= (_effectiveLearningRate * firstMomentEstimateHat.array() /
					  (secondMomentEstimateHat.array().sqrt() + _epsilon)).matrix();
	}

private:
	void _validateInputParameters() {
		if (_numParams < BatchNormalization) {
			throw std::invalid_argument("[Optimizer]: Number of parameters is not valid.");
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
		if (_numParams != FullyConnected && _numParams != BatchNormalization &&
			_numParams <= 0) {
			throw std::invalid_argument("[Optimizer]: Invalid number of parameters.");
		}
	}
	void _initializeMoments(size_t rows, size_t cols)
	{
		if (_numParams == FullyConnected) { //Fully-Connected - weights and bias
			_firstMomentEstimate.assign(1, Eigen::MatrixXd::Zero(rows, cols));
			_secondMomentEstimate.assign(1, Eigen::MatrixXd::Zero(rows, cols));
			_firstMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(rows));
			_secondMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(rows));
		}
		else if (_numParams == BatchNormalization) { //BatchNormalization - 2 vectors
			_firstMomentEstimateVector.assign(2, Eigen::VectorXd::Zero(rows));
			_secondMomentEstimateVector.assign(2, Eigen::VectorXd::Zero(rows));
		}
		else { //Convolution2D - filters and bias
			_firstMomentEstimate.assign(_numParams, Eigen::MatrixXd::Zero(rows, cols));
			_secondMomentEstimate.assign(_numParams, Eigen::MatrixXd::Zero(rows, cols));
			_firstMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(_numParams));
			_secondMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(_numParams));

		}
	}

	void _updateEffectiveLearningRate() {
		_biasCorrection1 = 1.0 - std::pow(_beta1, _timeStep);
		_biasCorrection2 = 1.0 - std::pow(_beta2, _timeStep);
		_effectiveLearningRate = _learningRate * std::sqrt(_biasCorrection2) / _biasCorrection1;
	}
};