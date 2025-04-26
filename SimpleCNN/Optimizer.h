#pragma once

#include <vector>
#include <Eigen/Dense>

enum {
	FullyConnectedMode = -1,
	BatchNormalizationMode = -2
};

class Optimizer {
public:
	virtual void updateStep(Eigen::MatrixXd& parameters, 
							const Eigen::MatrixXd& gradients, 
							const int paramIndex = 0) = 0;
	virtual void updateStep(Eigen::VectorXd& parameters, 
							const Eigen::VectorXd& gradients,
							const int paramIndex = 0) = 0;

	virtual ~Optimizer() {}
};

// Stochastic Gradient Descent optimizer
class SGD : public Optimizer {
private:
	double _learningRate;
public:
	SGD(double learningRate = 0.001)
		:_learningRate(learningRate) 
	{
		if (_learningRate <= 0) {
			throw std::invalid_argument("Learning rate must be positive.");
		}
	}

	void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients, 
		const int paramIndex = 0) override 
	{ // Matrices version

		if (parameters.size() != gradients.size()) {
			throw std::invalid_argument("Parameters and gradients must have the same size.");
		}

		parameters -= _learningRate * gradients;
	}

	void updateStep(Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients, 
					const int paramIndex = 0) override
	{ // Vectors version

		if (parameters.size() != gradients.size()) {
			throw std::invalid_argument("Parameters and gradients must have the same size.");
		}

		parameters -= _learningRate * gradients;
	}

	void updateStep(std::vector<Eigen::MatrixXd>& parameters, 
					const std::vector<Eigen::MatrixXd>& gradients)
	{ // 3D Matrices version

		assert(parameters.size() == gradients.size());
		int channels = parameters.size();

		for (int c = 0; c < channels; c++) {
			updateStep(parameters[c], gradients[c]);
		}
	}
};

// Adam optimizer
class AdamOptimizer :public Optimizer {
private:
	const double _learningRate;
	const double _beta1;
	const double _beta2;
	const double _epsilon;
	const int _numParams;
	
	size_t _timeStep;
	
	bool _isInitialized;

	double _biasCorrection1 = 1.0;
	double _biasCorrection2 = 1.0;
	double _effectiveLearningRate = 0.0;

	std::vector<Eigen::MatrixXd> _firstMomentEstimate;
	std::vector<Eigen::MatrixXd> _secondMomentEstimate;
	std::vector<Eigen::VectorXd> _firstMomentEstimateVector;
	std::vector<Eigen::VectorXd> _secondMomentEstimateVector;

public:
	AdamOptimizer(int numParams, double learningRate = 0.001, double beta1 = 0.9, 
		double beta2 = 0.999, double epsilon = 1.0e-10)
		: _numParams(numParams), _learningRate(learningRate), _beta1(beta1), 
		_beta2(beta2), _epsilon(epsilon), _timeStep(0), _isInitialized(false) 
	{
		if (_numParams < BatchNormalizationMode) {
			throw std::invalid_argument("Number of parameters is not valid.");
		}
		if (_learningRate <= 0) {
			throw std::invalid_argument("Learning rate must be positive.");
		}
		if (_beta1 <= 0 || _beta1 >= 1) {
			throw std::invalid_argument("Beta1 must be in the range (0, 1).");
		}
		if (_beta2 <= 0 || _beta2 >= 1) {
			throw std::invalid_argument("Beta2 must be in the range (0, 1).");
		}
		if (_epsilon <= 0) {
			throw std::invalid_argument("Epsilon must be positive.");
		}
	}

	void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients, 
					const int paramIndex = 0) override 
	{
		if(parameters.size() != gradients.size()){
			throw std::invalid_argument("Parameters and gradients must have the same size.");
		}
		if (!_isInitialized) {
			_initializeMoments(parameters.rows(), parameters.cols());
			_isInitialized = true;
		}
		if (!paramIndex) {
			_timeStep++;
		}

		double biasCorrection1 = 1.0 - std::pow(_beta1, _timeStep);
		double biasCorrection2 = 1.0 - std::pow(_beta2, _timeStep);
		double lr_t = _learningRate * std::sqrt(biasCorrection2) / biasCorrection1;
		
		// calculate moments - m_t, v_t
		_firstMomentEstimate[paramIndex] = _beta1 * _firstMomentEstimate[paramIndex].
										   array() + (1 - _beta1) * gradients.array();
		_secondMomentEstimate[paramIndex] = _beta2 * _secondMomentEstimate[paramIndex].
									array() + (1 - _beta2) * gradients.array().square();

		Eigen::MatrixXd firstMomentEstimateHat = _firstMomentEstimate[paramIndex] 
												 / biasCorrection1;
		Eigen::MatrixXd secondMomentEstimateHat = _secondMomentEstimate[paramIndex] 
												  / biasCorrection2;

		parameters -= (lr_t * firstMomentEstimateHat.array() / 
			(secondMomentEstimateHat.array().sqrt() + _epsilon)).matrix();
	}

	void updateStep(Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients, 
					const int paramIndex = 0)  override
	{ //Vector version - overlaod
		if (parameters.size() != gradients.size()) {
			throw std::invalid_argument("Parameters and gradients must have the same size.");
		}
		if (!_isInitialized) {
			_initializeMoments(parameters.size(), parameters.size());
			_isInitialized = true;
		}
		if (!paramIndex && _numParams != FullyConnectedMode) {
			_timeStep++;
			_updateEffectiveLearningRate();
		}

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

private:
	void _initializeMoments(int rows, int cols) 
	{
		if (_numParams == FullyConnectedMode) { //Fully-Connected - weights and bias
			_firstMomentEstimate.assign(1, Eigen::MatrixXd::Zero(rows, cols));
			_secondMomentEstimate.assign(1, Eigen::MatrixXd::Zero(rows, cols));
			_firstMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(rows));
			_secondMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(rows));
		}
		else if (_numParams == BatchNormalizationMode) { //BatchNormalization - 2 vectors
			_firstMomentEstimateVector.assign(2, Eigen::VectorXd::Zero(rows));
			_secondMomentEstimateVector.assign(2, Eigen::VectorXd::Zero(rows));
		}
		else { //Convolution2D - filters
			_firstMomentEstimate.resize(_numParams);
			_secondMomentEstimate.resize(_numParams);

			for (int i = 0; i < _numParams; ++i) {
				_firstMomentEstimate[i] = Eigen::MatrixXd::Zero(rows, cols);
				_secondMomentEstimate[i] = Eigen::MatrixXd::Zero(rows, cols);
			}
		}
	}

	void _updateEffectiveLearningRate() {
		_biasCorrection1 = 1.0 - std::pow(_beta1, _timeStep);
		_biasCorrection2 = 1.0 - std::pow(_beta2, _timeStep);
		_effectiveLearningRate = _learningRate * std::sqrt(_biasCorrection2) / _biasCorrection1;
	}

};