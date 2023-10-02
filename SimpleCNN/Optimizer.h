#pragma once

#include <vector>
#include <Eigen/Dense>


class Optimizer {
public:
	virtual void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients, const int paramIndex = 0) = 0;
	virtual ~Optimizer() {}
};

// Stochastic Gradient Descent optimizer
class SGD : public Optimizer {
private:
	double _learningRate;
public:
	SGD(double learningRate)
		:_learningRate(learningRate) {}


	void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients, const int paramIndex = 0) override {
		parameters -= _learningRate * gradients;
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

	std::vector<Eigen::MatrixXd> _firstMomentEstimates;
	std::vector<Eigen::MatrixXd> _secondMomentEstimates;
	std::vector<Eigen::VectorXd> _biasFirstMomentEstimates;
	std::vector<Eigen::VectorXd> _biasSecondMomentEstimates;

public:
	AdamOptimizer(int numParams, double learningRate = 0.01, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-10)
		: _numParams(numParams), _learningRate(learningRate), _beta1(beta1), _beta2(beta2), _epsilon(epsilon), _timeStep(0) {}

	void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients, const int paramIndex = 0) override {

		assert(parameters.size() == gradients.size());

		if (!_timeStep) {
			_initializeMoments(parameters.rows(), parameters.cols());
		}
		if (!paramIndex) {
			_timeStep++;
		}

		double biasCorrection1 = 1.0 - std::pow(_beta1, _timeStep);
		double biasCorrection2 = 1.0 - std::pow(_beta2, _timeStep);
		double lr_t = _learningRate * std::sqrt(biasCorrection2) / biasCorrection1;

		_firstMomentEstimates[paramIndex] = _beta1 * _firstMomentEstimates[paramIndex].array() + (1 - _beta1) * gradients.array();
		_secondMomentEstimates[paramIndex] = _beta2 * _secondMomentEstimates[paramIndex].array() + (1 - _beta2) * gradients.array().square();

		Eigen::MatrixXd firstMomentEstimateHat = _firstMomentEstimates[paramIndex] / biasCorrection1;
		Eigen::MatrixXd secondMomentEstimateHat = _secondMomentEstimates[paramIndex] / biasCorrection2;

		parameters -= (lr_t * firstMomentEstimateHat.array() / (secondMomentEstimateHat.array().sqrt() + _epsilon)).matrix();
	}

	void updateStep(Eigen::VectorXd& parameters, Eigen::VectorXd& gradients, const int paramIndex = 0) { //bias version - overlaod

		assert(parameters.size() == gradients.size());

		double biasCorrection1 = 1.0 - std::pow(_beta1, _timeStep);
		double biasCorrection2 = 1.0 - std::pow(_beta2, _timeStep);
		double lr_t = _learningRate * std::sqrt(biasCorrection2) / biasCorrection1;

		_biasFirstMomentEstimates[paramIndex] = _beta1 * _biasFirstMomentEstimates[paramIndex].array() + (1 - _beta1) * gradients.array();
		_biasSecondMomentEstimates[paramIndex] = _beta2 * _biasSecondMomentEstimates[paramIndex].array() + (1 - _beta2) * gradients.array().square();

		Eigen::VectorXd firstMomentEstimateHat = _biasFirstMomentEstimates[paramIndex] / biasCorrection1;
		Eigen::VectorXd secondMomentEstimateHat = _biasSecondMomentEstimates[paramIndex] / biasCorrection2;

		parameters -= (lr_t * firstMomentEstimateHat.array() / (secondMomentEstimateHat.array().sqrt() + _epsilon)).matrix();
	}

private:
	void _initializeMoments(int rows, int cols) {
		if (_numParams == -1) { //fully-Connected - weights and bias
			_firstMomentEstimates.assign(1, Eigen::MatrixXd::Zero(rows, cols));
			_secondMomentEstimates.assign(1, Eigen::MatrixXd::Zero(rows, cols));
			_biasFirstMomentEstimates.assign(1, Eigen::VectorXd::Zero(rows));
			_biasSecondMomentEstimates.assign(1, Eigen::VectorXd::Zero(rows));
		}
		else {
			_firstMomentEstimates.resize(_numParams);
			_secondMomentEstimates.resize(_numParams);

			for (int i = 0; i < _numParams; ++i) {
				_firstMomentEstimates[i] = Eigen::MatrixXd::Zero(rows, cols);
				_secondMomentEstimates[i] = Eigen::MatrixXd::Zero(rows, cols);
			}
		}
	}
};