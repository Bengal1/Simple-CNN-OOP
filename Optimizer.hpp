#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <Eigen/Dense>
#include <vector>
#include <cassert>
#include <stdexcept>


enum OptimizerMode {
	FullyConnectedMode = -1,
	BatchNormalizationMode = -2
};

class Optimizer {
public:
	virtual void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd&
		gradients, const int paramIndex = 0) = 0;
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
		if (learningRate <= 0) {
            throw std::invalid_argument("Learning rate must be positive.");
        }
	}

	void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients,
		const int paramIndex = 0) override 
	{
		validateSize(parameters, gradients);
		parameters -= _learningRate * gradients;
	}

	void updateStep(Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients, 
		const int paramIndex = 0) const 
	{
		validateSize(parameters, gradients);
		parameters -= _learningRate * gradients;
	}

private:
    void validateSize(const Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients) const
    {
        if (parameters.rows() != gradients.rows() || parameters.cols() != gradients.cols()) {
            throw std::invalid_argument("Parameter and gradient sizes must match.");
        }
    }

    void validateSize(const Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients) const
    {
        if (parameters.size() != gradients.size()) {
            throw std::invalid_argument("Parameter and gradient sizes must match.");
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

	std::vector<Eigen::MatrixXd> _firstMomentEstimate;
	std::vector<Eigen::MatrixXd> _secondMomentEstimate;
	std::vector<Eigen::VectorXd> _firstMomentEstimateVector;
	std::vector<Eigen::VectorXd> _secondMomentEstimateVector;

public:
	AdamOptimizer(int numParams, double learningRate = 1e-5, double beta1 = 0.9,
				  double beta2 = 0.999, double epsilon = 1.0e-8)
		: _numParams(numParams),
		  _learningRate(learningRate),
		  _beta1(beta1),
		  _beta2(beta2),
		  _epsilon(epsilon), 
		  _timeStep(0) 
	{
		if (_numParams != FullyConnectedMode && _numParams != BatchNormalizationMode && _numParams <= 0) {
            throw std::invalid_argument(
				"Number of parameters must be positive or use special mode values.");
        }
	}

	void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients,
		const int paramIndex = 0) override 
	{
		validateSize(parameters, gradients);

		if (_timeStep == 0) {
			_initializeMoments(parameters.rows(), parameters.cols());
		}
		if (!paramIndex) {
			_timeStep++;
		}

		double biasCorrection1 = 1.0 - std::pow(_beta1, _timeStep);
		double biasCorrection2 = 1.0 - std::pow(_beta2, _timeStep);

		// calculate moments - m_t, v_t
		_firstMomentEstimate[paramIndex] = _beta1 * _firstMomentEstimate[paramIndex].
										array() + (1 - _beta1) * gradients.array();
		_secondMomentEstimate[paramIndex] = _beta2 * _secondMomentEstimate[paramIndex].
								array() + (1 - _beta2) * gradients.array().square();

		Eigen::MatrixXd firstMomentEstimateHat = _firstMomentEstimate[paramIndex]
												/ biasCorrection1;
		Eigen::MatrixXd secondMomentEstimateHat = _secondMomentEstimate[paramIndex]
												/ biasCorrection2;

		parameters -= (_learningRate * firstMomentEstimateHat.array() /
				(secondMomentEstimateHat.array().sqrt() + _epsilon)).matrix();
	}

	void updateStep(Eigen::VectorXd& parameters, Eigen::VectorXd& gradients,
		const int paramIndex = 0) 
	{ //Vector version - overlaod
		validateSize(parameters, gradients);

		if (_timeStep == 0) {
			_initializeMoments(parameters.size(), parameters.size());
		}
		if (!paramIndex and _numParams == BatchNormalizationMode) {
			_timeStep++;
		}

		double biasCorrection1 = 1.0 - std::pow(_beta1, _timeStep);
		double biasCorrection2 = 1.0 - std::pow(_beta2, _timeStep);

		_firstMomentEstimateVector[paramIndex] = _beta1 * _firstMomentEstimateVector[paramIndex].
												 array() + (1 - _beta1) * gradients.array();
		_secondMomentEstimateVector[paramIndex] = _beta2 * _secondMomentEstimateVector[paramIndex].
												  array() + (1 - _beta2) * gradients.array().square();

		Eigen::VectorXd firstMomentEstimateHat = _firstMomentEstimateVector[paramIndex]
												/ biasCorrection1;
		Eigen::VectorXd secondMomentEstimateHat = _secondMomentEstimateVector[paramIndex]
												/ biasCorrection2;

		parameters -= (_learningRate * firstMomentEstimateHat.array() /
			(secondMomentEstimateHat.array().sqrt() + _epsilon)).matrix();
	}

private:
	void validateSize(const Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients) const
    	{
        	if (parameters.rows() != gradients.rows() || parameters.cols() != gradients.cols()) {
            	throw std::invalid_argument("Parameter and gradient sizes must match.");
        	}
    	}

    	void validateSize(const Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients) const
	{
	        if (parameters.size() != gradients.size()) {
	            throw std::invalid_argument("Parameter and gradient sizes must match.");
	        }
	}

	void _initializeMoments(size_t rows, size_t cols = 0) 
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
			_firstMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(_numParams));
			_secondMomentEstimateVector.assign(1, Eigen::VectorXd::Zero(_numParams));
		}
	}
};

#endif // OPTIMIZER_HPP
