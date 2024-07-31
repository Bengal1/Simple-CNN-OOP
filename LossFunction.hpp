#ifndef LOSSFUNCTION_HPP
#define LOSSFUNCTION_HPP

#include <cmath>
#include <vector>
#include <Eigen/Dense>


class LossFunction {
protected:
	const double _epsilon;
public:
	LossFunction() :_epsilon(1.0e-8) {}
	virtual double calculateLoss(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const = 0;
	virtual Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const = 0;
	virtual ~LossFunction() {}
};

// Mean Squared Error (MSE) loss
class MSE : public LossFunction {
public:

	double calculateLoss(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const override {
		assert(predictions.size() == targets.size());

		size_t classNum = predictions.size();
		double predictLoss = 0.0;
		for (size_t c = 0; c < classNum; ++c) {
			double error = predictions[c] - targets[c];
			predictLoss += std::pow(error, 2.0);
		}

		return predictLoss / (2.0 * predictions.size());
	}

	Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const override {
		assert(predictions.size() == targets.size());

		Eigen::VectorXd gradientMSE = predictions - targets;
		gradientMSE = (gradientMSE / predictions.size());

		return gradientMSE;
	}
};

// Cross-Entropy loss
class CrossEntropy : public LossFunction {
public:

	double calculateLoss(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const override {

		assert(predictions.size() == targets.size());
		size_t classNum = predictions.size();

		double loss = 0.0;
		for (size_t c = 0; c < classNum; ++c) {

			loss -= targets(c) * std::log(std::max(predictions(c), _epsilon));
		}

		return loss / classNum;
	}

	Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const override {

		assert(predictions.size() == targets.size());

		size_t classNum = predictions.size();
		Eigen::VectorXd gradientCE(classNum);

		for (size_t c = 0; c < classNum; ++c) {

			gradientCE(c) = -targets(c) / (predictions(c) + _epsilon);
		}

		return gradientCE;
	}
};

#endif // LOSSFUNCTION_HPP