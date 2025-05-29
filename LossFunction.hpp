#pragma once

#include <cmath>
#include <vector>
#include <Eigen/Dense>


class LossFunction {
protected:
	const double _epsilon;
public:
	LossFunction() :_epsilon(1.0e-8) {}
	LossFunction(double epsilon) : _epsilon(epsilon) {
		if (epsilon <= 0.0) {
			throw std::invalid_argument("Epsilon must be greater than zero.");
		}
	}

	virtual double calculateLoss(const Eigen::VectorXd& predictions,
								 const Eigen::VectorXd& targets) const = 0;
	virtual Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
											  const Eigen::VectorXd& targets) const = 0;
	virtual ~LossFunction() = default;
};

// Mean Squared Error (MSE) loss
class MSE : public LossFunction {
public:

	double calculateLoss(const Eigen::VectorXd& predictions, 
						 const Eigen::VectorXd& targets) const override
	{
		if(predictions.size() != targets.size()){
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}

		size_t classNum = predictions.size();
		double predictLoss = 0.0;
		for (size_t c = 0; c < classNum; ++c) {
			double error = predictions[c] - targets[c];
			predictLoss += std::pow(error, 2.0);
		}

		return predictLoss / (2.0 * predictions.size());
	}

	Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
									  const Eigen::VectorXd& targets) const override 
	{
		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}

		Eigen::VectorXd gradientMSE = predictions - targets;
		gradientMSE = (gradientMSE / predictions.size());

		return gradientMSE;
	}
};

// Cross-Entropy loss 
class CrossEntropy : public LossFunction {
public:
	CrossEntropy(double epsilon = 1.0e-10) : LossFunction(epsilon) {}

	double calculateLoss(const Eigen::VectorXd& predictions,
						 const Eigen::VectorXd& targets) const override {

		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}

		double loss = 0.0;
		for (int c = 0; c < predictions.size(); ++c) {
			loss -= targets(c) * std::log(std::max(predictions(c), _epsilon));
		}

		return loss;
	}

	double calculateLossBatch(const std::vector<Eigen::VectorXd>& predictionBatch,
							  const std::vector<Eigen::VectorXd>& targetBatch) const 
	{
		if (predictionBatch.size() != targetBatch.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}

		double batchLoss = 0.0;
		for (size_t b = 0; b < predictionBatch.size(); ++b) {
			batchLoss += calculateLoss(predictionBatch[b], targetBatch[b]);
		}

		return batchLoss;
	}

	Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
									  const Eigen::VectorXd& targets) const override
	{
		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}

		Eigen::VectorXd gradient(predictions.size());
		for (int i = 0; i < predictions.size(); ++i) {
			gradient(i) = -targets(i) / std::max(predictions(i), _epsilon);
		}

		return gradient;
	}

	Eigen::VectorXd softmaxCrossEntropyGradient(const Eigen::VectorXd& predictions,
												const Eigen::VectorXd& targets) const
	{
		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}
		return (predictions - targets);
	}

	std::vector<Eigen::VectorXd> calculateGradientBatch(
		const std::vector<Eigen::VectorXd>& predictionBatch,
		const std::vector<Eigen::VectorXd>& targetBatch) const 
	{
		if (predictionBatch.size() != targetBatch.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}

		std::vector<Eigen::VectorXd> gradientBatch;
		gradientBatch.reserve(predictionBatch.size());

		for (size_t b = 0; b < predictionBatch.size(); ++b) {
			gradientBatch.push_back(calculateGradient(predictionBatch[b], targetBatch[b]));
		}

		return gradientBatch;
	}
};
