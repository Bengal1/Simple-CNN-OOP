#pragma once

#include <cmath>
#include <vector>
#include <Eigen/Dense>


class LossFunction {
protected:
	const double _epsilon;
public:
	LossFunction() :_epsilon(1.0e-10) {}
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

	double calculateLoss(const Eigen::VectorXd& predictions,
						 const Eigen::VectorXd& targets) const override {

		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}

		size_t classNum = predictions.size();
		double loss = 0.0;
		for (size_t c = 0; c < classNum; ++c) {

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

		size_t batchSize = predictionBatch.size();
		double batchLoss = 0.0;
		for (size_t b = 0; b < batchSize; ++b) {
			batchLoss += calculateLoss(predictionBatch[b], targetBatch[b]);
		}

		return batchLoss;
	}

	Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
									  const Eigen::VectorXd& targets) const override 
	{
		if(predictions.size() != targets.size()){
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}
		return (predictions - targets);
	}

	/*Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const override {

		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}
		
		size_t classNum = predictions.size();
		Eigen::VectorXd gradientCE(classNum);

		for (size_t c = 0; c < classNum; ++c) {
			gradientCE(c) = -targets(c) / (predictions(c) +_epsilon);
		}
		
		return gradientCE;
	}*/

	std::vector<Eigen::VectorXd> calculateGradientBatch(
		const std::vector<Eigen::VectorXd>& predictionBatch, 
		const std::vector<Eigen::VectorXd>& targetBatch) const 
	{

		if (predictionBatch.size() != targetBatch.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}

		size_t batchSize = predictionBatch.size();
		size_t classNum = predictionBatch[0].size();
		std::vector<Eigen::VectorXd> gradientBatch(batchSize, 
									Eigen::VectorXd(classNum));

		for (size_t b = 0; b < batchSize; ++b) {
			gradientBatch[b] = calculateGradient(predictionBatch[b], 
				targetBatch[b]);
		}

		return gradientBatch;
	}
};
