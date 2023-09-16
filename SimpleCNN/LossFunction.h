#pragma once

#include <cmath>
#include <vector>
#include <Eigen/Dense>


class LossFunction {
protected:
	const double _epsilon;
public:
	LossFunction() :_epsilon(1e-15) {}
	virtual double calculateLoss(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const = 0;
	virtual Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const = 0;
	virtual ~LossFunction() {}
};

// Mean Squared Error (MSE) loss
class MSE : public LossFunction {
public:
	MSE() {}

	double calculateLoss(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets) const override {
		assert(predictions.size() == targets.size());

		double predictLoss = 0.0;
		for (int j = 0; j < predictions.size(); ++j) {
			double error = predictions[j] - targets[j];
			predictLoss += std::pow(error, 2.0);
		}

		return predictLoss / (2.0 * predictions.size());
	}

	Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const override {
		assert(predictions.size() == targets.size());

		Eigen::VectorXd gradients = predictions - targets;
		gradients = (gradients / predictions.size());  // Normalize by the number of samples in the batch

		return gradients;
	}
};

// Cross-Entropy loss
class CrossEntropy : public LossFunction {
public:
	CrossEntropy() {}

	double calculateLoss(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const override {
		assert(predictions.size() == targets.size());

		double loss = 0.0;
		for (int p = 0; p < predictions.size(); p++) {
			double probability = predictions(p);
			double target = targets(p);

			loss += -target * std::log(std::max(probability, _epsilon));
		}

		return loss;
	}

	double calculateLossBatch(const std::vector<Eigen::VectorXd>& predictionBatch,
		const std::vector<Eigen::VectorXd>& targetBatch) const {
		assert(predictionBatch.size() == targetBatch.size());

		int batchSize = predictionBatch.size();
		double batchLoss = 0.0;
		for (int b = 0; b < batchSize; b++) {
			batchLoss += calculateLoss(predictionBatch[b], targetBatch[b]);
		}

		return batchLoss;
	}

	Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
		const Eigen::VectorXd& targets) const override {
		assert(predictions.size() == targets.size());

		Eigen::VectorXd gradient(predictions.size());

		for (int p = 0; p < predictions.size(); p++) {
			double probability = predictions(p);
			double target = targets(p);

			gradient(p) = (probability - target) / (probability * (1.0 - probability) + _epsilon);
		}

		return gradient;
	}

	std::vector<Eigen::VectorXd> calculateGradientBatch(const std::vector<Eigen::VectorXd>& predictionBatch,
		const std::vector<Eigen::VectorXd>& targetBatch) const {
		assert(predictionBatch.size() == targetBatch.size());

		int batchSize = predictionBatch.size();
		std::vector<Eigen::VectorXd> gradientBatch(batchSize);

		for (int b = 0; b < batchSize; b++) {
			gradientBatch[b] = calculateGradient(predictionBatch[b], targetBatch[b]);
		}

		return gradientBatch;
	}
};
