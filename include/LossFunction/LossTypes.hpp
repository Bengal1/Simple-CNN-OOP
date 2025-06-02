// LossTypes.hpp
#pragma once

#include <algorithm>
#include <cmath>

#include "LossFunction.hpp"

// Mean Squared Error (MSE) Loss
class MSE : public LossFunction
{
   public:
    double calculateLoss(const Eigen::VectorXd& predictions,
                         const Eigen::VectorXd& targets) const override;

    Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
                                      const Eigen::VectorXd& targets) const override;
};

// Cross-Entropy Loss
class CrossEntropy : public LossFunction
{
   public:
    explicit CrossEntropy(double epsilon = 1.0e-10);

    double calculateLoss(const Eigen::VectorXd& predictions,
                         const Eigen::VectorXd& targets) const override;

    Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
                                      const Eigen::VectorXd& targets) const override;

    double calculateLossBatch(const std::vector<Eigen::VectorXd>& predictionBatch,
                              const std::vector<Eigen::VectorXd>& targetBatch) const;

    std::vector<Eigen::VectorXd> calculateGradientBatch(
        const std::vector<Eigen::VectorXd>& predictionBatch,
        const std::vector<Eigen::VectorXd>& targetBatch) const;

    Eigen::VectorXd softmaxCrossEntropyGradient(const Eigen::VectorXd& predictions,
                                                const Eigen::VectorXd& targets) const;
};
