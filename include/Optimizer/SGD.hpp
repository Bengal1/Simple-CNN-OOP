#pragma once

#include "Optimizer.hpp"

class SGD : public Optimizer
{
   public:
    SGD(double learningRate = 0.001, double maxGradNorm = -1.0, double weightDecay = 0.0);

    void updateStep(Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients,
                    const int paramIndex = 0) override;

    void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients,
                    const int paramIndex = 0) override;

    void updateStep(std::vector<Eigen::MatrixXd>& parameters,
                    const std::vector<Eigen::MatrixXd>& gradients,
                    const int paramIndex = 0) override;
};