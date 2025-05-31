#pragma once

#include <Eigen/Dense>
#include <vector>

class Optimizer
{
   protected:
    double _learningRate;
    double _maxGradNorm;
    double _weightDecay;

   public:
    Optimizer(double learningRate = 0.001, double maxGradNorm = -1.0, double weightDecay = 0.0);
    virtual ~Optimizer() = default;

    virtual void updateStep(Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients,
                            const int paramIndex = 0) = 0;
    virtual void updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients,
                            const int paramIndex = 0) = 0;
    virtual void updateStep(std::vector<Eigen::MatrixXd>& parameters,
                            const std::vector<Eigen::MatrixXd>& gradients,
                            const int paramIndex = 0) = 0;

    void setGradientClipping(double maxNorm);

    void setWeightDecay(double lambda);

   protected:
    template <typename Derived>
    void _clipGradient(Eigen::MatrixBase<Derived>& gradient) const;

    template <typename Derived>
    void _applyWeightDecay(Eigen::MatrixBase<Derived>& grad,
                           const Eigen::MatrixBase<Derived>& weights) const;
};

#include "Optimizer.tpp"