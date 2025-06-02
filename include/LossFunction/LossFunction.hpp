// LossFunction.hpp
#pragma once

#include <Eigen/Dense>
#include <vector>

class LossFunction
{
   protected:
    const double _epsilon;

   public:
    LossFunction();
    explicit LossFunction(double epsilon);
    virtual ~LossFunction() = default;

    virtual double calculateLoss(const Eigen::VectorXd& predictions,
                                 const Eigen::VectorXd& targets) const = 0;

    virtual Eigen::VectorXd calculateGradient(const Eigen::VectorXd& predictions,
                                              const Eigen::VectorXd& targets) const = 0;
};
