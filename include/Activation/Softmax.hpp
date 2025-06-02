#pragma once

#include <Eigen/Dense>
#include <vector>

#include "Activation.hpp"

template <typename T>
class Softmax : public Activation<T>
{
   private:
    T _softmaxOutput;
    bool _initialized = false;

   public:
    T Activate(const T& preActivationOutput) override;
    T computeGradient(const T& lossGradient) override;

   private:
    void _initialize(const T& input);
    T _applySoftmax(const T& input);
    T _computeSoftmaxGradient(const T& gradient);
    Eigen::MatrixXd _computeJacobian(Eigen::Ref<const Eigen::VectorXd> y) const;
};

#include "Softmax.tpp"
