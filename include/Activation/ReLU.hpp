#pragma once

#include <Eigen/Dense>
#include <vector>

#include "Activation.hpp"

template <typename T>
class ReLU : public Activation<T>
{
   private:
    T _reluInput;
    T _reluGradient;
    bool _initialized = false;

   public:
    T Activate(const T& preActivationOutput) override;
    T computeGradient(const T& lossGradient) override;

   private:
    void _initialize(const T& input);
    T _applyReLU(const T& input);
    T _computeReLUGradient(const T& gradient);
};

#include "ReLU.tpp"
