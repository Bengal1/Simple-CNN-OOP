// Activation.hpp
#pragma once

template <typename T>
class Activation
{
   public:
    virtual T Activate(const T& preActivationOutput) = 0;

    virtual T computeGradient(const T& lossGradient) = 0;

    virtual ~Activation() = default;
};
