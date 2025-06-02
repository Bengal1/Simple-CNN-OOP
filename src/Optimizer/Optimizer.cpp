#include "../../include/Optimizer/Optimizer.hpp"

Optimizer::Optimizer(double learningRate, double maxGradNorm, double weightDecay)
    : _learningRate(learningRate), _maxGradNorm(maxGradNorm), _weightDecay(weightDecay)
{
}

void Optimizer::setGradientClipping(double maxNorm)
{
    _maxGradNorm = (maxNorm <= 0) ? -1.0 : maxNorm;
}

void Optimizer::setWeightDecay(double lambda)
{
    _weightDecay = lambda;
}
