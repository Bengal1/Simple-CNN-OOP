#include "../../include/Optimizer/SGD.hpp"

SGD::SGD(double learningRate, double maxGradNorm, double weightDecay)
    : Optimizer(learningRate, maxGradNorm, weightDecay)
{
    if (_learningRate <= 0)
    {
        throw std::invalid_argument("[Optimizer]: Learning rate must be positive.");
    }
}

void SGD::updateStep(Eigen::VectorXd& parameters, const Eigen::VectorXd& gradients, const int)
{
    if (parameters.size() != gradients.size())
    {
        throw std::invalid_argument(
            "[Optimizer]: Parameters and gradients must have the same size.");
    }
    Eigen::VectorXd grad = gradients;
    _applyWeightDecay(grad, parameters);
    _clipGradient(grad);
    parameters -= _learningRate * grad;
}

void SGD::updateStep(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients, const int)
{
    if (parameters.rows() != gradients.rows() || parameters.cols() != gradients.cols())
    {
        throw std::invalid_argument(
            "[Optimizer]: Parameters and gradients must have the same size.");
    }
    Eigen::MatrixXd grad = gradients;
    _applyWeightDecay(grad, parameters);
    _clipGradient(grad);
    parameters -= _learningRate * grad;
}

void SGD::updateStep(std::vector<Eigen::MatrixXd>& parameters,
                     const std::vector<Eigen::MatrixXd>& gradients, const int paramIndex)
{
    if (parameters.size() != gradients.size())
    {
        throw std::invalid_argument(
            "[Optimizer]: Parameters and gradients must have the same size.");
    }
    Eigen::MatrixXd grad = gradients[paramIndex];
    if (parameters[paramIndex].rows() != grad.rows() ||
        parameters[paramIndex].cols() != grad.cols())
    {
        throw std::invalid_argument(
            "[Optimizer]: Parameters and gradients must have the same size.");
    }
    _applyWeightDecay(grad, parameters[paramIndex]);
    _clipGradient(grad);
    parameters[paramIndex] -= _learningRate * grad;
}
