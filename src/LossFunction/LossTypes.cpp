#include "../../include/LossFunction/LossTypes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// ---------- MSE Implementation ----------

double MSE::calculateLoss(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets) const
{
    if (predictions.size() != targets.size())
    {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }

    double loss = 0.0;
    for (int i = 0; i < predictions.size(); ++i)
    {
        double error = predictions[i] - targets[i];
        loss += error * error;
    }
    return loss / (2.0 * predictions.size());
}

Eigen::VectorXd MSE::calculateGradient(const Eigen::VectorXd& predictions,
                                       const Eigen::VectorXd& targets) const
{
    if (predictions.size() != targets.size())
    {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }
    return (predictions - targets) / predictions.size();
}

// ---------- CrossEntropy Implementation ----------

CrossEntropy::CrossEntropy(double epsilon) : LossFunction(epsilon) {}

double CrossEntropy::calculateLoss(const Eigen::VectorXd& predictions,
                                   const Eigen::VectorXd& targets) const
{
    if (predictions.size() != targets.size())
    {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }

    double loss = 0.0;
    for (int i = 0; i < predictions.size(); ++i)
    {
        loss -= targets(i) * std::log(std::max(predictions(i), _epsilon));
    }
    return loss;
}

Eigen::VectorXd CrossEntropy::calculateGradient(const Eigen::VectorXd& predictions,
                                                const Eigen::VectorXd& targets) const
{
    if (predictions.size() != targets.size())
    {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }

    Eigen::VectorXd grad(predictions.size());
    for (int i = 0; i < predictions.size(); ++i)
    {
        grad(i) = -targets(i) / std::max(predictions(i), _epsilon);
    }
    return grad;
}

double CrossEntropy::calculateLossBatch(const std::vector<Eigen::VectorXd>& predictionBatch,
                                        const std::vector<Eigen::VectorXd>& targetBatch) const
{
    if (predictionBatch.size() != targetBatch.size())
    {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }

    double batchLoss = 0.0;
    for (size_t i = 0; i < predictionBatch.size(); ++i)
    {
        batchLoss += calculateLoss(predictionBatch[i], targetBatch[i]);
    }
    return batchLoss;
}

std::vector<Eigen::VectorXd> CrossEntropy::calculateGradientBatch(
    const std::vector<Eigen::VectorXd>& predictionBatch,
    const std::vector<Eigen::VectorXd>& targetBatch) const
{
    if (predictionBatch.size() != targetBatch.size())
    {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }

    std::vector<Eigen::VectorXd> gradients;
    gradients.reserve(predictionBatch.size());

    for (size_t i = 0; i < predictionBatch.size(); ++i)
    {
        gradients.push_back(calculateGradient(predictionBatch[i], targetBatch[i]));
    }
    return gradients;
}

Eigen::VectorXd CrossEntropy::softmaxCrossEntropyGradient(const Eigen::VectorXd& predictions,
                                                          const Eigen::VectorXd& targets) const
{
    if (predictions.size() != targets.size())
    {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }
    return predictions - targets;
}
