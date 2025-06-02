#include "../../include/Regularization/Dropout.hpp"

#include <stdexcept>

Dropout::Dropout(double dropoutRate)
    : _dropoutRate(dropoutRate),
      _dropoutScale(1.0 / (1.0 - dropoutRate)),
      _isTraining(true),
      _inputHeight(0),
      _inputWidth(0),
      _numChannels(0),
      _randGen(std::random_device{}()),
      _dist(0.0, 1.0)
{
    if (_dropoutRate < 0.0 || _dropoutRate >= 1.0)
    {
        throw std::invalid_argument("[Dropout]: Dropout rate must be in the range [0, 1).");
    }
}

void Dropout::setTrainingMode(bool isTraining)
{
    _isTraining = isTraining;
}

Eigen::MatrixXd Dropout::_createRandomMask()
{
    return Eigen::MatrixXd::NullaryExpr(_inputHeight, _inputWidth,
                                        [this]() { return _dist(_randGen); });
}
