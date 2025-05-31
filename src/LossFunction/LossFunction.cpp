#include "../../include/LossFunction/LossFunction.hpp"

#include <stdexcept>

LossFunction::LossFunction() : _epsilon(1.0e-8) {}

LossFunction::LossFunction(double epsilon) : _epsilon(epsilon)
{
    if (epsilon <= 0.0)
    {
        throw std::invalid_argument("Epsilon must be greater than zero.");
    }
}
