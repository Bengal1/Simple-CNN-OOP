// Dropout.hpp
#pragma once

#include <Eigen/Dense>
#include <random>
#include <vector>

class Dropout
{
   private:
    // Default parameters
    static constexpr double DefaultDropoutRate = 0.5;
    size_t _inputHeight;
    size_t _inputWidth;
    size_t _numChannels;

    const double _dropoutRate;
    const double _dropoutScale;

    bool _isTraining;

    Eigen::MatrixXd _dropoutMask;
    std::vector<Eigen::MatrixXd> _dropoutMask3D;

    std::mt19937 _randGen;
    std::uniform_real_distribution<double> _dist;

   public:
    explicit Dropout(double dropoutRate = DefaultDropoutRate);
    ~Dropout() = default;

    template <typename T>
    T forward(const T& input);

    template <typename T>
    T backward(const T& dOutput);

    void setTrainingMode(bool isTraining);

   private:
    Eigen::MatrixXd _createRandomMask();

    template <typename T>
    void _getInputDimensions(const T& input);
};

#include "Dropout.tpp"
