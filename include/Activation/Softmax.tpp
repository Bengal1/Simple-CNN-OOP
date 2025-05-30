#include <stdexcept>

template<typename T>
T Softmax<T>::Activate(const T& preActivationOutput) {
    if (!_initialized) {
        _initialize(preActivationOutput);
    }
    _softmaxOutput = _applySoftmax(preActivationOutput);
    return _softmaxOutput;
}

template<typename T>
T Softmax<T>::computeGradient(const T& lossGradient) {
    return _computeSoftmaxGradient(lossGradient);
}

template<typename T>
void Softmax<T>::_initialize(const T& input) {
    if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
        _softmaxOutput = Eigen::VectorXd::Zero(input.size());
    } else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
        _softmaxOutput = Eigen::MatrixXd::Zero(input.rows(), input.cols());
    }
    _initialized = true;
}

template<typename T>
T Softmax<T>::_applySoftmax(const T& input) {
    if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
        if (input.size() != _softmaxOutput.size()) {
            throw std::invalid_argument("[Softmax]: Input size does not match.");
        }
        Eigen::VectorXd exp = input.array().exp();
        return exp / exp.sum();
    } else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
        if (input.rows() != _softmaxOutput.rows() || input.cols() != _softmaxOutput.cols()) {
            throw std::invalid_argument("[Softmax]: Input dimensions do not match.");
        }
        Eigen::MatrixXd exp = input.array().exp();
        return exp.array().colwise() / exp.rowwise().sum().array();
    } else {
        throw std::invalid_argument("[Softmax]: Unsupported input type.");
    }
}

template<typename T>
T Softmax<T>::_computeSoftmaxGradient(const T& gradient) {
    if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
        const Eigen::MatrixXd Jacob = _computeJacobian(_softmaxOutput);
        return Jacob.transpose() * gradient;
    } else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
        Eigen::MatrixXd dSoftmax_dx(gradient.rows(), gradient.cols());

        for (int i = 0; i < _softmaxOutput.rows(); ++i) {
            Eigen::VectorXd y = _softmaxOutput.row(i).transpose();
            Eigen::VectorXd dy = gradient.row(i).transpose();
            Eigen::MatrixXd Jacob = _computeJacobian(y);
            dSoftmax_dx.row(i) = (Jacob * dy).transpose();
        }
        return dSoftmax_dx;
    } else {
        throw std::invalid_argument("[_computeSoftmaxGradient]: Unsupported input type.");
    }
}

template<typename T>
Eigen::MatrixXd Softmax<T>::_computeJacobian(Eigen::Ref<const Eigen::VectorXd> y) const {
    return y.asDiagonal().toDenseMatrix() - y * y.transpose();
}
