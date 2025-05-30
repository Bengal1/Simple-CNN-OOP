#include <stdexcept>

template<typename T>
T ReLU<T>::Activate(const T& preActivationOutput) {
    if (!_initialized) {
        _initialize(preActivationOutput);
    }
    _reluInput = preActivationOutput;
    return _applyReLU(preActivationOutput);
}

template<typename T>
T ReLU<T>::computeGradient(const T& lossGradient) {
    return _computeReLUGradient(lossGradient);
}

template<typename T>
void ReLU<T>::_initialize(const T& input) {
    if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
        _reluInput = Eigen::VectorXd::Zero(input.size());
        _reluGradient = Eigen::VectorXd::Zero(input.size());
    } else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
        _reluInput = Eigen::MatrixXd::Zero(input.rows(), input.cols());
        _reluGradient = Eigen::MatrixXd::Zero(input.rows(), input.cols());
    } else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
        _reluInput.assign(input.size(), Eigen::MatrixXd::Zero(input[0].rows(), input[0].cols()));
        _reluGradient.assign(input.size(), Eigen::MatrixXd::Zero(input[0].rows(), input[0].cols()));
    }
    _initialized = true;
}

template<typename T>
T ReLU<T>::_applyReLU(const T& input) {
    if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
        if (input.size() != _reluInput.size()) {
            throw std::invalid_argument("[ReLU]: Input size does not match.");
        }
        return input.cwiseMax(0.0);
    } else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
        if (input.rows() != _reluInput.rows() || input.cols() != _reluInput.cols()) {
            throw std::invalid_argument("[ReLU]: Input dimensions do not match.");
        }
        return input.cwiseMax(0.0);
    } else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
        if (input.size() != _reluInput.size()) {
            throw std::invalid_argument("[ReLU]: Input channels do not match.");
        }
        std::vector<Eigen::MatrixXd> output = input;
        for (auto& mat : output) {
            mat = mat.cwiseMax(0.0);
        }
        return output;
    } else {
        throw std::invalid_argument("[ReLU]: Unsupported input type.");
    }
}

template<typename T>
T ReLU<T>::_computeReLUGradient(const T& gradient) {
    if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
        if (gradient.size() != _reluInput.size()) {
            throw std::invalid_argument("[ReLU]: Loss gradient and layer output sizes do not match.");
        }
        return (_reluInput.array() > 0.0).select(gradient, 0.0);
    } else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
        if (gradient.rows() != _reluInput.rows() || gradient.cols() != _reluInput.cols()) {
            throw std::invalid_argument("[ReLU]: Loss gradient and layer output sizes do not match.");
        }
        return (_reluInput.array() > 0.0).select(gradient, 0.0);
    } else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
        if (gradient.size() != _reluInput.size()) {
            throw std::invalid_argument("[ReLU]: Loss gradient and layer output sizes do not match.");
        }
        _reluGradient = gradient;
        for (size_t c = 0; c < _reluGradient.size(); ++c) {
            _reluGradient[c].setZero();
            _reluGradient[c] = (_reluInput[c].array() > 0.0).select(gradient[c], 0.0);
        }
        return _reluGradient;
    } else {
        throw std::invalid_argument("[ReLU]: Unsupported gradient type.");
    }
}
