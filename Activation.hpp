#pragma once

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>


template<typename T>
class Activation {
public:
	virtual T Activate(const T& preActivationOutput) = 0;
	virtual T computeGradient(const T& lossGradient) = 0;
	virtual ~Activation() = default;

};

// ReLU activation function
template<typename T>
class ReLU : public Activation<T> {
private:
	T _reluInput;
	T _reluGradient;
	bool _initialized = false;

public:
	T Activate(const T& preActivationOutput) override {
		if (!_initialized) {
			_initialize(preActivationOutput);
		}
		// ReLU activation function: f(x) = max(0, x)
		_reluInput = preActivationOutput;
		return _applyReLU(preActivationOutput);
	}

	T computeGradient(const T& lossGradient) override {
		//dReLU_dz = 0 if neg; 1 if pos
		return _computeReLUGradient(lossGradient);
	}

private:
	void _initialize(const T& input) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			_reluInput = Eigen::VectorXd::Zero(input.size());
			_reluGradient = Eigen::VectorXd::Zero(input.size());
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			_reluInput = Eigen::MatrixXd::Zero(input.rows(), input.cols());
			_reluGradient = Eigen::MatrixXd::Zero(input.rows(), input.cols());
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			_reluInput.assign(input.size(), Eigen::MatrixXd::Zero(
				input[0].rows(), input[0].cols()));
			_reluGradient.assign(input.size(), Eigen::MatrixXd::Zero(
				input[0].rows(), input[0].cols()));
		}
		_initialized = true;

	}

	T _applyReLU(const T& input) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			if (input.size() != _reluInput.size()) {
				throw std::invalid_argument("[ReLU]: Input size does not match.");
			}
			return input.cwiseMax(0.0);
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			if (input.rows() != _reluInput.rows() || input.cols() != _reluInput.cols()) {
				throw std::invalid_argument("[ReLU]: Input dimensions do not match.");
			}
			return input.cwiseMax(0.0);
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			if (input.size() != _reluInput.size()) {
				throw std::invalid_argument("[ReLU]: Input channels do not match.");
			}
			std::vector<Eigen::MatrixXd> output = input;
			for (auto& mat : output) {
				mat = mat.cwiseMax(0.0);
			}
			return output;
		}
		else {
			throw std::invalid_argument("[ReLU]: Unsupported input type.");
		}
	}

	T _computeReLUGradient(const T& gradient) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			if (gradient.size() != _reluInput.size()) {
				throw std::invalid_argument("[ReLU]: Loss gradient and layer output sizes do not match.");
			}
			return (_reluInput.array() > 0.0).select(gradient, 0.0);
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			if (gradient.rows() != _reluInput.rows() || gradient.cols() != _reluInput.cols()) {
				throw std::invalid_argument("[ReLU]: Loss gradient and layer output sizes do not match.");
			}
			return (_reluInput.array() > 0.0).select(gradient, 0.0);
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			if (gradient.size() != _reluInput.size()) {
				throw std::invalid_argument("[ReLU]: Loss gradient and layer output sizes do not match.");
			}
			_reluGradient = gradient;
			for (size_t c = 0; c < _reluGradient.size(); ++c) {
				_reluGradient[c].setZero();
				_reluGradient[c] = (_reluInput[c].array() > 0.0).select(gradient[c], 0.0);
			}
			return _reluGradient;
		}
		else {
			throw std::invalid_argument("[ReLU]: Unsupported gradient type.");
		}
	}
    
};

// Softmax activation function
template<typename T>
class Softmax : public Activation<T> {
private:
	T _softmaxOutput;
	bool _initialized = false;

public:
	T Activate(const T& preActivationOutput) override {
		if (!_initialized) {
			_initialize(preActivationOutput);
		}
		// Softmax activation function: f(x) = exp(x) / sum(exp(x))
		_softmaxOutput = _applySoftmax(preActivationOutput);
		return _softmaxOutput;
	}

	T computeGradient(const T& lossGradient) override {
		// dSoftmax_dx = softmax(x) * (1 - softmax(x))
		return _computeSoftmaxGradient(lossGradient);
	}
private:
	void _initialize(const T& input) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			_softmaxOutput = Eigen::VectorXd::Zero(input.size());
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			_softmaxOutput = Eigen::MatrixXd::Zero(input.rows(), input.cols());
		}

		_initialized = true;
	}

	T _applySoftmax(const T& input) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			if (input.size() != _softmaxOutput.size()) {
				throw std::invalid_argument("[Softmax]: Input size does not match.");
			}
			Eigen::VectorXd exp = input.array().exp();
			return exp.array() / exp.sum();
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			if (input.rows() != _softmaxOutput.rows() || input.cols() != _softmaxOutput.cols()) {
				throw std::invalid_argument("[Softmax]: Input dimensions do not match.");
			}
			Eigen::MatrixXd exp = input.array().exp();
			return exp.array().colwise() / exp.rowwise().sum().array();;
		}
		else {
			throw std::invalid_argument("[Softmax]: Unsupported input type.");
		}
	}

	T _computeSoftmaxGradient(const T& gradient) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			const Eigen::MatrixXd Jacob = _computeJacobian(_softmaxOutput);
			return Jacob.transpose() * gradient;
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			Eigen::MatrixXd dSoftmax_dx(gradient.rows(), gradient.cols());

			for (int i = 0; i < _softmaxOutput.rows(); ++i) {
				Eigen::VectorXd y = _softmaxOutput.row(i).transpose();
				Eigen::VectorXd dy = gradient.row(i).transpose();
				Eigen::MatrixXd Jacob = _computeJacobian(y);
				dSoftmax_dx.row(i) = (Jacob * dy).transpose();
			}
			return dSoftmax_dx;
		}
		else {
			throw std::invalid_argument("[_computeSoftmaxGradient]: Unsupported input type.");
		}
	}

	Eigen::MatrixXd _computeJacobian(Eigen::Ref<const Eigen::VectorXd> y) const {
		return y.asDiagonal().toDenseMatrix() - y * y.transpose();
	}
};
