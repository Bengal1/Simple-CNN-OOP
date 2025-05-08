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
	virtual ~Activation() {}

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
		_reluInput = preActivationOutput;
		// ReLU activation function: f(x) = max(0, x)
		return _applyReLU(preActivationOutput);
	}

	T computeGradient(const T& lossGradient) override {
		//dReLU_dz = 0 if neg; 1 if pos
		return _computeReLUGradient(lossGradient);
	}

private:
	void _initialize(const T& input) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			// For VectorXd
			_reluInput = Eigen::VectorXd::Zero(input.size());
			_reluGradient = Eigen::VectorXd::Zero(input.size());
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			// For MatrixXd
			_reluInput = Eigen::MatrixXd::Zero(input.rows(), input.cols());
			_reluGradient = Eigen::MatrixXd::Zero(input.rows(), input.cols());
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			// For std::vector<Eigen::MatrixXd>
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
	T _softmaxGradient;
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
		_softmaxGradient = _computeSoftmaxGradient(lossGradient);
		return _softmaxGradient;
	}
private:
	void _initialize(const T& input) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			// For VectorXd
			_softmaxOutput = Eigen::VectorXd::Zero(input.size());
			_softmaxGradient = Eigen::VectorXd::Zero(input.size());
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			// For MatrixXd
			_softmaxOutput = Eigen::MatrixXd::Zero(input.rows(), input.cols());
			_softmaxGradient = Eigen::MatrixXd::Zero(input.rows(), input.cols());
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			// For std::vector<Eigen::MatrixXd>
			_softmaxOutput.assign(input.size(), Eigen::MatrixXd::Zero(
				input[0].rows(), input[0].cols()));
			_softmaxGradient.assign(input.size(), Eigen::MatrixXd::Zero(
				input[0].rows(), input[0].cols()));
		}
		_initialized = true;
	}

	T _applySoftmax(const T& input) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			if (input.size() != _softmaxOutput.size()) {
				throw std::invalid_argument("[Softmax]: Input size does not match.");
			}
			Eigen::VectorXd expPreActivationOutput = input.array().exp();
			return expPreActivationOutput.array() /
				expPreActivationOutput.sum();
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			if (input.rows() != _softmaxOutput.rows() || input.cols() != _softmaxOutput.cols()) {
				throw std::invalid_argument("[Softmax]: Input dimensions do not match.");
			}
			Eigen::MatrixXd expPreActivationOutput = input.array().exp();
			return expPreActivationOutput.array() /
				expPreActivationOutput.sum();
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			if (input.size() != _softmaxOutput.size()) {
				throw std::invalid_argument("[Softmax]: Input channels do not match.");
			}
			_softmaxOutput = input;
			for (auto& mat : _softmaxOutput) {
				mat = mat.array().exp();
				mat = mat.array() / mat.sum();
			}
			return _softmaxOutput;
		}
		else {
			throw std::invalid_argument("[Softmax]: Unsupported input type.");
		}
	}
	T _computeSoftmaxGradient(const T& gradient) {
		if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
			if (gradient.size() != _softmaxOutput.size()) {
				throw std::invalid_argument("[Softmax]: Loss gradient and layer output sizes do not match.");
			}
			_softmaxGradient.setZero();
			_softmaxGradient = (_softmaxOutput.array() * (1.0 - _softmaxOutput.array())
				).matrix().cwiseProduct(gradient);
			return _softmaxGradient;
		}
		else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
			if (gradient.rows() != _softmaxOutput.rows() ||
				gradient.cols() != _softmaxOutput.cols()) {
				throw std::invalid_argument("[Softmax]: Loss gradient and layer output sizes do not match.");
			}
			_softmaxGradient.setZero();
			_softmaxGradient = (_softmaxOutput.array() * (1.0 - _softmaxOutput.array())
				).matrix().cwiseProduct(gradient);
			return _softmaxGradient;
		}
		else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>>) {
			if (gradient.size() != _softmaxOutput.size()) {
				throw std::invalid_argument("[Softmax]: Loss gradient and layer output sizes do not match.");
			}
			for (size_t c = 0; c < _softmaxGradient.size(); ++c) {
				_softmaxGradient[c].setZero();
				_softmaxGradient[c] = (_softmaxOutput[c].array() * (1.0 - _softmaxOutput[c]
					.array())).matrix().cwiseProduct(gradient[c]);
			}

			return _softmaxGradient;
		}
		else {
			throw std::invalid_argument("[Softmax]: Unsupported gradient type.");
		}
	}
    
};



/*Eigen::VectorXd Activate(const Eigen::VectorXd& preActivationOutput) override
	{
		// Check if the input is empty
		if (preActivationOutput.size() == 0) {
			throw std::invalid_argument("Input vector is empty.");
		}
		if (!_initialized) {
			_initialize(preActivationOutput);
		}
		_reluInputVec = preActivationOutput;
		// ReLU activation function: f(x) = max(0, x)
		Eigen::VectorXd activationResult = preActivationOutput.cwiseMax(0.0);

		return activationResult;
	}

	Eigen::MatrixXd Activate(const Eigen::MatrixXd& preActivationOutput) override
	{
		// Check if the input is empty
		if (preActivationOutput.rows() == 0 || preActivationOutput.cols() == 0) {
			throw std::invalid_argument("Input matrix is empty.");
		}
		if (!_initialized) {
			_initialize(preActivationOutput);
		}
		_reluInputMat = preActivationOutput;
		// ReLU activation function: f(x) = max(0, x)
		Eigen::MatrixXd activationResult = preActivationOutput.cwiseMax(0.0);

		return activationResult;
	}

	std::vector<Eigen::MatrixXd> Activate(const std::vector<Eigen::MatrixXd>&
										  preActivationOutput) override
	{
		// Check if the input is empty
		if (preActivationOutput.empty()) {
			throw std::invalid_argument("Input vector is empty.");
		}
		if (!_initialized) {
			_initialize(preActivationOutput);
		}

		_reluInput3D = preActivationOutput;
		// ReLU activation function: f(x) = max(0, x)
		std::vector<Eigen::MatrixXd> activationResult(_numChannels,
				Eigen::MatrixXd::Zero(preActivationOutput[0].rows(),
						preActivationOutput[0].cols()));

		for (size_t f = 0; f < _numChannels; ++f) {
			activationResult[f] = preActivationOutput[f].cwiseMax(0.0);
		}

		return activationResult;
	}


	Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient) override
	{
		if (lossGradient.size() != _reluInputVec.size()) {
			throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
		}
		_reluGradientVec.setZero();
		//dReLU_dz = 0 if neg; 1 if pos
		_reluGradientVec = ((_reluInputVec.array() > 0.0).select(lossGradient, 0.0)).matrix();

		return _reluGradientVec;
	}

	Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient) override
	{
		if (lossGradient.rows() != _reluInputMat.rows() || lossGradient.cols() != _reluInputMat.cols()) {
			throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
		}

		//dReLU_dz = 0 if neg; 1 if pos
		_reluGradientMat = ((_reluInputMat.array() > 0.0).select(lossGradient, 0.0)).matrix();

		return _reluGradientMat;
	}

	std::vector<Eigen::MatrixXd> computeGradient(
					const std::vector<Eigen::MatrixXd>& lossGradient) override
	{
		if (lossGradient.size() != _reluInput3D.size()) {
			throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
		}

		for (size_t f = 0; f < _numChannels; ++f) {
			_reluGradient3D[f].setZero();
			_reluGradient3D[f] = ((_reluInput3D[f].array() > 0.0).select(lossGradient[f],
				0.0)).matrix();
		}

		return _reluGradient3D;
	}
private:
	void _initialize(Eigen::VectorXd input) {
		_reluInputVec.setZero(input.size());
		_reluGradientVec.setZero(input.size());
		_numChannels = 1;
		_initialized = true;
	}
	void _initialize(Eigen::MatrixXd input) {
		_reluInputMat.setZero(input.rows(), input.cols());
		_reluGradientMat.setZero(input.rows(), input.cols());
		_numChannels = 1;
		_initialized = true;
	}
	void _initialize(std::vector<Eigen::MatrixXd> input) {
		_reluInput3D.assign(input.size(),
			Eigen::MatrixXd::Zero(input[0].rows(), input[0].cols()));
		_reluGradient3D.assign(input.size(),
			Eigen::MatrixXd::Zero(input[0].rows(), input[0].cols()));
		_numChannels = input.size();
		_initialized = true;
	}*/


	/*Eigen::VectorXd Activate(const Eigen::VectorXd& preActivationOutput)  override
		{
			// Check if the input is empty
			if (preActivationOutput.size() == 0) {
				throw std::invalid_argument("Input vector is empty.");
			}
			if (!_initialized) {
				_initialize(preActivationOutput);
			}

			Eigen::VectorXd expPreActivationOutput = preActivationOutput.array().exp();
			_softmaxOutputVec = expPreActivationOutput.array() /
								expPreActivationOutput.sum();

			return _softmaxOutputVec;
		}

		Eigen::MatrixXd Activate(const Eigen::MatrixXd& preActivationOutput) override
		{
			// Check if the input is empty
			if (preActivationOutput.rows() == 0 || preActivationOutput.cols() == 0) {
				throw std::invalid_argument("Input matrix is empty.");
			}
			if (!_initialized) {
				_initialize(preActivationOutput);
			}

			Eigen::MatrixXd expPreActivationOutput = preActivationOutput.array().exp();
			_softmaxOutputMat = expPreActivationOutput.array() /
								expPreActivationOutput.sum();

			return _softmaxOutputMat;
		}
		std::vector<Eigen::MatrixXd> Activate(const std::vector<Eigen::MatrixXd>& preActivationOutput) override
		{
			// Check if the input is empty
			if (preActivationOutput.empty()) {
				throw std::invalid_argument("Input vector is empty.");
			}
			if (!_initialized) {
				_initialize(preActivationOutput);
			}

			for (size_t c = 0; c < _numChannels; ++c) {
				_softmaxOutput3D[c].setZero();
				Eigen::MatrixXd expPreActivationOutput = preActivationOutput[c].array().exp();
				_softmaxOutput3D[c] = expPreActivationOutput.array() /
					expPreActivationOutput.sum();
			}

			return _softmaxOutput3D;
		}

		Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient) override
		{
			if (lossGradient.size() != _softmaxOutputVec.size()) {
				throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
			}

			_softmaxGradientVec.setZero();
			// dSoftmax_dx = softmax(x) * (1 - softmax(x))
			_softmaxGradientVec = (_softmaxOutputVec.array() * (1.0 - _softmaxOutputVec.array())
				).matrix().cwiseProduct(lossGradient);

			return _softmaxGradientVec;
		}

		Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient) override
		{
			if (lossGradient.rows() != _softmaxOutputMat.rows() ||
				lossGradient.cols() != _softmaxOutputMat.cols()) {
				throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
			}
			_softmaxGradientMat.setZero();
			// dSoftmax_dx = softmax(x) * (1 - softmax(x))
			_softmaxGradientMat = (_softmaxOutputMat.array() * (1.0 - _softmaxOutputMat.array())
				).matrix().cwiseProduct(lossGradient);

			return _softmaxGradientMat;
		}

		std::vector<Eigen::MatrixXd> computeGradient(
							const std::vector<Eigen::MatrixXd>& lossGradient) override
		{
			if (lossGradient.size() != _softmaxOutput3D.size()) {
				throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
			}

			for (size_t c = 0; c < _numChannels; ++c) {
				_softmaxGradient3D[c].setZero();
				Eigen::MatrixXd softmaxGrad = (_softmaxOutput3D[c].array() * (1.0 - _softmaxOutput3D[c].array())).matrix();
				softmaxGrad = softmaxGrad.cwiseProduct(lossGradient[c]);
				_softmaxGradient3D[c] = softmaxGrad;
			}
			return _softmaxGradient3D;
		}

	private:
		void _initialize(Eigen::VectorXd input) {
			_softmaxOutputVec.setZero(input.size());
			_softmaxGradientVec.setZero(input.size());
			_numChannels = 1;
			_initialized = true;
		}
		void _initialize(Eigen::MatrixXd input) {
			_softmaxOutputMat.setZero(input.rows(), input.cols());
			_softmaxGradientMat.setZero(input.rows(), input.cols());
			_numChannels = 1;
			_initialized = true;
		}
		void _initialize(std::vector<Eigen::MatrixXd> input) {
			_softmaxOutput3D.assign(input.size(),
				Eigen::MatrixXd::Zero(input[0].rows(), input[0].cols()));
			_softmaxGradient3D.assign(input.size(),
				Eigen::MatrixXd::Zero(input[0].rows(), input[0].cols()));
			_numChannels = input.size();
			_initialized = true;
		}*/