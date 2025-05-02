#pragma once

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>


class Activation {
public:
    virtual Eigen::VectorXd Activate(const Eigen::VectorXd& preActivationOutput) = 0;
	virtual Eigen::MatrixXd Activate(const Eigen::MatrixXd& preActivationOutput) = 0;
	virtual std::vector<Eigen::MatrixXd> Activate(
                        const std::vector<Eigen::MatrixXd>& preActivationOutput) = 0;
    virtual Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient) = 0;
	virtual Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient) = 0;
	virtual std::vector<Eigen::MatrixXd> computeGradient(
                               const std::vector<Eigen::MatrixXd>& lossGradient) = 0;
    virtual ~Activation() {}
};

// ReLU activation function
class ReLU : public Activation {
private:
	bool _initialized;
	size_t _numChannels;
	// Pre-activation output
    Eigen::VectorXd _reluInputVec;
	Eigen::MatrixXd _reluInputMat;
	std::vector<Eigen::MatrixXd> _reluInput3D;
	// Gradient
	Eigen::VectorXd _reluGradientVec;
	Eigen::MatrixXd _reluGradientMat;
	std::vector<Eigen::MatrixXd> _reluGradient3D;

public:

    Eigen::VectorXd Activate(const Eigen::VectorXd& preActivationOutput) override 
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
	}
};

// Softmax activation function
class Softmax : public Activation {
private:
	bool _initialized;
	size_t _numChannels;
	// Post-activation output
    Eigen::VectorXd _softmaxOutputVec;
	Eigen::MatrixXd _softmaxOutputMat;
	std::vector<Eigen::MatrixXd> _softmaxOutput3D;
	// Gradient
	Eigen::VectorXd _softmaxGradientVec;
    Eigen::MatrixXd _softmaxGradientMat;
    std::vector<Eigen::MatrixXd> _softmaxGradient3D;

public:

    Eigen::VectorXd Activate(const Eigen::VectorXd& preActivationOutput)  override 
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
	}
};
