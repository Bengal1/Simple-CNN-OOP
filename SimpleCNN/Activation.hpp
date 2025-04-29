#pragma once

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>


class Activation {
public:
    virtual Eigen::VectorXd Activate(const Eigen::VectorXd& preActivationOutput) const = 0;
	virtual Eigen::MatrixXd Activate(const Eigen::MatrixXd& preActivationOutput) const = 0;
	virtual std::vector<Eigen::MatrixXd> Activate(
                        const std::vector<Eigen::MatrixXd>& preActivationOutput) const = 0;
    virtual Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient, 
                                            const Eigen::VectorXd& layerOutput) const = 0;
	virtual Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient, 
                                            const Eigen::MatrixXd& layerOutput) const = 0;
	virtual std::vector<Eigen::MatrixXd> computeGradient(
                                const std::vector<Eigen::MatrixXd>& lossGradient,
		                        const std::vector<Eigen::MatrixXd>& layerOutput) const = 0;
    virtual ~Activation() {}
};

// ReLU activation function
class ReLU : public Activation {
public:

    Eigen::VectorXd Activate(const Eigen::VectorXd& preActivationOutput) 
                             const override 
    {    
        Eigen::VectorXd activationResult = preActivationOutput.cwiseMax(0.0);
        
        return activationResult;
    }

    Eigen::MatrixXd Activate(const Eigen::MatrixXd& preActivationOutput) 
                             const override 
    {    
        Eigen::MatrixXd activationResult = preActivationOutput.cwiseMax(0.0);
        
        return activationResult;
    }

    std::vector<Eigen::MatrixXd> Activate(const std::vector<Eigen::MatrixXd>& 
                                          preActivationOutput) const override
    {
        size_t numChannels = preActivationOutput.size();
        std::vector<Eigen::MatrixXd> activationResult(numChannels,
            Eigen::MatrixXd::Zero(preActivationOutput[0].rows(), 
                preActivationOutput[0].cols()));

        for (size_t f = 0; f < numChannels; ++f) {
            activationResult[f] = preActivationOutput[f].cwiseMax(0.0);
        }

        return activationResult;
    }


    Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient, 
                                    const Eigen::VectorXd& layerOutput) const override 
    {
		if (lossGradient.size() != layerOutput.size()) {
			throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
		}

        Eigen::VectorXd reluGradient = Eigen::VectorXd::Zero(layerOutput.size());
        //dReLU_dz = 0 if neg; 1 if pos
        reluGradient = ((layerOutput.array() > 0.0).select(lossGradient, 0.0)).matrix();

        return reluGradient;
    }

    Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient, 
                                    const Eigen::MatrixXd& layerOutput) const override
    {
		if (lossGradient.rows() != layerOutput.rows() || lossGradient.cols() != layerOutput.cols()) {
			throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
		}

        Eigen::MatrixXd reluGradient = Eigen::MatrixXd::Zero(layerOutput.rows(), 
            layerOutput.cols());
        //dReLU_dz = 0 if neg; 1 if pos
        reluGradient = ((layerOutput.array() > 0.0).select(lossGradient, 0.0)).matrix();
       
        return reluGradient;
    }

    std::vector<Eigen::MatrixXd> computeGradient(
                    const std::vector<Eigen::MatrixXd>& lossGradient, 
                    const std::vector<Eigen::MatrixXd>& layerOutput) const override
    {
		if (lossGradient.size() != layerOutput.size()) {
			throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
		}
        size_t numChannels = layerOutput.size();
        std::vector<Eigen::MatrixXd> reluGradient(numChannels, 
            Eigen::MatrixXd::Zero(layerOutput[0].rows(), layerOutput[0].cols()));

        for (size_t f = 0; f < numChannels; ++f) {
            reluGradient[f] = ((layerOutput[f].array() > 0.0).select(lossGradient[f], 
                0.0)).matrix();
        }

        return reluGradient;
    }
};

// Softmax activation function
class Softmax : public Activation {
public:

    Eigen::VectorXd Activate(const Eigen::VectorXd& preActivationOutput) 
                             const override 
    {
        Eigen::VectorXd activationResult = Eigen::VectorXd::Zero(
            preActivationOutput.size());
        
        Eigen::VectorXd expPreActivationOutput = preActivationOutput.array().exp();
        activationResult = expPreActivationOutput.array() /
                expPreActivationOutput.sum();
        
        return activationResult;
    }

    Eigen::MatrixXd Activate(const Eigen::MatrixXd& preActivationOutput) 
                             const override
    {
        Eigen::MatrixXd activationResult = Eigen::MatrixXd::Zero(
            preActivationOutput.rows(), preActivationOutput.cols());

        Eigen::MatrixXd expPreActivationOutput = preActivationOutput.array().exp();
        activationResult = expPreActivationOutput.array() / 
            expPreActivationOutput.sum();

        return activationResult;
    }
    std::vector<Eigen::MatrixXd> Activate(const std::vector<Eigen::MatrixXd>& preActivationOutput) 
                                          const override 
    {
        size_t numSamples = preActivationOutput.size();
        std::vector<Eigen::MatrixXd> activationResult;
        activationResult.reserve(numSamples);

        for (const auto& vec : preActivationOutput) {
            Eigen::MatrixXd expVec = vec.array().exp();
            Eigen::MatrixXd softmaxVec = expVec.array() / expVec.sum();
            activationResult.push_back(softmaxVec);
        }
        return activationResult;
    }

    Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient, 
                                    const Eigen::VectorXd& preActivationOutput) const override 
    {   
		if (lossGradient.size() != preActivationOutput.size()) {
			throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
		}
		Eigen::VectorXd layerOutput = Activate(preActivationOutput);
        size_t numClasses = layerOutput.size();
        Eigen::VectorXd softmaxGradient(numClasses);
        softmaxGradient.setZero();
        // dSoftmax_dx = softmax(x) * (1 - softmax(x))
        softmaxGradient = (layerOutput.array() * (1.0 - layerOutput.array())
            ).matrix().cwiseProduct(lossGradient);

        return softmaxGradient;
    }

    Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient, 
                                    const Eigen::MatrixXd& preActivationOutput) const override
    { 
		if (lossGradient.rows() != preActivationOutput.rows() ||
			lossGradient.cols() != preActivationOutput.cols()) {
			throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
		}
		Eigen::MatrixXd layerOutput = Activate(preActivationOutput);
        Eigen::MatrixXd softmaxGradient(layerOutput.rows(), layerOutput.cols());
        softmaxGradient.setZero();

        softmaxGradient = (layerOutput.array() * (1.0 - layerOutput.array())
            ).matrix().cwiseProduct(lossGradient);

        return softmaxGradient;
    }
    
    std::vector<Eigen::MatrixXd> computeGradient(const std::vector<Eigen::MatrixXd>& lossGradient,
                                                 const std::vector<Eigen::MatrixXd>& preActivationOutput)
                                                 const override 
    {
		if (lossGradient.size() != preActivationOutput.size()) {
			throw std::invalid_argument("Loss gradient and layer output sizes do not match.");
		}
        
        size_t numSamples = lossGradient.size();
        std::vector<Eigen::MatrixXd> gradientResult;
        gradientResult.reserve(numSamples);

        for (size_t i = 0; i < numSamples; ++i) {
            Eigen::MatrixXd layerOutput = Activate(preActivationOutput[i]);
            Eigen::MatrixXd softmaxGrad = (layerOutput.array() * (1.0 - layerOutput.array())).matrix();
            softmaxGrad = softmaxGrad.cwiseProduct(lossGradient[i]);
            gradientResult.push_back(softmaxGrad);
        }
        return gradientResult;
    }
};
