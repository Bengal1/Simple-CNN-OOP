#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>


class Activation {
public:
    virtual Eigen::VectorXd Activate(const Eigen::VectorXd& preActivationOutput) const = 0;
    virtual Eigen::MatrixXd Activate(const Eigen::MatrixXd& preActivationOutput) const = 0;
    virtual Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient, const Eigen::VectorXd& layerOutput) const = 0;
    virtual Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient, const Eigen::MatrixXd& layerOutput) const = 0; 
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
        preActivationOutput) const
    {  //Overload for 3D tensors
        size_t numChannels = preActivationOutput.size();
        std::vector<Eigen::MatrixXd> activationResult(numChannels,
            Eigen::MatrixXd::Zero(preActivationOutput[0].rows(),
                                  preActivationOutput[0].cols()));

        for (size_t c = 0; c < numChannels; ++c) {

            activationResult[c] = preActivationOutput[c].cwiseMax(0.0);
        }

        return activationResult;
    }


    Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient,
        const Eigen::VectorXd& layerOutput) const override 
    {   
        //dReLU_dz = 0 if neg; 1 if pos
        Eigen::VectorXd dRelu_dZ = (layerOutput.array() > 0).cast<double>(); 

        Eigen::VectorXd reluGradient = dRelu_dZ.cwiseProduct(lossGradient);

        return reluGradient;
    }

    Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient,
        const Eigen::MatrixXd& layerOutput) const override
    { 
        //dReLU_dz = 0 if neg; 1 if pos
        Eigen::MatrixXd dRelu_dZ = (layerOutput.array() > 0).cast<double>(); 

        Eigen::MatrixXd reluGradient = dRelu_dZ.cwiseProduct(lossGradient);

        return reluGradient;
    }

    std::vector<Eigen::MatrixXd> computeGradient(const std::vector<Eigen::MatrixXd>&
        lossGradient, const std::vector<Eigen::MatrixXd>& layerOutput) const
    { //Overload for 3D tensors
        size_t numChannels = layerOutput.size();
        std::vector<Eigen::MatrixXd> reluGradient(numChannels);
        //dReLU_dz = 0 if neg; 1 if pos
        for (size_t c = 0; c < numChannels; ++c) {
            Eigen::MatrixXd dRelu_dZ = (layerOutput[c].array() > 0).cast<double>();
            reluGradient[c] = dRelu_dZ.cwiseProduct(lossGradient[c]);
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
        Eigen::VectorXd expPreActivationOutput = preActivationOutput.array().exp();
        
        Eigen::VectorXd activationResult = expPreActivationOutput.array() / expPreActivationOutput.sum();

        return activationResult;
    }

    Eigen::MatrixXd Activate(const Eigen::MatrixXd& preActivationOutput) 
        const override
    {
        Eigen::MatrixXd expPreActivationOutput = preActivationOutput.array().exp();
        
        Eigen::MatrixXd activationResult = expPreActivationOutput.array() / expPreActivationOutput.sum();

        return activationResult;
    }

    Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient,
        const Eigen::VectorXd& layerOutput) const override 
    {
        Eigen::VectorXd dSoftmax_dZ = (layerOutput.array() * (1.0 - layerOutput.array()));

        Eigen::VectorXd softmaxGradient = dSoftmax_dZ.cwiseProduct(lossGradient);

        return softmaxGradient;
    }

    Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient,
        const Eigen::MatrixXd& layerOutput) const override
    {   
        Eigen::MatrixXd dSoftmax_dZ = (layerOutput.array() * (1.0 - layerOutput.array()));
        
        Eigen::MatrixXd softmaxGradient = dSoftmax_dZ.cwiseProduct(lossGradient);

        return softmaxGradient;
    }

};

#endif // ACTIVATION_HPP