#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>


class Activation {
public:
    virtual Eigen::VectorXd Activate(const Eigen::VectorXd&
        preActivationOutput) const = 0;
    virtual Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient,
        const Eigen::VectorXd& layerOutput) const = 0;
    virtual ~Activation() {}
};

// ReLU activation function
class ReLU : public Activation {
public:

    Eigen::VectorXd Activate(const Eigen::VectorXd& preActivationOutput)
        const override {

        Eigen::VectorXd activationResult = preActivationOutput.cwiseMax(0.0);

        return activationResult;
    }

    Eigen::MatrixXd Activate(const Eigen::MatrixXd& preActivationOutput) const
    {  //Overload for matrices

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
        Eigen::VectorXd reluGradient = Eigen::VectorXd::Zero(layerOutput.size());
        //dReLU_dz = 0 if neg; 1 if pos
        reluGradient = ((layerOutput.array() > 0.0).select(lossGradient, 0.0)).matrix();

        return reluGradient;
    }

    Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient,
        const Eigen::MatrixXd& layerOutput) const 
    { //Overload for matrices
        Eigen::MatrixXd reluGradient = Eigen::MatrixXd::Zero(layerOutput.rows(),
            layerOutput.cols());
        //dReLU_dz = 0 if neg; 1 if pos
        reluGradient = ((layerOutput.array() > 0.0).select(lossGradient, 0.0)).matrix();

        return reluGradient;
    }

    std::vector<Eigen::MatrixXd> computeGradient(const std::vector<Eigen::MatrixXd>&
        lossGradient, const std::vector<Eigen::MatrixXd>& layerOutput) const
    { //Overload for 3D tensors
        size_t numChannels = layerOutput.size();
        std::vector<Eigen::MatrixXd> reluGradient(numChannels,
            Eigen::MatrixXd::Zero(layerOutput[0].rows(), layerOutput[0].cols()));

        for (size_t c = 0; c < numChannels; ++c) {
            reluGradient[c] = ((layerOutput[c].array() > 0.0).select(lossGradient[c],
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
    { // Overload for vectors
        Eigen::VectorXd activationResult = Eigen::VectorXd::Zero(
            preActivationOutput.size());

        Eigen::VectorXd expPreActivationOutput = preActivationOutput.array().exp();
        activationResult = expPreActivationOutput.array() /
            expPreActivationOutput.sum();

        return activationResult;
    }

    Eigen::MatrixXd Activate(const Eigen::MatrixXd& preActivationOutput) const
    {
        Eigen::MatrixXd activationResult = Eigen::MatrixXd::Zero(
            preActivationOutput.rows(), preActivationOutput.cols());

        Eigen::MatrixXd expPreActivationOutput = preActivationOutput.array().exp();
        activationResult = expPreActivationOutput.array() /
            expPreActivationOutput.sum();

        return activationResult;
    }

    std::vector<Eigen::VectorXd> Activate(const std::vector<Eigen::VectorXd>&
        preActivationOutput) const 
    {  //Overload for vector batch    
        size_t numChannels = preActivationOutput.size();
        std::vector<Eigen::VectorXd> activationResult(numChannels,
            Eigen::VectorXd::Zero(preActivationOutput[0].size()));


        for (size_t c = 0; c < numChannels; ++c) {
            Eigen::VectorXd expPreActivationOutput =
                preActivationOutput[c].array().exp();
            activationResult[c] = expPreActivationOutput.array() /
                expPreActivationOutput.sum();
        }
        return activationResult;
    }

    Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient,
        const Eigen::VectorXd& layerOutput) const override 
    {
        size_t numClasses = layerOutput.size();
        Eigen::VectorXd softmaxGradient(numClasses);
        softmaxGradient.setZero();
        // dSoftmax_dx = softmax(x) * (1 - softmax(x))
        softmaxGradient = (layerOutput.array() * (1.0 - layerOutput.array())
            ).matrix().cwiseProduct(lossGradient);

        return softmaxGradient;
    }

    Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient,
        const Eigen::MatrixXd& layerOutput) const 
    { //Overload for matrices    
        Eigen::MatrixXd softmaxGradient(layerOutput.rows(), layerOutput.cols());
        softmaxGradient.setZero();

        softmaxGradient = (layerOutput.array() * (1.0 - layerOutput.array())
            ).matrix().cwiseProduct(lossGradient);

        return softmaxGradient;
    }

    std::vector<Eigen::VectorXd> computeGradient(const std::vector<Eigen::VectorXd>&
        lossGradient, const std::vector<Eigen::VectorXd>& layerOutput) const 
    { //Overload for vector batch
        size_t numChannels = lossGradient.size();
        size_t numClasses = layerOutput.size();
        std::vector<Eigen::VectorXd> softmaxGradient(numChannels,
            Eigen::VectorXd::Zero(numClasses));

        for (size_t c = 0; c < numChannels; ++c) {
            softmaxGradient[c] = (layerOutput[c].array() * (1.0 - layerOutput[c].
                array())).matrix().cwiseProduct(lossGradient[c]);
        }
        return softmaxGradient;
    }
};

#endif // ACTIVATION_HPP