#pragma once

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>


class Activation {
public:
    virtual Eigen::VectorXd activate(const Eigen::VectorXd& preActivationOutput) const = 0;
    virtual Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient, 
        const Eigen::VectorXd& layerOutput) const = 0;
    virtual ~Activation() {}
};

// ReLU activation function
class ReLU : public Activation {
public:

    Eigen::VectorXd activate(const Eigen::VectorXd& preActivationOutput) const override {
        
        Eigen::VectorXd activationResult = preActivationOutput.cwiseMax(0.0);
        
        return activationResult;
    }

    Eigen::MatrixXd activate(const Eigen::MatrixXd& preActivationOutput) const {  //Overload for matrices
        
        Eigen::MatrixXd activationResult = preActivationOutput.cwiseMax(0.0);
        
        return activationResult;
    }

    std::vector<Eigen::MatrixXd> activate(const std::vector<Eigen::MatrixXd>& preActivationOutput) const {  //Overload for 3D tensors
        int numChannels = preActivationOutput.size();
        std::vector<Eigen::MatrixXd> activationResult(numChannels,
            Eigen::MatrixXd::Zero(preActivationOutput[0].rows(), preActivationOutput[0].cols()));

        for (int f = 0; f < numChannels; f++) {
            activationResult[f] = preActivationOutput[f].cwiseMax(0.0);
        }

        return activationResult;
    }


    Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient, 
        const Eigen::VectorXd& layerOutput) const override {
        Eigen::VectorXd reluGradient = Eigen::VectorXd::Zero(layerOutput.size());

        reluGradient = (layerOutput.array() > 0.0).select(lossGradient, 0.0);

        return reluGradient;
    }

    Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient, const Eigen::MatrixXd& layerOutput) const { //Overload for matrices
        Eigen::MatrixXd reluGradient = Eigen::MatrixXd::Zero(layerOutput.rows(), layerOutput.cols());

        reluGradient = (layerOutput.array() > 0.0).select(lossGradient, 0.0);

        return reluGradient;
    }

    std::vector<Eigen::MatrixXd> computeGradient(const std::vector<Eigen::MatrixXd>& lossGradient,  //Overload for 3D tensors 
        const std::vector<Eigen::MatrixXd>& layerOutput) const {
        int numChannels = layerOutput.size();
        std::vector<Eigen::MatrixXd> reluGradient(numChannels, Eigen::MatrixXd::Zero(layerOutput[0].rows(), layerOutput[0].cols()));

        for (int f = 0; f < numChannels; f++) {
            reluGradient[f] = (layerOutput[f].array() > 0.0).select(lossGradient[f], 0.0);
        }

        return reluGradient;
    }
};

// Softmax activation function
class Softmax : public Activation {
public:

    Eigen::VectorXd activate(const Eigen::VectorXd& preActivationOutput) const override { //Overload for vectors
        Eigen::VectorXd activationResult = Eigen::VectorXd::Zero(preActivationOutput.size());

        Eigen::VectorXd exppreActivationOutput = preActivationOutput.array().exp();
        activationResult = exppreActivationOutput.array() / exppreActivationOutput.sum();
        
        return activationResult;
    }

    Eigen::MatrixXd activate(const Eigen::MatrixXd& preActivationOutput) const {
        Eigen::MatrixXd activationResult = Eigen::MatrixXd::Zero(preActivationOutput.rows(), preActivationOutput.cols());

        Eigen::MatrixXd exppreActivationOutput = preActivationOutput.array().exp();
        activationResult = exppreActivationOutput.array() / exppreActivationOutput.sum();

        return activationResult;
    }

    std::vector<Eigen::VectorXd> activate(const std::vector<Eigen::VectorXd>& preActivationOutput) const {  //Overload for vector batch
        int numChannels = preActivationOutput.size();
        std::vector<Eigen::VectorXd> activationResult(numChannels, Eigen::VectorXd::Zero(preActivationOutput[0].size()));


        for (int f = 0; f < numChannels; ++f) {
            Eigen::VectorXd exppreActivationOutput = preActivationOutput[f].array().exp();
            activationResult[f] = exppreActivationOutput.array() / exppreActivationOutput.sum();
        }
        return activationResult;
    }

    Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient, const Eigen::VectorXd& layerOutput) const override {
        size_t numClasses = layerOutput.size();
        Eigen::VectorXd softmaxGradient(numClasses);
        softmaxGradient.setZero();

        softmaxGradient = (layerOutput.array() * (1.0 - layerOutput.array())).matrix().cwiseProduct(lossGradient);
        
        return softmaxGradient;
    }

    Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient, const Eigen::MatrixXd& layerOutput) const { //Overload for matrices
        size_t rows = layerOutput.rows(), cols = layerOutput.cols();
        Eigen::MatrixXd softmaxGradient(rows, cols);
        softmaxGradient.setZero();

        softmaxGradient = (layerOutput.array() * (1.0 - layerOutput.array())).matrix().cwiseProduct(lossGradient);

        return softmaxGradient;
    }

    std::vector<Eigen::VectorXd> computeGradient(const std::vector<Eigen::VectorXd>& lossGradient, //Overload for vector batch
        const std::vector<Eigen::VectorXd>& layerOutput) const {
        int numChannels = lossGradient.size();
        size_t numClasses = layerOutput.size();
        std::vector<Eigen::VectorXd> softmaxGradient(numChannels, Eigen::VectorXd::Zero(numClasses));
        
        for (int f = 0; f < numChannels; f++) {
            softmaxGradient[f] = (layerOutput[f].array() * (1.0 - layerOutput[f].array())).matrix().
                cwiseProduct(lossGradient[f]);
        }
        return softmaxGradient;
    }
};
