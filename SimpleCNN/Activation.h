#pragma once

#include <vector>
#include <Eigen/Dense>


class Activation {
public:
    virtual Eigen::VectorXd activate(const Eigen::VectorXd& preActivationOutput) const = 0;
    virtual Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient, const Eigen::VectorXd& preActivationOutput) const = 0;
    virtual ~Activation() {}
};

// ReLU activation function
class ReLU : public Activation {
public:

    Eigen::VectorXd activate(const Eigen::VectorXd& preActivationOutput) const override {
        Eigen::VectorXd activationResult = Eigen::VectorXd::Zero(preActivationOutput.size());

        activationResult.cwiseMax(0.0);

        return activationResult;
    }

    Eigen::MatrixXd activate(const Eigen::MatrixXd& preActivationOutput) const {  //Overload for matrices
        Eigen::MatrixXd activationResult = Eigen::MatrixXd::Zero(preActivationOutput.rows(), preActivationOutput.cols());

        activationResult.cwiseMax(0.0);

        return activationResult;
    }

    std::vector<Eigen::MatrixXd> activate(const std::vector<Eigen::MatrixXd>& preActivationOutput) const {  //Overload for 3D tensors
        std::vector<Eigen::MatrixXd> activationResult(preActivationOutput.size(),
            Eigen::MatrixXd::Zero(preActivationOutput[0].rows(), preActivationOutput[0].cols()));

        for (int f = 0; f < preActivationOutput.size(); ++f) {
            activationResult[f].cwiseMax(0.0);
        }
        return activationResult;
    }


    Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient, const Eigen::VectorXd& preActivationOutput) const override {
        Eigen::VectorXd reluGrad = preActivationOutput;

        reluGrad.unaryExpr([](double x) { return x > 0 ? 1.0 : 0.0; }).cwiseProduct(lossGradient);

        return reluGrad;
    }

    Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient, const Eigen::MatrixXd& preActivationOutput) const { //Overload for matrices
        Eigen::MatrixXd reluGrad = preActivationOutput;

        reluGrad.unaryExpr([](double x) { return x > 0 ? 1.0 : 0.0; }).cwiseProduct(lossGradient);

        return reluGrad;
    }

    std::vector<Eigen::MatrixXd> computeGradient(const std::vector<Eigen::MatrixXd>& lossGradient,  //Overload for 3D tensors 
        const std::vector<Eigen::MatrixXd>& preActivationOutput) const {
        std::vector<Eigen::MatrixXd> reluGrad = preActivationOutput;

        for (int f = 0; f <= preActivationOutput.size(); ++f) {
            reluGrad[f].unaryExpr([](double x) { return x > 0 ? 1.0 : 0.0; }).cwiseProduct(lossGradient[f]);
        }
        return reluGrad;
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
        std::vector<Eigen::VectorXd> activationResult(preActivationOutput.size(), Eigen::VectorXd::Zero(preActivationOutput[0].size()));

        for (int f = 0; f < preActivationOutput.size(); ++f) {
            Eigen::MatrixXd exppreActivationOutput = preActivationOutput[f].array().exp();
            activationResult[f] = exppreActivationOutput.array() / exppreActivationOutput.sum();
        }
        return activationResult;
    }

    Eigen::VectorXd computeGradient(const Eigen::VectorXd& lossGradient, const Eigen::VectorXd& preActivationOutput) const override {
        Eigen::VectorXd softmaxGrad = activate(preActivationOutput);

        softmaxGrad.array() *= (lossGradient.array() - (softmaxGrad.array() * lossGradient.array()).sum());

        return softmaxGrad;
    }

    Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& lossGradient, const Eigen::MatrixXd& preActivationOutput) const { //Overload for matrices
        Eigen::MatrixXd softmaxGrad = activate(preActivationOutput);

        softmaxGrad.array() *= (lossGradient.array() - (softmaxGrad.array() * lossGradient.array()).sum());

        return softmaxGrad;
    }

    std::vector<Eigen::VectorXd> computeGradient(const std::vector<Eigen::VectorXd>& lossGradient, //Overload for vector batch
        const std::vector<Eigen::VectorXd>& preActivationOutput) const {
        int batchSize = lossGradient.size();
        std::vector<Eigen::VectorXd> softmaxGrad = activate(preActivationOutput);

        for (int f = 0; f < batchSize; f++) {
            softmaxGrad[f].array() *= (lossGradient[f].array() - (softmaxGrad[f].array() * lossGradient[f].array()).sum());
        }

        return softmaxGrad;
    }
};
