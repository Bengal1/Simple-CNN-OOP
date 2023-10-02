#pragma once 

#include "Utils.h"
#include "Layers/Convolution2D.h"
#include "Layers/FullyConnected.h"
#include "Layers/MaxPooling.h"
#include "Activation.h"
#include "MNISTLoader.h"
#include "LossFunction.h"


class SimpleCNN {
private:
    Convolution2D _conv1;
    Convolution2D _conv2;
    FullyConnected _fc1;
    FullyConnected _fc2;
    MaxPooling _pool1;
    MaxPooling _pool2;

public:
    CrossEntropy CEloss;

public:
    SimpleCNN() :
        _conv1(28, 28, 1, 32, 5),
        _pool1(24, 24, 32, 2),
        _conv2(12, 12, 32, 64, 5),
        _pool2(8, 8, 64, 2),
        _fc1(4 * 4 * 64, 512, std::make_unique<ReLU>()),
        _fc2(512, 10, std::make_unique<Softmax>())
    {}

    Eigen::VectorXd ForwardPass(Eigen::MatrixXd& input) {
        /*Forward propagation*/
        std::vector<Eigen::MatrixXd> outputConv1 = _conv1.forward(input);
        std::vector<Eigen::MatrixXd> outputPool1 = _pool1.forward(outputConv1);
        std::vector<Eigen::MatrixXd> outputConv2 = _conv2.forward(outputPool1);
        std::vector<Eigen::MatrixXd> outputPool2 = _pool2.forward(outputConv2);
        Eigen::VectorXd outputFc1 = _fc1.forward(outputPool2);
        Eigen::VectorXd outputFc2 = _fc2.forward(outputFc1);

        return outputFc2;
    }

    void Backpropagation(Eigen::VectorXd& lossGradient) {
        /*Backward*/
        Eigen::VectorXd fc2BackGrad = _fc2.backward(lossGradient);
        std::vector<Eigen::MatrixXd> fc1BackGrad = _fc1.backward(fc2BackGrad, true);
        std::vector<Eigen::MatrixXd> pool2BackGrad = _pool2.backward(fc1BackGrad);
        std::vector<Eigen::MatrixXd> conv2BackGrad = _conv2.backward(pool2BackGrad);
        std::vector<Eigen::MatrixXd> pool1BackGrad = _pool1.backward(conv2BackGrad);
        _conv1.backward(pool1BackGrad);
        /*Optimizer - upadte step*/
        _fc2.updateParameters();
        _fc1.updateParameters();
        _conv2.updateParameters();
        _conv1.updateParameters();
    }

    std::vector<Eigen::VectorXd> ForwardPassBatch(std::vector<Eigen::MatrixXd>& inputBatch) {
        /*Forward propagation*/
        std::vector<std::vector<Eigen::MatrixXd>> outputConv1 = _conv1.forwardBatch(inputBatch);
        std::vector<std::vector<Eigen::MatrixXd>> outputPool1 = _pool1.forwardBatch(outputConv1);
        std::vector<std::vector<Eigen::MatrixXd>> outputConv2 = _conv2.forwardBatch(outputPool1);
        std::vector<std::vector<Eigen::MatrixXd>> outputPool2 = _pool2.forwardBatch(outputConv2);
        std::vector<Eigen::VectorXd> outputFc1 = _fc1.forwardBatch(outputPool2);
        std::vector<Eigen::VectorXd> outputFc2 = _fc2.forwardBatch(outputFc1);

        return outputFc2;
    }

    void BackpropagationBatch(std::vector<Eigen::VectorXd>& lossGradientBatch) {
        /*Backward*/
        std::vector<Eigen::VectorXd> fc2BackGrad = _fc2.backwardBatch(lossGradientBatch);
        std::vector<std::vector<Eigen::MatrixXd>> fc1BackGrad = _fc1.backwardBatch(fc2BackGrad, true);
        std::vector<std::vector<Eigen::MatrixXd>> pool2BackGrad = _pool2.backwardBatch(fc1BackGrad);
        std::vector<std::vector<Eigen::MatrixXd>> conv2BackGrad = _conv2.backwardBatch(pool2BackGrad);
        std::vector<std::vector<Eigen::MatrixXd>> pool1BackGrad = _pool1.backwardBatch(conv2BackGrad);
        _conv1.backwardBatch(pool1BackGrad);

        /*Optimizer - upadte step*/
        _fc2.updateParameters();
        _fc1.updateParameters();
        _conv2.updateBatch();
        _conv1.updateBatch();
    }
};

int main()
{
    int epochs = 10, classes = 10;

    SimpleCNN model;

    /* Load MNIST dataset */
    MNISTLoader loader("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
        "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
    if (!loader.loadTrainData() || !loader.loadTestData()) {
        std::cerr << "Error: Loading data failed." << std::endl;
        return -1;
    }
    const std::vector<Eigen::MatrixXd>& trainImages = loader.getTrainImages();
    const std::vector<Eigen::MatrixXd>& testImages = loader.getTestImages();
    const std::vector<Eigen::VectorXd>& oneHotTrainLabels = loader.getOneHotTrainLabels();
    const std::vector<Eigen::VectorXd>& oneHotTestLabels = loader.getOneHotTestLabels();
    int numTrainImages = loader.numTrain;
    int numTestImages = loader.numTest;

    std::vector<Eigen::VectorXd> trainOutput(numTrainImages, Eigen::VectorXd(classes));

    std::cout << "\nStart training..." << std::endl;

    /* Train */
    std::vector<double> epochLoss(epochs);
    std::vector<double> trainAccuracy(epochs);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double accuracy = 0.0;
        Eigen::VectorXd outputEpoch(classes);
        std::cout << "\nepoch #" << (epoch + 1) << std::endl;

        int imageNum = 0;
        for (Eigen::MatrixXd image : trainImages) {
            /*Forward pass*/
            Eigen::VectorXd singleTrainOutput = model.ForwardPass(image);
            trainOutput[imageNum] = singleTrainOutput;
            /*Loss*/
            epochLoss[epoch] += model.CEloss.calculateLoss(singleTrainOutput, oneHotTrainLabels[imageNum]);
            Eigen::VectorXd lossGrad = model.CEloss.calculateGradient(singleTrainOutput,
                oneHotTrainLabels[imageNum]);
            /*Backpropagation*/
            model.Backpropagation(lossGrad);

            imageNum++;

            //TESTING
            if (imageNum % 5000 == 0) {
                std::cout << "[" << imageNum << "/60000] " << "; Loss = " << epochLoss[epoch] << std::endl;
            }
            //TESTING
        }
        trainAccuracy[epoch] = accuracyCalculation(trainOutput, oneHotTrainLabels);
        std::cout << "Train Accuracy: " << trainAccuracy[epoch] << "%" << 
            " ; Epoch Loss: " << epochLoss[epoch] << std::endl;
    }

    std::vector<Eigen::VectorXd> testOutput(numTestImages, Eigen::VectorXd(classes));

    std::cout << "\nStart testing...\n" << std::endl;

    /* Test */
    double testAccuracy = 0.0;
    int imageNum = 0;
    for (Eigen::MatrixXd image : testImages) {
        Eigen::VectorXd singleTestOutput = model.ForwardPass(image);
        testOutput[imageNum] = singleTestOutput;

        imageNum++;
    }
    testAccuracy = accuracyCalculation(testOutput, oneHotTestLabels);
    std::cout << "Test Accuracy: " << testAccuracy << "%\n" << std::endl;

    return 0;
}





//void trainSimpleCNN(const std::vector<Eigen::MatrixXd>& trainImages, const std::vector<Eigen::VectorXd>& oneHotTrainLabels, int epochs = 10) {
//    int classes = 10;
//    std::vector<Eigen::VectorXd> trainOutput(trainImages.size(), Eigen::VectorXd(classes));
//
//    std::cout << "\nStart training..." << std::endl;
//
//    /* Train */
//    double totalLoss = 0.0;
//    std::vector<double> trainAccuracy(epochs);
//    for (int epoch = 0; epoch < epochs; epoch++) {
//        double accuracy = 0.0;
//        Eigen::VectorXd outputEpoch(classes);
//        std::cout << "\nepoch #" << (epoch + 1) << std::endl;
//
//        int imageNum = 0;
//        for (Eigen::MatrixXd image : trainImages) {
//            /*Forward pass*/
//            Eigen::VectorXd singleTrainOutput = model.ForwardPass(image);
//            trainOutput[imageNum] = singleTrainOutput;
//            /*Loss*/
//            totalLoss += model.CEloss.calculateLoss(singleTrainOutput, oneHotTrainLabels[imageNum]);
//            Eigen::VectorXd lossGrad = model.CEloss.calculateGradient(singleTrainOutput,
//                oneHotTrainLabels[imageNum]);
//            /*Backpropagation*/
//            model.Backpropagation(lossGrad);
//
//            imageNum++;
//        }
//        trainAccuracy[epoch] = accuracyCalculation(trainOutput, oneHotTrainLabels);
//        std::cout << "Train Accuracy: " << trainAccuracy[epoch] << "%" << " ; Loss: " << totalLoss << std::endl;
//    }
//}

/*void testSimpleCNN(int epochs) {
    int epochs = 10, classes = 10, 

}*/

/*void trainModelBatch() {
    int epochs = 10, classes = 10, batchSize = 256;
    std::iterator startbatch, endBatch;

}*/