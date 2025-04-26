#pragma once 

#include "Utils.h"
#include "Layers/Convolution2D.h"
#include "Layers/FullyConnected.h"
#include "Layers/MaxPooling.h"
#include "Activation.h"
#include "MNISTLoader.h"
#include "LossFunction.h"
#include "Regularization.h"

/*TODO:
* look at the regularization
* look at the loss function
* look al al layers again
* new functions: save model, load model, save loss to csv
*/
class SimpleCNN {
private:
    //Layers
    Convolution2D _conv1;
    Convolution2D _conv2;
    FullyConnected _fc1;
    FullyConnected _fc2;
    MaxPooling _pool1;
    MaxPooling _pool2;
    //Regularization
    Dropout _dropout1;
    Dropout _dropout2;


public:
    CrossEntropy CEloss;

public:
    SimpleCNN() :
        _conv1(28, 28, 1, 32, 5, std::make_unique<ReLU>()),
        _pool1(24, 24, 32, 2),
        _conv2(12, 12, 32, 64, 5, std::make_unique<ReLU>()),
        _pool2(8, 8, 64, 2),
        _fc1(4 * 4 * 64, 512, std::make_unique<ReLU>()),
        _fc2(512, 10, std::make_unique<Softmax>()), 
        _dropout1(0.45), _dropout2(0.35)
    {}

    Eigen::VectorXd ForwardPass(Eigen::MatrixXd& input) {
        /*Forward propagation*/
        std::vector<Eigen::MatrixXd> outputConv1 = _conv1.forward(input);
        std::vector<Eigen::MatrixXd> outputPool1 = _pool1.forward(outputConv1);
        std::vector<Eigen::MatrixXd> outputDrop1 = _dropout1.forward(outputPool1);
        
        std::vector<Eigen::MatrixXd> outputConv2 = _conv2.forward(outputDrop1);
        std::vector<Eigen::MatrixXd> outputPool2 = _pool2.forward(outputConv2);
        std::vector<Eigen::MatrixXd> outputDrop2 = _dropout2.forward(outputPool2);
        
        Eigen::VectorXd outputFc1 = _fc1.forward(outputDrop2);
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
        /*_fc2.updateParameters();
        _fc1.updateParameters();
        _conv2.updateParameters();
        _conv1.updateParameters();*/

    }

  //  std::vector<Eigen::VectorXd> ForwardPassBatch(std::vector<Eigen::MatrixXd>& inputBatch) {
        /*Forward propagation*/
   /*     std::vector<std::vector<Eigen::MatrixXd>> outputConv1 =
            _conv1.forwardBatch(inputBatch);
        std::vector<std::vector<Eigen::MatrixXd>> outputPool1 = 
            _pool1.forwardBatch(outputConv1);
        std::vector<std::vector<Eigen::MatrixXd>> outputConv2 = 
            _conv2.forwardBatch(outputPool1);
        std::vector<std::vector<Eigen::MatrixXd>> outputPool2 = 
            _pool2.forwardBatch(outputConv2);
        std::vector<Eigen::VectorXd> outputFc1 = _fc1.forwardBatch(outputPool2);
        std::vector<Eigen::VectorXd> outputFc2 = _fc2.forwardBatch(outputFc1);

        return outputFc2;
    }

    void BackpropagationBatch(std::vector<Eigen::VectorXd>& lossGradientBatch) {
        /*Backward*/
   /*     std::vector<Eigen::VectorXd> fc2BackGrad =
            _fc2.backwardBatch(lossGradientBatch);
        std::vector<std::vector<Eigen::MatrixXd>> fc1BackGrad = 
            _fc1.backwardBatch(fc2BackGrad, true);
        std::vector<std::vector<Eigen::MatrixXd>> pool2BackGrad = 
            _pool2.backwardBatch(fc1BackGrad);
        std::vector<std::vector<Eigen::MatrixXd>> conv2BackGrad = 
            _conv2.backwardBatch(pool2BackGrad);
        std::vector<std::vector<Eigen::MatrixXd>> pool1BackGrad = 
            _pool1.backwardBatch(conv2BackGrad);
        _conv1.backwardBatch(pool1BackGrad);

        /*Optimizer - upadte step*/
   /*     _fc2.updateParameters();
        _fc1.updateParameters();
        _conv2.updateBatch();
        _conv1.updateBatch();
    }  */
};


void trainSimpleCNN(MNISTLoader& dataLoader, SimpleCNN& model, int epochs = 20)
{
    int classes = 10;
    /* Load MNIST Train dataset */
    const std::vector<Eigen::MatrixXd>& trainImages =
                                dataLoader.getTrainImages();
    const std::vector<Eigen::VectorXd>& oneHotTrainLabels =
                            dataLoader.getOneHotTrainLabels();
    int numTrainImages = dataLoader.getNumTrain();

    std::vector<Eigen::VectorXd> trainOutput(numTrainImages,
                                             Eigen::VectorXd(classes));

    /* Load MNIST Validation dataset */
    const std::vector<Eigen::MatrixXd>& validationImages =
        dataLoader.getValidationImages();
    const std::vector<Eigen::VectorXd>& oneHotValidationLabels =
        dataLoader.getOneHotValidationLabels();
    int numValidationImages = dataLoader.getNumValidation();

    std::vector<Eigen::VectorXd> validationOutput(numValidationImages,
                                             Eigen::VectorXd(classes));

    std::cout << "\nStart training..." << std::endl;

    /*double trainLoss = 0.0;*/
    std::vector<double> trainLoss(epochs);
    std::vector<double> trainAccuracy(epochs);
    std::vector<double> validationLoss(epochs);
    std::vector<double> validationAccuracy(epochs);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        double accuracy = 0.0;
        //Eigen::VectorXd outputEpoch(classes);
        std::cout << "\nepoch #" << (epoch) << std::endl; // Debug!

        int imageNum = 0;
        for (Eigen::MatrixXd image : trainImages) {
            /*Forward pass*/
            Eigen::VectorXd singleTrainOutput = model.ForwardPass(image);
            trainOutput[imageNum] = singleTrainOutput;
            /*Loss*/
            trainLoss[epoch] += model.CEloss.calculateLoss(singleTrainOutput,
                                            oneHotTrainLabels[imageNum]);
            Eigen::VectorXd lossGrad = model.CEloss.calculateGradient(
                            singleTrainOutput, oneHotTrainLabels[imageNum]);
            /*Backpropagation*/
            model.Backpropagation(lossGrad);
            
            /*Debug!*/
            if (imageNum % 1000 == 0) {
                std::cout << imageNum << ": " << std::endl;
                std::cout << singleTrainOutput << std::endl << std::endl;
                if (isnan(singleTrainOutput[0])) {
                    std::cout << "\nimage No. : " << imageNum << std::endl;
                    exit(-1);
                }
            }
            if (imageNum % 10000 == 0 and imageNum != 0) {
                std::vector<Eigen::VectorXd> tempTrainO(&trainOutput[0], &trainOutput[imageNum]);
                std::vector<Eigen::VectorXd> tempTrainL(&oneHotTrainLabels[0], &oneHotTrainLabels[imageNum]);
                std::cout << "Train Accuracy: " << accuracyCalculation(tempTrainO, tempTrainL) << "%\n" << std::endl;
            }
            /*Debug!*/

            imageNum++;
        }
        int validationImageNum = 0;
		//double valLoss = 0.0;
        for (Eigen::MatrixXd image : validationImages) {
            Eigen::VectorXd singleValidationOutput = model.ForwardPass(image);
            validationOutput[validationImageNum] = singleValidationOutput;
            validationLoss[epoch] += model.CEloss.calculateLoss(singleValidationOutput,
                                            oneHotValidationLabels[validationImageNum]);
            validationImageNum++;
        }
        validationAccuracy[epoch] = accuracyCalculation(validationOutput, oneHotValidationLabels);
        trainAccuracy[epoch] = accuracyCalculation(trainOutput,oneHotTrainLabels);
        std::cout << "Epoch" << epoch << ": Train Loss: " << trainLoss[epoch] << " ; Train Accuracy: " << trainAccuracy[epoch] << "% "
            << "| Validation Loss: " << validationLoss[epoch] << " ; Validation Accuracy: " << validationAccuracy[epoch] << "% " << std::endl;
    }
}

void testSimpleCNN(MNISTLoader& dataLoader, SimpleCNN& model)
{
    int classes = 10;
    /* Load MNIST Test dataset */
    const std::vector<Eigen::MatrixXd>& testImages = dataLoader.getTestImages();
    const std::vector<Eigen::VectorXd>& oneHotTestLabels =
        dataLoader.getOneHotTestLabels();
    int numTestImages = dataLoader.getNumTest();

    std::vector<Eigen::VectorXd> testOutput(numTestImages,
                                Eigen::VectorXd(classes));

    std::cout << "\nStart testing...\n" << std::endl;

    double testAccuracy = 0.0;
    int imageNum = 0;
    for (Eigen::MatrixXd image : testImages) {
        Eigen::VectorXd singleTestOutput = model.ForwardPass(image);
        testOutput[imageNum] = singleTestOutput;

        imageNum++;
    }
    testAccuracy = accuracyCalculation(testOutput, oneHotTestLabels);
    std::cout << "Test Accuracy: " << testAccuracy << "%\n" << std::endl;
}

int main()
{
    int epochs = 10, classes = 10;
	double validationRatio = 0.2;

    SimpleCNN model;

    /* Load MNIST dataset */
    MNISTLoader loader("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
                       "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
                        true, validationRatio);
    if (!loader.loadTrainData() || !loader.loadTestData()) {
        std::cerr << "Error: Loading data failed." << std::endl;
		return -1;
    }

    trainSimpleCNN(loader, model, epochs);

    testSimpleCNN(loader, model);

    return 0;
}




