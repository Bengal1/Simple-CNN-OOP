#include "Layers/Convolution2D.hpp"
#include "Layers/FullyConnected.hpp"
#include "Layers/MaxPooling.hpp"
#include "Activation.hpp"
#include "MNISTLoader.hpp"
#include "LossFunction.hpp"
#include "Regularization.hpp"
#include <math.h>

/* TODO:
    * TrainMode/TestMode methods
    * code std.
*/


const size_t classes = 10;

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
    SimpleCNN() 
    : _conv1(28, 28, 1, 32, 5),
      _pool1(24, 24, 32, 2),
      _conv2(12, 12, 32, 64, 5),
      _pool2(8, 8, 64, 2),
      _fc1(4 * 4 * 64, 512, std::make_unique<ReLU>()),
      _fc2(512, 10, std::make_unique<Softmax>()),
      _dropout1(0.45), 
      _dropout2(0.35)
    {}

    Eigen::VectorXd ForwardPass(const Eigen::MatrixXd& input) {
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
    }
};

double accuracyCalculation(std::vector<Eigen::VectorXd>& modelOutput,
    const std::vector<Eigen::VectorXd>& oneHotTargets) {

    if (modelOutput.size() != oneHotTargets.size()) {
        throw std::invalid_argument("Input vectors have different sizes.");
    }

    double correctPredictions = 0;
    size_t dataSize = modelOutput.size();
    Eigen::Index predictedClass = 0, trueClass = 0;

    for (size_t d = 0; d < dataSize; ++d) {
        modelOutput[d].maxCoeff(&predictedClass);
        oneHotTargets[d].maxCoeff(&trueClass);
        if (predictedClass == trueClass){
            correctPredictions++;
        }
    }

    return correctPredictions / static_cast<double>(dataSize) * 100.0;
}

void trainSimpleCNN(MNISTLoader& dataLoader, SimpleCNN& model, size_t epochs = 10)
{
    /* Load MNIST Train dataset */
    const std::vector<Eigen::MatrixXd>& trainImages =
        dataLoader.getTrainImages();
    const std::vector<Eigen::VectorXd>& oneHotTrainLabels =
        dataLoader.getOneHotTrainLabels();

    std::vector<Eigen::VectorXd> trainOutput(dataLoader.numTrain,
        Eigen::VectorXd(classes));

    std::cout << "\nStart training..." << std::endl;

    std::vector<double> trainAccuracy(epochs);
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        Eigen::VectorXd outputEpoch(classes);
        std::cout << "\nepoch #" << (epoch + 1) << std::endl;
        
        size_t imageNum = 0;
        double totalEpochLoss = 0.0;
        for (const auto image : trainImages) {
            /*Forward pass*/
            Eigen::VectorXd singleTrainOutput = model.ForwardPass(image);
            trainOutput[imageNum] = singleTrainOutput;
            /*Loss*/
            totalEpochLoss += model.CEloss.calculateLoss(singleTrainOutput,
                oneHotTrainLabels[imageNum]);
            Eigen::VectorXd lossGrad = model.CEloss.calculateGradient(
                singleTrainOutput, oneHotTrainLabels[imageNum]);
            /*Backpropagation*/
            model.Backpropagation(lossGrad);

            /*TEST*/
            if (imageNum % 100 == 0) {
                std::cout << imageNum << ": " << std::endl;
                std::cout << singleTrainOutput << std::endl << std::endl;
                if (isnan(singleTrainOutput[0])) {
                    std::cout << "\nimage No. : " << imageNum << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            if (imageNum % 10000 == 0 and imageNum != 0) {
                std::vector<Eigen::VectorXd> tempTrainO(&trainOutput[0], &trainOutput[imageNum]);
                std::vector<Eigen::VectorXd> tempTrainL(&oneHotTrainLabels[0], &oneHotTrainLabels[imageNum]);
                std::cout << "Train Accuracy: " << accuracyCalculation(tempTrainO, tempTrainL) << "%\n" << std::endl;
            }
            /*TEST*/

            imageNum++;
        }
        trainAccuracy[epoch] = accuracyCalculation(trainOutput,
            oneHotTrainLabels);
        std::cout << "Train Accuracy: " << trainAccuracy[epoch] << "%"
            << " ; Loss: " << totalEpochLoss << std::endl;
    }
}

void testSimpleCNN(MNISTLoader& dataLoader, SimpleCNN& model)
{
    /* Load MNIST Test dataset */
    const std::vector<Eigen::MatrixXd>& testImages = 
        dataLoader.getTestImages();
    const std::vector<Eigen::VectorXd>& oneHotTestLabels =
        dataLoader.getOneHotTestLabels();

    std::vector<Eigen::VectorXd> testOutput(dataLoader.numTest,
        Eigen::VectorXd(classes));

    std::cout << "\nStart testing...\n" << std::endl;

    double testAccuracy = 0.0;
    size_t imageNum = 0;
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
    size_t epochs = 1;

    SimpleCNN model;

    /* Load MNIST dataset */
    MNISTLoader loader("MNIST/train-images.idx3-ubyte", "MNIST/train-labels.idx1-ubyte",
        "MNIST/t10k-images.idx3-ubyte", "MNIST/t10k-labels.idx1-ubyte");
    if (!loader.loadTrainData() or !loader.loadTestData()) {
        std::cerr << "Error: Loading data failed." << std::endl;
        return EXIT_FAILURE;
    }

    trainSimpleCNN(loader, model, epochs);

    testSimpleCNN(loader, model);

    return EXIT_SUCCESS;
}
