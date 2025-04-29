#pragma once 

#include "Utils.hpp"
#include "Layers/Convolution2D.hpp"
#include "Layers/FullyConnected.hpp"
#include "Layers/MaxPooling.hpp"
#include "Activation.hpp"
#include "MNISTLoader.hpp"
#include "LossFunction.hpp"
#include "Regularization.hpp"

/*TODO:
* new functions: save model, load model, save loss to csv
*/
class SimpleCNN {
private:
    // **Training statistics index**
    enum TrainingStatIndex {
        TRAIN_LOSS = 0,
        TRAIN_ACCURACY = 1,
        VALIDATION_LOSS = 2,
        VALIDATION_ACCURACY = 3
    };
	// **Training statistics**
    std::vector<std::vector<double>> _trainingStats;
    double _testAccuracy;

    // Layers
    Convolution2D _conv1;
    Convolution2D _conv2;
    FullyConnected _fc1;
    FullyConnected _fc2;
    MaxPooling _pool1;
    MaxPooling _pool2;
    
    // Regularization
    Dropout _dropout1;
    Dropout _dropout2;
	
    // Loss function
    CrossEntropy CEloss;
    
	// **Training parameters**
    const size_t _classes;


public:
    SimpleCNN(size_t classes = 10) :
        _conv1(28, 28, 1, 32, 5, std::make_unique<ReLU>()),
        _pool1(24, 24, 32, 2),
        _conv2(12, 12, 32, 64, 5, std::make_unique<ReLU>()),
        _pool2(8, 8, 64, 2),
        _fc1(4 * 4 * 64, 512, std::make_unique<ReLU>()),
        _fc2(512, 10, std::make_unique<Softmax>()),
        _dropout1(0.45),
		_dropout2(0.35),
		CEloss(),
		_classes(classes)
	{}

    void trainSimpleCNN(MNISTLoader& dataLoader,
                        const size_t epochs = 20)
    {
        /* Load MNIST Train dataset */
        const std::vector<Eigen::MatrixXd>& trainImages =
            dataLoader.getTrainImages();
        const std::vector<Eigen::VectorXd>& oneHotTrainLabels =
            dataLoader.getOneHotTrainLabels();
        size_t numTrainImages = dataLoader.getNumTrain();

        std::vector<Eigen::VectorXd> trainOutput(numTrainImages,
            Eigen::VectorXd(_classes));

        /* Load MNIST Validation dataset */
        const std::vector<Eigen::MatrixXd>& validationImages =
            dataLoader.getValidationImages();
        const std::vector<Eigen::VectorXd>& oneHotValidationLabels =
            dataLoader.getOneHotValidationLabels();
        size_t numValidationImages = dataLoader.getNumValidation();

        

        std::cout << "\nStart training..." << std::endl;

        std::vector<double> trainLoss(epochs);
        std::vector<double> trainAccuracy(epochs);
        std::vector<double> validationLoss(epochs);
        std::vector<double> validationAccuracy(epochs);

        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            std::cout << "\nepoch #" << (epoch) << std::endl; // Debug!
            size_t trainImageNum = 0;
            for (Eigen::MatrixXd image : trainImages) {
                /*Forward pass*/
                Eigen::VectorXd singleTrainOutput = _ForwardPass(image);
                trainOutput[trainImageNum] = singleTrainOutput;
                /*Loss*/
                trainLoss[epoch] += CEloss.calculateLoss(singleTrainOutput,
                    oneHotTrainLabels[trainImageNum]);
                Eigen::VectorXd lossGrad = CEloss.calculateGradient(
                    singleTrainOutput, oneHotTrainLabels[trainImageNum]);
                /*Backpropagation*/
                _Backpropagation(lossGrad);

                _updateParameters();
                /*Debug!*/
                if (trainImageNum % 1000 == 0) {
                    std::cout << trainImageNum << ": " << std::endl;
                    std::cout << singleTrainOutput << std::endl << std::endl;
                    if (isnan(singleTrainOutput[0])) {
                        std::cout << "\nimage No. : " << trainImageNum << std::endl;
                        exit(-1);
                    }
                }
                if (trainImageNum % 10000 == 0 and trainImageNum != 0) {
                    std::vector<Eigen::VectorXd> tempTrainO(&trainOutput[0], &trainOutput[trainImageNum]);
                    std::vector<Eigen::VectorXd> tempTrainL(&oneHotTrainLabels[0], &oneHotTrainLabels[trainImageNum]);
                    std::cout << "Train Accuracy: " << accuracyCalculation(tempTrainO, tempTrainL) << "%\n" << std::endl;
                }
                /*Debug!*/

                ++trainImageNum;
            }
            std::vector<Eigen::VectorXd> validationOutput(numValidationImages,
                                                          Eigen::VectorXd(_classes));
			validationLoss[epoch] = _validateEpoch(validationImages, 
                                                   oneHotValidationLabels, 
                                                   numValidationImages);
            
            validationAccuracy[epoch] = accuracyCalculation(validationOutput, oneHotValidationLabels);
            trainAccuracy[epoch] = accuracyCalculation(trainOutput, oneHotTrainLabels);
            std::cout << "Epoch" << epoch << ": Train Loss: " << trainLoss[epoch] << " ; Train Accuracy: " << trainAccuracy[epoch] << "% "
                << "| Validation Loss: " << validationLoss[epoch] << " ; Validation Accuracy: " << validationAccuracy[epoch] << "% " << std::endl;
        }

        _trainingStats.assign(4, std::vector<double>(epochs, 0.0));
        _trainingStats[TRAIN_LOSS] = trainLoss;
        _trainingStats[TRAIN_ACCURACY] = trainAccuracy;
        _trainingStats[VALIDATION_LOSS] = validationLoss;
        _trainingStats[VALIDATION_ACCURACY] = validationAccuracy;
    }

    void testSimpleCNN(MNISTLoader& dataLoader)
    {
        /* Load MNIST Test dataset */
        const std::vector<Eigen::MatrixXd>& testImages = dataLoader.getTestImages();
        const std::vector<Eigen::VectorXd>& oneHotTestLabels =
            dataLoader.getOneHotTestLabels();
        size_t numTestImages = dataLoader.getNumTest();

        std::vector<Eigen::VectorXd> testOutput(numTestImages,
            Eigen::VectorXd(_classes));

        std::cout << "\nStart testing...\n" << std::endl;

        double testAccuracy = 0.0;
        size_t testImageNum = 0;
        for (Eigen::MatrixXd image : testImages) {
            Eigen::VectorXd singleTestOutput = _ForwardPass(image);
            testOutput[testImageNum] = singleTestOutput;

            ++testImageNum;
        }
        _testAccuracy = accuracyCalculation(testOutput, oneHotTestLabels);
        std::cout << "\nTest Accuracy: " << _testAccuracy << "%\n" << std::endl;
    }

    std::vector<std::vector<double>> getLastTrainingStats() {
		return _trainingStats;
	}

	double getLastTestAccuracy() {
		return _testAccuracy;
	}

private:
    Eigen::VectorXd _ForwardPass(Eigen::MatrixXd& input) {
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

    void _Backpropagation(Eigen::VectorXd& lossGradient) {
        /*Backward*/
        Eigen::VectorXd fc2BackGrad = _fc2.backward(lossGradient);
        std::vector<Eigen::MatrixXd> fc1BackGrad = _fc1.backward(fc2BackGrad, true);
        std::vector<Eigen::MatrixXd> pool2BackGrad = _pool2.backward(fc1BackGrad);
        std::vector<Eigen::MatrixXd> conv2BackGrad = _conv2.backward(pool2BackGrad);
        std::vector<Eigen::MatrixXd> pool1BackGrad = _pool1.backward(conv2BackGrad);
        _conv1.backward(pool1BackGrad);
    }

    void _updateParameters() {
        _conv1.updateParameters();
        _conv2.updateParameters();
        _fc1.updateParameters();
        _fc2.updateParameters();
    }

    double _trainEpoch(const std::vector<Eigen::MatrixXd>& trainImages, 
                       const std::vector<Eigen::VectorXd>& oneHotTrainLabels,
                       const size_t  numTrainImages)
    {
        size_t trainImageNum = 0;
		double trainLoss = 0.0;
        std::vector<Eigen::VectorXd> trainOutput(numTrainImages,
            Eigen::VectorXd(_classes));
        for (Eigen::MatrixXd image : trainImages) {
            /*Forward pass*/
            Eigen::VectorXd singleTrainOutput = _ForwardPass(image);
            trainOutput[trainImageNum] = singleTrainOutput;
            /*Loss*/
            trainLoss += CEloss.calculateLoss(singleTrainOutput,
                oneHotTrainLabels[trainImageNum]);
            Eigen::VectorXd lossGrad = CEloss.calculateGradient(
                singleTrainOutput, oneHotTrainLabels[trainImageNum]);
            /*Backpropagation*/
            _Backpropagation(lossGrad);

            _updateParameters();
            /*Debug!*/
            if (trainImageNum % 1000 == 0) {
                std::cout << trainImageNum << ": " << std::endl;
                std::cout << singleTrainOutput << std::endl << std::endl;
                if (isnan(singleTrainOutput[0])) {
                    std::cout << "\nimage No. : " << trainImageNum << std::endl;
                    exit(-1);
                }
            }
            if (trainImageNum % 10000 == 0 and trainImageNum != 0) {
                std::vector<Eigen::VectorXd> tempTrainO(&trainOutput[0], &trainOutput[trainImageNum]);
                std::vector<Eigen::VectorXd> tempTrainL(&oneHotTrainLabels[0], &oneHotTrainLabels[trainImageNum]);
                std::cout << "Train Accuracy: " << accuracyCalculation(tempTrainO, tempTrainL) << "%\n" << std::endl;
            }
            /*Debug!*/

            ++trainImageNum;
        }

		return trainLoss;
    }

    double _validateEpoch(const std::vector<Eigen::MatrixXd>& validationImages, 
                          const std::vector<Eigen::VectorXd>& oneHotValidationLabels, 
                          const size_t numValidationImages)
    {
        size_t validationImageNum = 0;
		double validationLoss = 0.0;
        std::vector<Eigen::VectorXd> validationOutput(numValidationImages,
                                                      Eigen::VectorXd(_classes));

        for (Eigen::MatrixXd image : validationImages) {
            Eigen::VectorXd singleValidationOutput = _ForwardPass(image);
            validationOutput[validationImageNum] = singleValidationOutput;
            validationLoss += CEloss.calculateLoss(singleValidationOutput,
                oneHotValidationLabels[validationImageNum]);
            ++validationImageNum;
        }

        return validationLoss;
    }
	
};




int main()
{
    size_t epochs = 10, classes = 10;
	double validationRatio = 0.2;

	std::filesystem::path trainImage = "train-images.idx3-ubyte";
	std::filesystem::path trainLabel = "train-labels.idx1-ubyte";
	std::filesystem::path testImage = "t10k-images.idx3-ubyte";
	std::filesystem::path testLabel = "t10k-labels.idx1-ubyte";
 
    SimpleCNN model(classes);

    /* Load MNIST dataset */
    MNISTLoader loader("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
                       "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 
                        validationRatio);
    if (!loader.loadTrainData() || !loader.loadTestData()) {
        return -1;
    }

    model.trainSimpleCNN(loader, epochs);

    model.testSimpleCNN(loader);

    return 0;
}


/*void getModelState() {
    std::vector<Eigen::MatrixXd> conv1Filters =  _conv1.getFilters();
    Eigen::VectorXd conv1Biases = _conv1.getBiases();
    std::vector<Eigen::MatrixXd> conv2Filters = _conv2.getFilters();
    Eigen::VectorXd conv2Biases = _conv2.getBiases();
    Eigen::MatrixXd fc1Weights = _fc1.getWeights();
    Eigen::VectorXd fc1Bias = _fc1.getBias();
    Eigen::MatrixXd fc2Weights = _fc2.getWeights();
    Eigen::VectorXd fc2Bias = _fc2.getBias();
}

void setModelState() {
    _conv1.setParameters();
    _conv2.setParameters();
    _fc1.setParameters();
    _fc2.setParameters();
}*/


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
