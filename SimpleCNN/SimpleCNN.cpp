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
        TRAIN_ACCURACY,
        VALIDATION_LOSS,
        VALIDATION_ACCURACY,
        NUM_TRAIN_STATS
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
	BatchNormalization _bn1;
	BatchNormalization _bn2;
    Dropout _dropout1;
    Dropout _dropout2;

	// Activation functions
    ReLU _relu1;
	ReLU _relu2;
	ReLU _relu3;
	Softmax _softmax;
	
    // Loss function
    CrossEntropy CEloss;
    
	// **Training parameters**
    const size_t _classes;

public:
    SimpleCNN(size_t classes = 10) :
        _conv1(28, 28, 1, 32, 5),
        _pool1(24, 24, 32, 2),
        _conv2(12, 12, 32, 64, 5),
        _pool2(8, 8, 64, 2),
        _fc1(4 * 4 * 64, 512),
        _fc2(512, classes),
        _dropout1(0.45),
		_dropout2(0.35),
		CEloss(),
		_classes(classes),
		_testAccuracy(0.0)
	{
		if (classes <= 2) {
			throw std::invalid_argument("[SimpleCNN]: Number of classes must be greater than 2.");
		}

    }

    void trainSimpleCNN(MNISTLoader& dataLoader,
                        const size_t epochs = 20)
    {
		if (epochs <= 0) {
			throw std::invalid_argument("[SimpleCNN]: Number of epochs must be greater than 0.");
		}
        /* Load MNIST Train dataset */
        const std::vector<Eigen::MatrixXd>& trainImages =
            dataLoader.getTrainImages();
        const std::vector<Eigen::VectorXd>& oneHotTrainLabels =
            dataLoader.getOneHotTrainLabels();
        size_t numTrainImages = dataLoader.getNumTrain();

        /* Load MNIST Validation dataset */
        const std::vector<Eigen::MatrixXd>& validationImages =
            dataLoader.getValidationImages();
        const std::vector<Eigen::VectorXd>& oneHotValidationLabels =
            dataLoader.getOneHotValidationLabels();
        size_t numValidationImages = dataLoader.getNumValidation();

        std::cout << "\nStart training...\n" << std::endl;

        _initializeTrainingStats(epochs);

        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            //std::cout << "\nepoch #" << (epoch) << std::endl; // Debug!
            std::vector<Eigen::VectorXd> trainOutput(numTrainImages,
                                            Eigen::VectorXd(_classes));
            _trainingStats[TRAIN_LOSS][epoch - 1] = _trainEpoch(trainImages,
                                                                oneHotTrainLabels, 
                                                                trainOutput, 
                                                                numTrainImages);
			_trainingStats[TRAIN_ACCURACY][epoch - 1] = accuracyCalculation(trainOutput,
				                                                      oneHotTrainLabels);
            std::vector<Eigen::VectorXd> validationOutput(numValidationImages,
                                                      Eigen::VectorXd(_classes));
            _trainingStats[VALIDATION_LOSS][epoch - 1] = _validateEpoch(validationImages,
                                                                        oneHotValidationLabels, 
                                                                        validationOutput,
                                                                        numValidationImages);
			_trainingStats[VALIDATION_ACCURACY][epoch - 1] = accuracyCalculation(validationOutput,
				                                                            oneHotValidationLabels);

			std::cout << "Epoch" << epoch << ":"
                << "Train Loss = " << _trainingStats[TRAIN_LOSS][epoch - 1] << " ; "
                << "Train Accuracy = " << _trainingStats[TRAIN_ACCURACY][epoch - 1] << "% | "
                << "Validation Loss = " << _trainingStats[VALIDATION_LOSS][epoch - 1] << " ; "
                << "Validation Accuracy = " << _trainingStats[VALIDATION_ACCURACY][epoch - 1] << "%"
                << std::endl;
        }
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
		std::vector<Eigen::MatrixXd> outputBN1 = _bn1.forward(outputConv1);
		std::vector<Eigen::MatrixXd> outputRelu1 = _relu1.Activate(outputBN1);
        std::vector<Eigen::MatrixXd> outputPool1 = _pool1.forward(outputRelu1);
        std::vector<Eigen::MatrixXd> outputDrop1 = _dropout1.forward(outputPool1);

        std::vector<Eigen::MatrixXd> outputConv2 = _conv2.forward(outputDrop1);
		std::vector<Eigen::MatrixXd> outputBN2 = _bn2.forward(outputConv2);
		std::vector<Eigen::MatrixXd> outputRelu2 = _relu2.Activate(outputBN2);
        std::vector<Eigen::MatrixXd> outputPool2 = _pool2.forward(outputRelu2);
        std::vector<Eigen::MatrixXd> outputDrop2 = _dropout2.forward(outputPool2);

        Eigen::VectorXd outputFc1 = _fc1.forward(outputDrop2);
		Eigen::VectorXd outputRelu3 = _relu3.Activate(outputFc1);
        Eigen::VectorXd outputFc2 = _fc2.forward(outputRelu3);
        Eigen::VectorXd outputSoftmax = _softmax.Activate(outputFc2);

        return outputSoftmax;
    }

    void _Backpropagation(Eigen::VectorXd& lossGradient) {
        /*Backward*/
        Eigen::VectorXd softmaxBackGrad = _softmax.computeGradient(lossGradient);
        Eigen::VectorXd fc2BackGrad = _fc2.backward(softmaxBackGrad);
		Eigen::VectorXd relu3BackGrad = _relu3.computeGradient(fc2BackGrad);
		std::vector<Eigen::MatrixXd> fc1BackGrad = _fc1.backward(relu3BackGrad, true);

		std::vector<Eigen::MatrixXd> dropout2BackGrad = _dropout2.backward(fc1BackGrad);
		std::vector<Eigen::MatrixXd> pool2BackGrad = _pool2.backward(dropout2BackGrad);
		std::vector<Eigen::MatrixXd> relu2BackGrad = _relu2.computeGradient(pool2BackGrad);
		std::vector<Eigen::MatrixXd> bn2BackGrad = _bn2.backward(relu2BackGrad);
		std::vector<Eigen::MatrixXd> conv2BackGrad = _conv2.backward(bn2BackGrad);

		std::vector<Eigen::MatrixXd> dropout1BackGrad = _dropout1.backward(conv2BackGrad);
		std::vector<Eigen::MatrixXd> pool1BackGrad = _pool1.backward(dropout1BackGrad);
		std::vector<Eigen::MatrixXd> relu1BackGrad = _relu1.computeGradient(pool1BackGrad);
		std::vector<Eigen::MatrixXd> bn1BackGrad = _bn1.backward(relu1BackGrad);
		_conv1.backward(bn1BackGrad);
    }

    void _updateParameters() {
        _conv1.updateParameters();
		_bn1.updateParameters();
        _conv2.updateParameters();
		_bn2.updateParameters();
        _fc1.updateParameters();
        _fc2.updateParameters();
    }
	void _initializeTrainingStats(size_t epochs) {
		if (epochs <= 0) {
			throw std::invalid_argument("[SimpleCNN]: Number of epochs must be greater than 0.");
		}
		/* Initialize training statistics */
		_trainingStats.clear();
		_trainingStats.resize(NUM_TRAIN_STATS, std::vector<double>(epochs, 0.0));
	}

    double _trainEpoch(const std::vector<Eigen::MatrixXd>& trainImages, 
                       const std::vector<Eigen::VectorXd>& oneHotTrainLabels,
                       std::vector<Eigen::VectorXd>& trainOutput,
                       const size_t  numTrainImages)
    {
        size_t trainImageNum = 0;
		double trainLoss = 0.0;
		setTrainingMode(true);

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
            /*Optimizer step*/
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

		return trainLoss / static_cast<double>(numTrainImages);
    }

    double _validateEpoch(const std::vector<Eigen::MatrixXd>& validationImages, 
                          const std::vector<Eigen::VectorXd>& oneHotValidationLabels,
                          std::vector<Eigen::VectorXd>& validationOutput,
                          const size_t numValidationImages)
    {
        size_t validationImageNum = 0;
		double validationLoss = 0.0;
		setTrainingMode(false);

        for (Eigen::MatrixXd image : validationImages) {
            Eigen::VectorXd singleValidationOutput = _ForwardPass(image);
            validationOutput[validationImageNum] = singleValidationOutput;
            validationLoss += CEloss.calculateLoss(singleValidationOutput,
                oneHotValidationLabels[validationImageNum]);
            ++validationImageNum;
        }

        return validationLoss / static_cast<double>(numValidationImages);
    }

	void setTrainingMode(bool isTraining) {
		_bn1.setTrainingMode(isTraining);
		_bn2.setTrainingMode(isTraining);
		_dropout1.setTrainingMode(isTraining);
		_dropout2.setTrainingMode(isTraining);
	}
	

	void _printTrainingStats() {
		std::cout << "\nTraining statistics:" << std::endl;
		for (size_t i = 0; i < _trainingStats[TRAIN_LOSS].size(); ++i) {
			std::cout << "Epoch " << i + 1 << ": "
				<< "Train Loss: " << _trainingStats[TRAIN_LOSS][i] << ", "
				<< "Train Accuracy: " << _trainingStats[TRAIN_ACCURACY][i] << "% | "
				<< "Validation Loss: " << _trainingStats[VALIDATION_LOSS][i] << ", "
				<< "Validation Accuracy: " << _trainingStats[VALIDATION_ACCURACY][i] << "%"
				<< std::endl;
		}
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
    MNISTLoader loader(trainImage, trainLabel,
                       testImage, testLabel, validationRatio);
    if (!loader.loadTrainData() || !loader.loadTestData()) {
        return -1;
    }

    model.trainSimpleCNN(loader, epochs);

    model.testSimpleCNN(loader);

    return 0;
}


/*Eigen::VectorXd _ForwardPass(Eigen::MatrixXd& input) {

    std::vector<Eigen::MatrixXd> outputConv1 = _conv1.forward(input);
    std::vector<Eigen::MatrixXd> outputPool1 = _pool1.forward(outputConv1);
    std::vector<Eigen::MatrixXd> outputDrop1 = _dropout1.forward(outputPool1);

    std::vector<Eigen::MatrixXd> outputConv2 = _conv2.forward(outputDrop1);
    std::vector<Eigen::MatrixXd> outputPool2 = _pool2.forward(outputConv2);
    std::vector<Eigen::MatrixXd> outputDrop2 = _dropout2.forward(outputPool2);

    Eigen::VectorXd outputFc1 = _fc1.forward(outputDrop2);
    Eigen::VectorXd outputFc2 = _fc2.forward(outputFc1);

    return outputFc2;
}*/

/*void _Backpropagation(Eigen::VectorXd& lossGradient) {
    Eigen::VectorXd fc2BackGrad = _fc2.backward(lossGradient);
    std::vector<Eigen::MatrixXd> fc1BackGrad = _fc1.backward(fc2BackGrad, true);
    std::vector<Eigen::MatrixXd> pool2BackGrad = _pool2.backward(fc1BackGrad);
    std::vector<Eigen::MatrixXd> conv2BackGrad = _conv2.backward(pool2BackGrad);
    std::vector<Eigen::MatrixXd> pool1BackGrad = _pool1.backward(conv2BackGrad);
    _conv1.backward(pool1BackGrad);
}*/

/*void _setTrainigStats(const std::vector<double> trainLoss,
                      const std::vector<double> trainAccuracy,
                      const std::vector<double> validationLoss,
                      const std::vector<double> validationAccuracy,
                      const size_t epochs)
{
    _trainingStats.clear();
    _trainingStats.assign(NUM_TRAIN_STATS, std::vector<double>(epochs, 0.0));

    _trainingStats[TRAIN_LOSS] = trainLoss;
    _trainingStats[TRAIN_ACCURACY] = trainAccuracy;
    _trainingStats[VALIDATION_LOSS] = validationLoss;
    _trainingStats[VALIDATION_ACCURACY] = validationAccuracy;
}*/


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
