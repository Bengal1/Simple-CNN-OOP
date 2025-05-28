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
* new functions: save model, load model
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
    double _testAccuracy = 0.0;
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
    ReLU<std::vector<Eigen::MatrixXd>> _relu1;
    ReLU<std::vector<Eigen::MatrixXd>> _relu2;
    ReLU<Eigen::VectorXd> _relu3;
    Softmax<Eigen::VectorXd> _softmax;
    // Loss function
    CrossEntropy CEloss;
    // **Training parameters**
    const size_t _classes;

    Eigen::VectorXd testLogit;//Debug!

public:
    SimpleCNN(size_t classes = 10) :
        _conv1(28, 28, 1, 32, 5),
        _pool1(24, 24, 32, 2),
        _conv2(12, 12, 32, 64, 5),
        _pool2(8, 8, 64, 2),
        _fc1(4 * 4 * 64, 512),
        _fc2(512, classes),

        _dropout1(/*dropoutRate*/0.45),
        _dropout2(/*dropoutRate*/0.35),
        _bn1(),
        _bn2(),

        _classes(classes)
    {
        if (classes < 2) {
            throw std::invalid_argument("[SimpleCNN]: Number of classes must be greater than 1.");
        }
		testLogit = Eigen::VectorXd::Zero(classes);//Debug!
    }

    void trainSimpleCNN(MNISTLoader& dataLoader,
                        const size_t epochs = 10)
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
            // Training
            std::vector<Eigen::VectorXd> trainOutput(numTrainImages,
                Eigen::VectorXd(_classes));
            _trainingStats[TRAIN_LOSS][epoch - 1] = _trainEpoch(trainImages,
                oneHotTrainLabels,
                trainOutput);
            _trainingStats[TRAIN_ACCURACY][epoch - 1] = accuracyCalculation(trainOutput,
                oneHotTrainLabels);
            // Validation
            std::vector<Eigen::VectorXd> validationOutput(numValidationImages,
                Eigen::VectorXd(_classes));
            _trainingStats[VALIDATION_LOSS][epoch - 1] = _validateEpoch(validationImages,
                oneHotValidationLabels,
                validationOutput);
            _trainingStats[VALIDATION_ACCURACY][epoch - 1] = accuracyCalculation(validationOutput,
                oneHotValidationLabels);
			// Print epoch statistics
			_printEpoch(epoch);
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

        size_t testImageNum = 0;
        for (Eigen::MatrixXd image : testImages) {
            Eigen::VectorXd singleTestOutput = _ForwardPass(image);
            testOutput[testImageNum] = singleTestOutput;

            ++testImageNum;
        }
        _testAccuracy = accuracyCalculation(testOutput, oneHotTestLabels);
        std::cout << "\nTest Accuracy: " << _testAccuracy << "%\n" << std::endl;
    }

    std::vector<std::vector<double>> getLastTrainingStats() const {
        return _trainingStats;
    }

    const double getLastTestAccuracy() const {
        return _testAccuracy;
    }

    void exportTrainingDataToCSV(std::filesystem::path
        tragetPath = "training_data.csv") const
    {
        if (_trainingStats.empty()) {
            std::cerr << "Training data is empty." << std::endl;
            return;
            //throw std::invalid_argument("Training data is empty.");
        }
        // Check if the path is empty, if so, use the default file name
        if (tragetPath.empty()) {
            tragetPath = "training_data.csv";
        }
        // Open the file for writing
        std::ofstream file(tragetPath);
        if (file.is_open()) {
            file << "Epoch,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy\n";
            for (size_t i = 0; i < _trainingStats[TRAIN_LOSS].size(); ++i) {
                file << i + 1 << ","
                    << _trainingStats[TRAIN_LOSS][i] << ","
                    << _trainingStats[TRAIN_ACCURACY][i] << ","
                    << _trainingStats[VALIDATION_LOSS][i] << ","
                    << _trainingStats[VALIDATION_ACCURACY][i] << "\n";
            }
            file.close();
        }
        else {
            throw std::ios_base::failure("Unable to open file for writing.");
        }
    }

    //void saveModelParameters() const {}
	//void loadModelParameters() {}

private:
    Eigen::VectorXd _ForwardPass(const Eigen::MatrixXd& input) {
        // First convolution block
        std::vector<Eigen::MatrixXd> outputConv1 = _conv1.forward(input);
        std::vector<Eigen::MatrixXd> outputBN1 = _bn1.forward(outputConv1);
        std::vector<Eigen::MatrixXd> outputRelu1 = _relu1.Activate(outputBN1);
        std::vector<Eigen::MatrixXd> outputPool1 = _pool1.forward(outputRelu1);
        std::vector<Eigen::MatrixXd> outputDrop1 = _dropout1.forward(outputPool1);
        // Second convolution block
        std::vector<Eigen::MatrixXd> outputConv2 = _conv2.forward(outputDrop1);
        std::vector<Eigen::MatrixXd> outputBN2 = _bn2.forward(outputConv2);
        std::vector<Eigen::MatrixXd> outputRelu2 = _relu2.Activate(outputBN2);
        std::vector<Eigen::MatrixXd> outputPool2 = _pool2.forward(outputRelu2);
        std::vector<Eigen::MatrixXd> outputDrop2 = _dropout2.forward(outputPool2);
        // Fully connected layers
        Eigen::VectorXd outputFc1 = _fc1.forward(outputDrop2);
        Eigen::VectorXd outputRelu3 = _relu3.Activate(outputFc1);
        Eigen::VectorXd outputFc2 = _fc2.forward(outputRelu3);
		testLogit = outputFc2; // Debug: store logits for testing
        Eigen::VectorXd outputSoftmax = _softmax.Activate(outputFc2);

        return outputSoftmax;
    }

    void _Backpropagation(const Eigen::VectorXd& softmaxCrossEntropyGradient) {
        // Fully connected layers
        Eigen::VectorXd fc2BackGrad = _fc2.backward<Eigen::VectorXd>(softmaxCrossEntropyGradient);
        Eigen::VectorXd relu3BackGrad = _relu3.computeGradient(fc2BackGrad);
        std::vector<Eigen::MatrixXd> fc1BackGrad = _fc1.backward<
                                std::vector<Eigen::MatrixXd>>(relu3BackGrad);
        // Second convolution block
        std::vector<Eigen::MatrixXd> dropout2BackGrad = _dropout2.backward(fc1BackGrad);
        std::vector<Eigen::MatrixXd> pool2BackGrad = _pool2.backward(dropout2BackGrad);
        std::vector<Eigen::MatrixXd> relu2BackGrad = _relu2.computeGradient(pool2BackGrad);
        std::vector<Eigen::MatrixXd> bn2BackGrad = _bn2.backward(relu2BackGrad);
        std::vector<Eigen::MatrixXd> conv2BackGrad = _conv2.backward(bn2BackGrad);
        // First convolution block
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

    const double _trainEpoch(const std::vector<Eigen::MatrixXd>& trainImages,
        const std::vector<Eigen::VectorXd>& oneHotTrainLabels,
        std::vector<Eigen::VectorXd>& trainOutput)
    {
        size_t trainImageNum = 0;
        double trainLoss = 0.0;
        _setTrainingMode(true);

        for (Eigen::MatrixXd image : trainImages) {
            /*Forward pass*/
            Eigen::VectorXd singleTrainOutput = _ForwardPass(image);
            trainOutput[trainImageNum] = singleTrainOutput;
            /*Loss*/
            trainLoss += CEloss.calculateLoss(singleTrainOutput,
                oneHotTrainLabels[trainImageNum]);
            Eigen::VectorXd lossGrad = CEloss.softmaxCrossEntropyGradient(
                singleTrainOutput, oneHotTrainLabels[trainImageNum]);
            /*Backpropagation*/
            _Backpropagation(lossGrad);
            /*Optimizer step*/
            _updateParameters();

            /*Debug!*/
            if (trainImageNum % 1000 == 0) {
                std::cout << trainImageNum << ": " << std::endl;
                //std::cout << singleTrainOutput << std::endl << std::endl;
                std::cout << testLogit << std::endl << std::endl;
                if (isnan(singleTrainOutput[0])) {
                    std::cout << "\nimage No. : " << trainImageNum << std::endl;
                    exit(-1);
                }
            }
			int numcheck = 10000;
            if (trainImageNum % numcheck == 0 and trainImageNum != 0) {
                std::vector<Eigen::VectorXd> tempTrainO(&trainOutput[0], &trainOutput[trainImageNum]);
                std::vector<Eigen::VectorXd> tempTrainL(&oneHotTrainLabels[0], &oneHotTrainLabels[trainImageNum]);
                std::cout << "Train Accuracy: " << accuracyCalculation(tempTrainO, tempTrainL) << "%\n" << std::endl;
            }
            /*Debug!*/

            ++trainImageNum;
        }

        return trainLoss / static_cast<double>(trainImageNum);
    }

    const double _validateEpoch(const std::vector<Eigen::MatrixXd>& validationImages,
        const std::vector<Eigen::VectorXd>& oneHotValidationLabels,
        std::vector<Eigen::VectorXd>& validationOutput)
    {
        size_t validationImageNum = 0;
        double validationLoss = 0.0;
        _setTrainingMode(false);

        for (Eigen::MatrixXd image : validationImages) {
            Eigen::VectorXd singleValidationOutput = _ForwardPass(image);
            validationOutput[validationImageNum] = singleValidationOutput;
            validationLoss += CEloss.calculateLoss(singleValidationOutput,
                oneHotValidationLabels[validationImageNum]);

            ++validationImageNum;
        }

        return validationLoss / static_cast<double>(validationImageNum);
    }

    void _setTrainingMode(bool isTraining) {
        _bn1.setTrainingMode(isTraining);
        _bn2.setTrainingMode(isTraining);
        _dropout1.setTrainingMode(isTraining);
        _dropout2.setTrainingMode(isTraining);
    }

    void _initializeTrainingStats(size_t epochs) {
        if (epochs <= 0) {
            throw std::invalid_argument("[SimpleCNN]: Number of epochs must be greater than 0.");
        }
        /* Initialize training statistics */
        _trainingStats.clear();
        _trainingStats.resize(NUM_TRAIN_STATS, std::vector<double>(epochs, 0.0));
    }

	void _printEpoch(const size_t epoch) const {
		if (epoch < 1 || epoch > _trainingStats[TRAIN_LOSS].size()) {
			std::cerr << "[SimpleCNN]: Invalid epoch number." << std::endl;
			return;
		}
		std::cout << "Epoch " << epoch << ": "
			<< "Train Loss: " << _trainingStats[TRAIN_LOSS][epoch - 1] << ", "
			<< "Train Accuracy: " << _trainingStats[TRAIN_ACCURACY][epoch - 1] << "% | "
			<< "Validation Loss: " << _trainingStats[VALIDATION_LOSS][epoch - 1] << ", "
			<< "Validation Accuracy: " << _trainingStats[VALIDATION_ACCURACY][epoch - 1] << "%"
			<< std::endl;
	}

    void _printTrainingStats() const {
        if (_trainingStats.empty()) {
            std::cout << "[SimpleCNN]: No training statistics available." << std::endl;
            return;
        }
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

class TrainableLayer {
public:
    virtual void updateParameters() = 0;
    virtual void assignOptimizer(std::unique_ptr<Optimizer> opt) = 0;
    virtual ~TrainableLayer() = default;
};

