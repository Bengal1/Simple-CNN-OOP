#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

class MNISTLoader {
private:
	// MNIST file paths
    std::string _trainImagesFile;
    std::string _trainLabelsFile;
    std::string _testImagesFile;
    std::string _testLabelsFile;
	// MNIST images
    std::vector<Eigen::MatrixXd> _trainImages;
    std::vector<Eigen::MatrixXd> _validationImages;
    std::vector<Eigen::MatrixXd> _testImages;
	// MNIST labels
    std::vector<uint8_t> _trainLabels;
    std::vector<uint8_t> _validationLabels;
    std::vector<uint8_t> _testLabels;
	// One-hot encoded labels
    std::vector<Eigen::VectorXd> _oneHotTrainLabels;
    std::vector<Eigen::VectorXd> _oneHotTestLabels;
    std::vector<Eigen::VectorXd> _oneHotValidationLabels;
    
    size_t numTrain;
    size_t numValidation;
    size_t numTest;
    bool _splitValidation;
    double _validationRatio;

public:
    

    MNISTLoader(const std::string& trainImagesFile, const std::string& trainLabelsFile,
        const std::string& testImagesFile, const std::string& testLabelsFile,
        bool splitValidation = false, double validationRatio = 0.0) :
        _trainImagesFile(trainImagesFile), _trainLabelsFile(trainLabelsFile),
        _testImagesFile(testImagesFile), _testLabelsFile(testLabelsFile),
        _splitValidation(splitValidation), _validationRatio(validationRatio), 
        numTrain(0), numValidation(0), numTest(0)
    {
		if (_validationRatio <= 0.0 || _validationRatio >= 1.0) {
			throw std::invalid_argument("Validation ratio must be in the range (0, 1).");
		}
		if (_trainImagesFile.empty() || _trainLabelsFile.empty() ||
			_testImagesFile.empty() || _testLabelsFile.empty()) {
			throw std::invalid_argument("File paths cannot be empty.");
		}
    }

    bool loadTrainData() {
        return _loadImages(_trainImagesFile, _trainLabelsFile, 
            _trainImages, _trainLabels, true);
    }

    bool loadTestData() {
        return _loadImages(_testImagesFile, _testLabelsFile, 
            _testImages, _testLabels, false);
    }

    const std::vector<Eigen::MatrixXd>& getTrainImages() const {
        return _trainImages;
    }

    const std::vector<Eigen::MatrixXd>& getValidationImages() const {
        return _validationImages;
    }

    const std::vector<Eigen::MatrixXd>& getTestImages() const {
        return _testImages;
    }

    const std::vector<Eigen::VectorXd>& getOneHotTrainLabels() {
        if (_oneHotTrainLabels.empty()) {
            _createOneHotLabels(_trainLabels, _oneHotTrainLabels);
        }
        return _oneHotTrainLabels;
    }

    const std::vector<Eigen::VectorXd>& getOneHotValidationLabels() {
        if (_oneHotValidationLabels.empty()) {
            _createOneHotLabels(_validationLabels, _oneHotValidationLabels);
        }
        return _oneHotValidationLabels;
    }

    const std::vector<Eigen::VectorXd>& getOneHotTestLabels() {
        if (_oneHotTestLabels.empty()) {
            _createOneHotLabels(_testLabels, _oneHotTestLabels);
        }
        return _oneHotTestLabels;
    }

	const size_t getNumTrain() const {
		return numTrain;
	}
	const size_t getNumValidation() const {
		return numValidation;
	}
	const size_t getNumTest() const {
		return numTest;
	}
	/*void setValidationSplit(double ratio) {
		_validationSplit = ratio;
		_splitTrainValidation(ratio);
	}
	double getValidationSplit() const {
		return _validationSplit;
	}*/


private:

    bool _loadImages(const std::string& imagesFile, const std::string& labelsFile,
        std::vector<Eigen::MatrixXd>& images, std::vector<uint8_t>& labels, bool isTrain) 
    {
        std::ifstream fImages(imagesFile, std::ios::binary);

        if (!fImages.is_open()) {
            std::cerr << "Failed to open images file: " << imagesFile << std::endl;
            return false;
        }
        if (isTrain)
            std::cout << "Reading Train Data From Source." << std::endl;
        else
            std::cout << "Reading Test Data From Source." << std::endl;

        uint32_t magicNumber, numImages, numRows, numCols;
        fImages.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        fImages.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        fImages.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
        fImages.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

        magicNumber = _byteswap_ulong(magicNumber);  //check: '_byteswap_ulong' 
        numImages = _byteswap_ulong(numImages);
        numRows = _byteswap_ulong(numRows);
        numCols = _byteswap_ulong(numCols);

        if (magicNumber != 0x803) {
            std::cerr << "Invalid magic number in images file" << std::endl;
            return false;
        }

        isTrain ? numTrain = numImages : numTest = numImages;

        images.resize(numImages);
        for (size_t i = 0; i < numImages; ++i) {
            Eigen::MatrixXd imageMatrix(numRows, numCols);
            for (size_t r = 0; r < numRows; ++r) {
                for (size_t c = 0; c < numCols; ++c) {
                    uint8_t pixelValue;
                    fImages.read(reinterpret_cast<char*>(&pixelValue), sizeof(pixelValue));
                    imageMatrix(r, c) = static_cast<float>(pixelValue) / 255.0; // Normalize to [0, 1]
                }
            }
            images[i] = imageMatrix;
        }

        fImages.close();

        std::ifstream fLabels(labelsFile, std::ios::binary);

        if (!fLabels.is_open()) {
            std::cerr << "Failed to open labels file: " << labelsFile << std::endl;
            return false;
        }

        uint32_t labelsMagicNumber, numLabels;
        fLabels.read(reinterpret_cast<char*>(&labelsMagicNumber), sizeof(labelsMagicNumber));
        fLabels.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

        labelsMagicNumber = _byteswap_ulong(labelsMagicNumber);
        numLabels = _byteswap_ulong(numLabels);

        assert(numImages == numLabels);

        if (labelsMagicNumber != 0x801) {
            std::cerr << "Invalid magic number in labels file" << std::endl;
            return false;
        }

        labels.resize(numLabels);
        fLabels.read(reinterpret_cast<char*>(labels.data()), numLabels);

        fLabels.close();

        if(_splitValidation && isTrain)
			_splitTrainValidation(_validationRatio);

        return true;
    }

    void _splitTrainValidation(double ratio) {
        if (_trainImages.empty() || _trainLabels.empty()) {
            throw std::runtime_error("Train data not loaded. Call loadTrainData() before splitting.");
        }
        if (ratio <= 0.0 || ratio >= 1.0) {
            throw std::invalid_argument("Validation ratio must be between 0 and 1 (exclusive).");
        }
        std::cout << "Performing train-validation split." << std::endl;


        size_t total = _trainImages.size();
        size_t splitIndex = static_cast<size_t>(total * (1.0 - ratio));

        numValidation = total - splitIndex;
        numTrain = splitIndex;

        _validationImages.assign(_trainImages.begin() + splitIndex, _trainImages.end());
        _validationLabels.assign(_trainLabels.begin() + splitIndex, _trainLabels.end());

        _trainImages.resize(splitIndex);
        _trainLabels.resize(splitIndex);
        _oneHotTrainLabels.clear(); // force regeneration on next access
    }

    void _createOneHotLabels(const std::vector<uint8_t>& labels, 
                    std::vector<Eigen::VectorXd>& oneHotLabels) 
    {
        oneHotLabels.clear();
        for (uint8_t label : labels) {
            Eigen::VectorXd oneHot = Eigen::VectorXd::Zero(10);
            oneHot(static_cast<int>(label)) = 1.0;
            oneHotLabels.push_back(oneHot);
        }
    }
};
