#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <Eigen/Dense>

class MNISTLoader {
private:
	// MNIST file paths
    const std::filesystem::path _trainImagesFile;
    const std::filesystem::path _trainLabelsFile;
    const std::filesystem::path _testImagesFile;
    const std::filesystem::path _testLabelsFile;
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
	// Number of images
    size_t _numTrain = 0;
    size_t _numValidation = 0;
    size_t _numTest = 0;
	size_t _numClasses = 10; // MNIST has 10 classes (0-9)
    // Validation
    bool _splitValidation = false;
    double _validationRatio;

public:
    

    MNISTLoader(const std::filesystem::path& trainImagesFile, 
                const std::filesystem::path& trainLabelsFile,
                const std::filesystem::path& testImagesFile, 
                const std::filesystem::path& testLabelsFile,
                double validationRatio = 0.0) 
        : _trainImagesFile(trainImagesFile), 
          _trainLabelsFile(trainLabelsFile),
          _testImagesFile(testImagesFile), 
          _testLabelsFile(testLabelsFile),
          _validationRatio(validationRatio)
    {
		// Check that the file paths are not empty
        if (_trainImagesFile.empty() || _trainLabelsFile.empty() ||
            _testImagesFile.empty() || _testLabelsFile.empty()) {
            throw std::invalid_argument("[MNISTLoader]: File paths cannot be empty.");
        }
        // Check that each file exists and is a regular file
        for (const auto& path : { _trainImagesFile, _trainLabelsFile, _testImagesFile, _testLabelsFile }) {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("[MNISTLoader]: File not found: " + path.string());
            }
            if (!std::filesystem::is_regular_file(path)) {
                throw std::runtime_error("[MNISTLoader]: Not a regular file: " + path.string());
            }
		}
		// Check that the validation ratio is in the range [0, 1)
		if (_validationRatio < 0.0 || _validationRatio >= 1.0) {
			throw std::invalid_argument("[MNISTLoader]: Validation ratio must be in the range [0, 1).");
		}
		// Check if validation split is needed
		_splitValidation = (_validationRatio > 0) ? true : false;
    }
	~MNISTLoader() = default;

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
		return _numTrain;
	}
	
    const size_t getNumValidation() const {
		return _numValidation;
	}
	
    const size_t getNumTest() const {
		return _numTest;
	}

private:

    bool _loadImages(const std::filesystem::path& imagesFile, 
                     const std::filesystem::path& labelsFile,
                     std::vector<Eigen::MatrixXd>& images, 
                     std::vector<uint8_t>& labels, 
                     bool isTrain) 
    {
        std::ifstream fImages(imagesFile, std::ios::binary);
        if (!fImages) {
            throw std::runtime_error("[MNISTLoader]: Failed to open images file: " + imagesFile.string());
        }
        std::cout << "Reading " << (isTrain ? "Train" : "Test") << " Data From Source." << std::endl;

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
			throw std::runtime_error("[MNISTLoader]: Invalid magic number in images file");
        }

        isTrain ? _numTrain = numImages : _numTest = numImages;

        images.resize(numImages);
        for (size_t i = 0; i < numImages; ++i) {
            Eigen::MatrixXd imageMatrix(numRows, numCols);
            for (size_t r = 0; r < numRows; ++r) {
                for (size_t c = 0; c < numCols; ++c) {
                    uint8_t pixelValue;
                    if (!fImages.read(reinterpret_cast<char*>(&pixelValue), sizeof(pixelValue))) {
                        fImages.close();
                        throw std::runtime_error("[MNISTLoader]: Unexpected end of file while reading image data in " + imagesFile.string());
                    }
                    imageMatrix(r, c) = static_cast<float>(pixelValue) / 255.0f; // Normalize to [0, 1]
                }
            }
            images[i] = imageMatrix;
        }

        fImages.close();

        std::ifstream fLabels(labelsFile, std::ios::binary);

        if (!fLabels) {
            throw std::runtime_error("[MNISTLoader]: Failed to open labels file: " + labelsFile.string());
        }

        uint32_t labelsMagicNumber, numLabels;
        fLabels.read(reinterpret_cast<char*>(&labelsMagicNumber), sizeof(labelsMagicNumber));
        fLabels.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

        labelsMagicNumber = _byteswap_ulong(labelsMagicNumber);
        numLabels = _byteswap_ulong(numLabels);
       
        if (labelsMagicNumber != 0x801) {
            throw std::runtime_error("[MNISTLoader]: Invalid magic number in labels file");
        }

        if(numImages != numLabels){
            throw std::runtime_error("[MNISTLoader]: Number of images (" + std::to_string(numImages) +
                ") does not match number of labels (" + std::to_string(numLabels) + ")");
		}

        labels.resize(numLabels);
        if (!fLabels.read(reinterpret_cast<char*>(labels.data()), numLabels)) {
            fLabels.close();
            throw std::runtime_error("[MNISTLoader]: Failed to read labels data from file: " + labelsFile.string());
        }

        fLabels.close();

        if(_splitValidation && isTrain)
			_splitTrainValidation(_validationRatio);

        return true;
    }

    void _splitTrainValidation(const double ratio) {
        if (_trainImages.empty() || _trainLabels.empty()) {
            throw std::runtime_error("[MNISTLoader]: Train data not loaded. Call loadTrainData() before splitting.");
        }
        if (ratio <= 0.0 || ratio >= 1.0) {
            throw std::invalid_argument("[MNISTLoader]: Validation ratio must be between 0 and 1 (exclusive).");
        }
        std::cout << "Performing train-validation split." << std::endl;


        size_t total = _trainImages.size();
        size_t splitIndex = static_cast<size_t>(total * (1.0 - ratio));

        _numValidation = total - splitIndex;
        _numTrain = splitIndex;

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
        oneHotLabels.reserve(labels.size());

        for (const auto& label : labels) {
            if (label >= _numClasses) {
                throw std::out_of_range("[MNISTLoader]: Label exceeds number of classes.");
            }

            Eigen::VectorXd oneHot = Eigen::VectorXd::Zero(_numClasses);
            oneHot(static_cast<int>(label)) = 1.0;
            oneHotLabels.emplace_back(std::move(oneHot));
        }
    }

};
