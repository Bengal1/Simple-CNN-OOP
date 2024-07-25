#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

class MNISTLoader {
private:
    std::string _trainImagesFile;
    std::string _trainLabelsFile;
    std::string _testImagesFile;
    std::string _testLabelsFile;

    std::vector<Eigen::MatrixXd> _trainImages;
    std::vector<uint8_t> _trainLabels;
    std::vector<Eigen::MatrixXd> _testImages;
    std::vector<uint8_t> _testLabels;
    
    std::vector<Eigen::VectorXd> _oneHotTrainLabels;
    std::vector<Eigen::VectorXd> _oneHotTestLabels;

public:
    int numTrain = 0;
    int numTest = 0;

    MNISTLoader(const std::string& trainImagesFile, const std::string& trainLabelsFile,
        const std::string& testImagesFile, const std::string& testLabelsFile) :
        _trainImagesFile(trainImagesFile), _trainLabelsFile(trainLabelsFile),
        _testImagesFile(testImagesFile), _testLabelsFile(testLabelsFile) {}

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

    const std::vector<Eigen::MatrixXd>& getTestImages() const {
        return _testImages;
    }

    const std::vector<Eigen::VectorXd>& getOneHotTrainLabels() {
        if (_oneHotTrainLabels.empty()) {
            _createOneHotLabels(_trainLabels, _oneHotTrainLabels);
        }
        return _oneHotTrainLabels;
    }

    const std::vector<Eigen::VectorXd>& getOneHotTestLabels() {
        if (_oneHotTestLabels.empty()) {
            _createOneHotLabels(_testLabels, _oneHotTestLabels);
        }
        return _oneHotTestLabels;
    }

private:

    bool _loadImages(const std::string& imagesFile, const std::string& labelsFile,
        std::vector<Eigen::MatrixXd>& images, std::vector<uint8_t>& labels, bool isTrain) {
        std::ifstream fImages(imagesFile, std::ios::binary);

        if (!fImages.is_open()) {
            std::cerr << "Failed to open images file: " << imagesFile << std::endl;
            return false;
        }
        if (isTrain)
            std::cout << "Reading Train Data From Source." << std::endl;
        else
            std::cout << "Reading Test Data From Source." << std::endl;

        uint32_t magicNumber = 0, numImages = 0, numRows = 0, numCols = 0;
        fImages.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        fImages.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        fImages.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
        fImages.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

        magicNumber = _byteswap_ulong(magicNumber);
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
                    uint8_t pixelValue = 0;
                    fImages.read(reinterpret_cast<char*>(&pixelValue), sizeof(pixelValue));
                    imageMatrix(r, c) = static_cast<float>(pixelValue) / 255.0;
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

        uint32_t labelsMagicNumber = 0, numLabels = 0;
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

        return true;
    }

    void _createOneHotLabels(const std::vector<uint8_t>& labels,
        std::vector<Eigen::VectorXd>& oneHotLabels) {
        oneHotLabels.clear();
        for (uint8_t label : labels) {
            Eigen::VectorXd oneHot = Eigen::VectorXd::Zero(10);
            oneHot(static_cast<int>(label)) = 1.0;
            oneHotLabels.push_back(oneHot);
        }
    }
};
