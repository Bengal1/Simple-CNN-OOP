/**
 * @file MNISTLoader.hpp
 * @brief Header file for the MNISTLoader class.
 *
 * This header defines the MNISTLoader class, a utility for loading the
 * MNIST dataset from its binary files, including handling train/test splits
 * and optional validation set creation.
 */
#pragma once

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

/**
 * @class MNISTLoader
 * @brief A utility class for loading and preprocessing the MNIST dataset.
 *
 * The MNISTLoader class handles the loading of image and label data from
 * the standard MNIST dataset files. It provides functionality to split a
 * portion of the training data into a validation set and to convert labels
 * to a one-hot encoded format.
 */
class MNISTLoader
{
   private:
    // Default parameters
    static constexpr int MNISTClasses = 10;
    static constexpr double DefaultValRatio = 0.0;
    // MNIST file paths
    const std::filesystem::path _trainImagesFile;
    const std::filesystem::path _trainLabelsFile;
    const std::filesystem::path _testImagesFile;
    const std::filesystem::path _testLabelsFile;

    // MNIST data
    std::vector<Eigen::MatrixXd> _trainImages, _validationImages, _testImages;
    std::vector<uint8_t> _trainLabels, _validationLabels, _testLabels;
    std::vector<Eigen::VectorXd> _oneHotTrainLabels, _oneHotValidationLabels, _oneHotTestLabels;

    // Metadata
    size_t _numTrain = 0, _numValidation = 0, _numTest = 0, _numClasses = MNISTClasses;
    bool _splitValidation = false;
    double _validationRatio;

   public:
    MNISTLoader(const std::filesystem::path& trainImagesFile,
                const std::filesystem::path& trainLabelsFile,
                const std::filesystem::path& testImagesFile,
                const std::filesystem::path& testLabelsFile,
                double validationRatio = DefaultValRatio);

    ~MNISTLoader() = default;

    const std::vector<Eigen::MatrixXd>& getTrainImages() const;
    const std::vector<Eigen::MatrixXd>& getValidationImages() const;
    const std::vector<Eigen::MatrixXd>& getTestImages() const;

    const std::vector<Eigen::VectorXd>& getOneHotTrainLabels();
    const std::vector<Eigen::VectorXd>& getOneHotValidationLabels();
    const std::vector<Eigen::VectorXd>& getOneHotTestLabels();

    const size_t getNumTrain() const;
    const size_t getNumValidation() const;
    const size_t getNumTest() const;

   private:
    bool _loadTrainData();
    bool _loadTestData();
    bool _loadImages(const std::filesystem::path& imagesFile,
                     const std::filesystem::path& labelsFile, std::vector<Eigen::MatrixXd>& images,
                     std::vector<uint8_t>& labels, bool isTrain);

    void _splitTrainValidation(double ratio);

    void _createOneHotLabels(const std::vector<uint8_t>& labels,
                             std::vector<Eigen::VectorXd>& oneHotLabels);
};
