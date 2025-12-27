/**
 * @file MNISTLoader.hpp
 * @brief Declaration of the MNISTLoader class.
 *
 * This header defines the MNISTLoader class, which provides functionality
 * for loading, parsing, and preprocessing the MNIST dataset from its
 * standard binary file format. The loader supports optional splitting
 * of the training set into training and validation subsets and exposes
 * the data in Eigen-friendly structures suitable for numerical processing.
 */

#pragma once

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

/**
 * @class MNISTLoader
 * @brief Utility class for loading and preprocessing the MNIST dataset.
 *
 * MNISTLoader is responsible for:
 *  - Reading MNIST image and label files in IDX binary format
 *  - Converting image data into Eigen matrices
 *  - Managing train, validation, and test splits
 *  - Converting class labels into one-hot encoded vectors
 *
 * The class is designed to be used as a data provider for machine learning
 * models and abstracts away all dataset-specific I/O and preprocessing logic.
 */
class MNISTLoader
{
   private:
    /** @brief Default number of MNIST classes (digits 0–9). */
    static constexpr int MNISTClasses = 10;

    /** @brief Default validation split ratio (no validation split). */
    static constexpr double DefaultValRatio = 0.0;
    
    // ------------------------------------------------------------------
    // Dataset file paths
    // ------------------------------------------------------------------

    /** @brief Path to the training images file. */
    const std::filesystem::path _trainImagesFile;

    /** @brief Path to the training labels file. */
    const std::filesystem::path _trainLabelsFile;

    /** @brief Path to the test images file. */
    const std::filesystem::path _testImagesFile;

    /** @brief Path to the test labels file. */
    const std::filesystem::path _testLabelsFile;

    // ------------------------------------------------------------------
    // Dataset storage
    // ------------------------------------------------------------------

    /** @brief Training images stored as 2D Eigen matrices. */
    std::vector<Eigen::MatrixXd> _trainImages;

    /** @brief Validation images stored as 2D Eigen matrices. */
    std::vector<Eigen::MatrixXd> _validationImages;

    /** @brief Test images stored as 2D Eigen matrices. */
    std::vector<Eigen::MatrixXd> _testImages;

    /** @brief Raw training labels (class indices). */
    std::vector<uint8_t> _trainLabels;

    /** @brief Raw validation labels (class indices). */
    std::vector<uint8_t> _validationLabels;

    /** @brief Raw test labels (class indices). */
    std::vector<uint8_t> _testLabels;

    /** @brief One-hot encoded training labels. */
    std::vector<Eigen::VectorXd> _oneHotTrainLabels;

    /** @brief One-hot encoded validation labels. */
    std::vector<Eigen::VectorXd> _oneHotValidationLabels;

    /** @brief One-hot encoded test labels. */
    std::vector<Eigen::VectorXd> _oneHotTestLabels;

    // ------------------------------------------------------------------
    // Dataset metadata
    // ------------------------------------------------------------------

    /** @brief Number of training samples. */
    size_t _numTrain = 0;

    /** @brief Number of validation samples. */
    size_t _numValidation = 0;

    /** @brief Number of test samples. */
    size_t _numTest = 0;

    /** @brief Number of classification classes. */
    size_t _numClasses = MNISTClasses;

    /** @brief Indicates whether a validation split is enabled. */
    bool _splitValidation = false;

    /** @brief Ratio of training data reserved for validation. */
    double _validationRatio;

   public:
    /**
     * @brief Constructs an MNISTLoader and loads the dataset.
     *
     * Loads the MNIST training and test datasets from the specified file paths.
     * Optionally splits a portion of the training data into a validation set.
     *
     * @param trainImagesFile Path to the training images IDX file.
     * @param trainLabelsFile Path to the training labels IDX file.
     * @param testImagesFile  Path to the test images IDX file.
     * @param testLabelsFile  Path to the test labels IDX file.
     * @param validationRatio Fraction of training data used for validation
     *                        (range: [0.0, 1.0]).
     */
    MNISTLoader(const std::filesystem::path& trainImagesFile,
                const std::filesystem::path& trainLabelsFile,
                const std::filesystem::path& testImagesFile,
                const std::filesystem::path& testLabelsFile,
                double validationRatio = DefaultValRatio);
    
    /** @brief Default destructor. */
    ~MNISTLoader() = default;

    // ------------------------------------------------------------------
    // Dataset accessors
    // ------------------------------------------------------------------

    /** @return Constant reference to the training images. */
    const std::vector<Eigen::MatrixXd>& getTrainImages() const;

    /** @return Constant reference to the validation images. */
    const std::vector<Eigen::MatrixXd>& getValidationImages() const;

    /** @return Constant reference to the test images. */
    const std::vector<Eigen::MatrixXd>& getTestImages() const;

    /** @return One-hot encoded training labels. */
    const std::vector<Eigen::VectorXd>& getOneHotTrainLabels();

    /** @return One-hot encoded validation labels. */
    const std::vector<Eigen::VectorXd>& getOneHotValidationLabels();

    /** @return One-hot encoded test labels. */
    const std::vector<Eigen::VectorXd>& getOneHotTestLabels();

    /** @return Number of training samples. */
    const size_t getNumTrain() const;

    /** @return Number of validation samples. */
    const size_t getNumValidation() const;

    /** @return Number of test samples. */
    const size_t getNumTest() const;

   private:
    // ------------------------------------------------------------------
    // Internal loading and preprocessing helpers
    // ------------------------------------------------------------------

    /**
     * @brief Loads the training dataset from disk.
     * @return True if loading succeeded, false otherwise.
     */
    bool _loadTrainData();

    /**
     * @brief Loads the test dataset from disk.
     * @return True if loading succeeded, false otherwise.
     */
    bool _loadTestData();

    /**
     * @brief Loads MNIST images and labels from IDX files.
     *
     * @param imagesFile Path to the images IDX file.
     * @param labelsFile Path to the labels IDX file.
     * @param images     Output container for loaded images.
     * @param labels     Output container for loaded labels.
     * @param isTrain    Indicates whether the data belongs to the training set.
     *
     * @return True if loading succeeded, false otherwise.
     */
    bool _loadImages(const std::filesystem::path& imagesFile,
                     const std::filesystem::path& labelsFile,
                     std::vector<Eigen::MatrixXd>& images,
                     std::vector<uint8_t>& labels,
                     bool isTrain);

    /**
     * @brief Splits the training data into training and validation sets.
     *
     * @param ratio Fraction of training data assigned to validation.
     */
    void _splitTrainValidation(double ratio);

    /**
     * @brief Converts class labels to one-hot encoded vectors.
     *
     * @param labels         Input vector of class indices.
     * @param oneHotLabels   Output vector of one-hot encoded labels.
     */
    void _createOneHotLabels(const std::vector<uint8_t>& labels,
                             std::vector<Eigen::VectorXd>& oneHotLabels);
};
