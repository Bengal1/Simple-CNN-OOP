/**
 * @file main.cpp
 * @brief Demonstrates the functionality of the SimpleCNN model.
 *
 * This program serves as the main entry point to train and test a SimpleCNN
 * model on the MNIST dataset. It handles data loading, model training,
 * evaluation on a test set, and exports the training history to a CSV file.
 * The program is designed to showcase a complete training and testing pipeline
 * for an object-oriented CNN implementation.
 *
 * @author Bengal1
 * @version 1.0
 * @date May 2025
 */
#include "../include/SimpleCNN.hpp"

int main()
{
    const size_t epochs = 10, classes = 10;
    const double validationRatio = 0.2;
    // MNIST dataset paths
    std::filesystem::path trainImages = "MNIST/train-images.idx3-ubyte";
    std::filesystem::path trainLabels = "MNIST/train-labels.idx1-ubyte";
    std::filesystem::path testImages = "MNIST/t10k-images.idx3-ubyte";
    std::filesystem::path testLabels = "MNIST/t10k-labels.idx1-ubyte";

    // Create a SimpleCNN model
    SimpleCNN model(classes);

    try
    {
        // Load MNIST dataset
        MNISTLoader loader(trainImages, trainLabels, testImages, testLabels, validationRatio);

        // Train the model
        model.trainSimpleCNN(loader, epochs);

        // Test the model
        model.testSimpleCNN(loader);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    model.exportTrainingDataToCSV();

    return 0;
}