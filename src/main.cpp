#include "../include/SimpleCNN.hpp"

int main()
{
    size_t epochs = 10, classes = 10;
    double validationRatio = 0.2;
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
        loader.loadTrainData();
        loader.loadTestData();

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