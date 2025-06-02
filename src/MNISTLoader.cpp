#include "../include/MNISTLoader.hpp"

MNISTLoader::MNISTLoader(const std::filesystem::path& trainImagesFile,
                         const std::filesystem::path& trainLabelsFile,
                         const std::filesystem::path& testImagesFile,
                         const std::filesystem::path& testLabelsFile, double validationRatio)
    : _trainImagesFile(trainImagesFile),
      _trainLabelsFile(trainLabelsFile),
      _testImagesFile(testImagesFile),
      _testLabelsFile(testLabelsFile),
      _validationRatio(validationRatio)
{
    if (_trainImagesFile.empty() || _trainLabelsFile.empty() || _testImagesFile.empty() ||
        _testLabelsFile.empty())
    {
        throw std::invalid_argument("[MNISTLoader]: File paths cannot be empty.");
    }

    for (const auto& path : {_trainImagesFile, _trainLabelsFile, _testImagesFile, _testLabelsFile})
    {
        if (!std::filesystem::exists(path))
        {
            throw std::runtime_error("[MNISTLoader]: File not found: " + path.string());
        }
        if (!std::filesystem::is_regular_file(path))
        {
            throw std::runtime_error("[MNISTLoader]: Not a regular file: " + path.string());
        }
    }

    if (_validationRatio < 0.0 || _validationRatio >= 1.0)
    {
        throw std::invalid_argument("[MNISTLoader]: Validation ratio must be in the range [0, 1).");
    }

    _splitValidation = (_validationRatio > 0.0);
}

bool MNISTLoader::loadTrainData()
{
    return _loadImages(_trainImagesFile, _trainLabelsFile, _trainImages, _trainLabels, true);
}

bool MNISTLoader::loadTestData()
{
    return _loadImages(_testImagesFile, _testLabelsFile, _testImages, _testLabels, false);
}

const std::vector<Eigen::MatrixXd>& MNISTLoader::getTrainImages() const
{
    return _trainImages;
}
const std::vector<Eigen::MatrixXd>& MNISTLoader::getValidationImages() const
{
    return _validationImages;
}
const std::vector<Eigen::MatrixXd>& MNISTLoader::getTestImages() const
{
    return _testImages;
}

const std::vector<Eigen::VectorXd>& MNISTLoader::getOneHotTrainLabels()
{
    if (_oneHotTrainLabels.empty()) _createOneHotLabels(_trainLabels, _oneHotTrainLabels);
    return _oneHotTrainLabels;
}

const std::vector<Eigen::VectorXd>& MNISTLoader::getOneHotValidationLabels()
{
    if (_oneHotValidationLabels.empty())
        _createOneHotLabels(_validationLabels, _oneHotValidationLabels);
    return _oneHotValidationLabels;
}

const std::vector<Eigen::VectorXd>& MNISTLoader::getOneHotTestLabels()
{
    if (_oneHotTestLabels.empty()) _createOneHotLabels(_testLabels, _oneHotTestLabels);
    return _oneHotTestLabels;
}

const size_t MNISTLoader::getNumTrain() const
{
    return _numTrain;
}
const size_t MNISTLoader::getNumValidation() const
{
    return _numValidation;
}
const size_t MNISTLoader::getNumTest() const
{
    return _numTest;
}

bool MNISTLoader::_loadImages(const std::filesystem::path& imagesFile,
                              const std::filesystem::path& labelsFile,
                              std::vector<Eigen::MatrixXd>& images, std::vector<uint8_t>& labels,
                              bool isTrain)
{
    std::ifstream fImages(imagesFile, std::ios::binary);
    if (!fImages)
        throw std::runtime_error("[MNISTLoader]: Failed to open images file: " +
                                 imagesFile.string());

    std::cout << "Reading " << (isTrain ? "Train" : "Test") << " Data From Source." << std::endl;

    uint32_t magicNumber, numImages, numRows, numCols;
    fImages.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    fImages.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    fImages.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    fImages.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

    magicNumber = _byteswap_ulong(magicNumber);
    numImages = _byteswap_ulong(numImages);
    numRows = _byteswap_ulong(numRows);
    numCols = _byteswap_ulong(numCols);

    if (magicNumber != 0x803)
        throw std::runtime_error("[MNISTLoader]: Invalid magic number in images file");

    if (isTrain)
        _numTrain = numImages;
    else
        _numTest = numImages;

    images.resize(numImages);
    for (size_t i = 0; i < numImages; ++i)
    {
        Eigen::MatrixXd imageMatrix(numRows, numCols);
        for (size_t r = 0; r < numRows; ++r)
        {
            for (size_t c = 0; c < numCols; ++c)
            {
                uint8_t pixelValue;
                if (!fImages.read(reinterpret_cast<char*>(&pixelValue), sizeof(pixelValue)))
                    throw std::runtime_error(
                        "[MNISTLoader]: Unexpected end of file while reading image data in " +
                        imagesFile.string());
                imageMatrix(r, c) = static_cast<float>(pixelValue) / 255.0f;
            }
        }
        images[i] = imageMatrix;
    }

    fImages.close();

    std::ifstream fLabels(labelsFile, std::ios::binary);
    if (!fLabels)
        throw std::runtime_error("[MNISTLoader]: Failed to open labels file: " +
                                 labelsFile.string());

    uint32_t labelsMagicNumber, numLabels;
    fLabels.read(reinterpret_cast<char*>(&labelsMagicNumber), sizeof(labelsMagicNumber));
    fLabels.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

    labelsMagicNumber = _byteswap_ulong(labelsMagicNumber);
    numLabels = _byteswap_ulong(numLabels);

    if (labelsMagicNumber != 0x801)
        throw std::runtime_error("[MNISTLoader]: Invalid magic number in labels file");

    if (numImages != numLabels)
        throw std::runtime_error("[MNISTLoader]: Number of images does not match number of labels");

    labels.resize(numLabels);
    if (!fLabels.read(reinterpret_cast<char*>(labels.data()), numLabels))
        throw std::runtime_error("[MNISTLoader]: Failed to read labels data from file: " +
                                 labelsFile.string());

    fLabels.close();

    if (_splitValidation && isTrain) _splitTrainValidation(_validationRatio);

    return true;
}

void MNISTLoader::_splitTrainValidation(const double ratio)
{
    if (_trainImages.empty() || _trainLabels.empty())
    {
        throw std::runtime_error(
            "[MNISTLoader]: Train data not loaded. Call loadTrainData() before splitting.");
    }
    if (ratio <= 0.0 || ratio >= 1.0)
    {
        throw std::invalid_argument(
            "[MNISTLoader]: Validation ratio must be between 0 and 1 (exclusive).");
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

void MNISTLoader::_createOneHotLabels(const std::vector<uint8_t>& labels,
                                      std::vector<Eigen::VectorXd>& oneHotLabels)
{
    oneHotLabels.clear();
    oneHotLabels.reserve(labels.size());

    for (const auto& label : labels)
    {
        if (label >= _numClasses)
            throw std::out_of_range("[MNISTLoader]: Label exceeds number of classes.");

        Eigen::VectorXd oneHot = Eigen::VectorXd::Zero(_numClasses);
        oneHot(static_cast<int>(label)) = 1.0;
        oneHotLabels.emplace_back(std::move(oneHot));
    }
}
