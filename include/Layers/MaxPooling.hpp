#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <vector>

class MaxPooling
{
   private:
    using Index3D = std::tuple<size_t, size_t, size_t>;
    using Index4D = std::tuple<size_t, size_t, size_t, size_t>;

    const size_t _inputHeight;
    const size_t _inputWidth;
    const size_t _inputChannels;

    const size_t _kernelSize;
    const size_t _stride;
    const size_t _batchSize;

    size_t _outputHeight;
    size_t _outputWidth;

    std::vector<Eigen::MatrixXd> _output;
    std::vector<std::vector<Eigen::MatrixXd>> _outputBatch;

    std::vector<Index3D> _inputGradientMap;
    std::vector<Index4D> _inputGradientMapBatch;

   public:
    MaxPooling(size_t inputHeight, size_t inputWidth, size_t inputChannels, size_t poolSize,
               size_t batchSize = 1, size_t stride = 2);

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input);
    std::vector<std::vector<Eigen::MatrixXd>> forwardBatch(
        const std::vector<std::vector<Eigen::MatrixXd>>& inputBatch);

    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& gradient);
    std::vector<std::vector<Eigen::MatrixXd>> backwardBatch(
        const std::vector<std::vector<Eigen::MatrixXd>>& gradientBatch);

   private:
    void _initialize();
    void _validate() const;

    void _getDataLocation(size_t& dataChannel, size_t& dataRow, size_t& dataColumn,
                          size_t& mapIndex);

    void _getDataLocation(size_t& dataBatch, size_t& dataChannel, size_t& dataRow,
                          size_t& dataColumn, size_t& mapIndex);
};
