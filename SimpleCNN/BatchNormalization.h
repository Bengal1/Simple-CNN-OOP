#pragma once

#include <vector>
#include <Eigen/Dense>

class BatchNormaliztion {
private:
	const int _numChannels;

public:
	std::vector<Eigen::MatrixXd> forwardNormalization(std::vector<Eigen::MatrixXd> unNormalizedInput){}

	std::vector<Eigen::MatrixXd> backwardDerivative() {}

};

//double mean = matA.mean();
//double std_dev = std::sqrt((matA.array() - mean).square().sum() / matA.size());