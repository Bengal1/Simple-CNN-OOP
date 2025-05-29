#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
//#include "matplotlibcpp.h"


double accuracyCalculation(const std::vector<Eigen::VectorXd>& modelOutput,
						   const std::vector<Eigen::VectorXd>& oneHotTargets) 
{	
	if (modelOutput.size() != oneHotTargets.size()) {
		throw std::invalid_argument("Model output and targets must have the same size.");
	}
	
	double correctPredictions = 0; 
	size_t dataSize = modelOutput.size();
	Eigen::Index predictedClass, trueClass;

	for (size_t i = 0; i < dataSize; ++i) {
		modelOutput[i].maxCoeff(&predictedClass);
		oneHotTargets[i].maxCoeff(&trueClass);
		if (predictedClass == trueClass) {
			++correctPredictions;
		}
	}

	return static_cast<double>(correctPredictions) / static_cast<double>(dataSize) * 100.0;
}

/*void saveModelState(SimpleCNN model) 
{
	std::ofstream file("model_state.txt");
	if (file.is_open()) {
		file << model.getModelState();
		file.close();
	}
	else {
		std::cerr << "Unable to open file for writing." << std::endl;
	}
}

void loadModelState(void) 
{
	std::ifstream file("model_state.txt");
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			std::cout << line << std::endl;
		}
		file.close();
	}
	else {
		std::cerr << "Unable to open file for reading." << std::endl;
	}
}*/

