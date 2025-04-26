#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>


double accuracyCalculation(const std::vector<Eigen::VectorXd>& modelOutput,
						   const std::vector<Eigen::VectorXd>& oneHotTargets) 
{	
	if (modelOutput.size() != oneHotTargets.size()) {
		std::cerr << "Error: Input vectors have different sizes." << std::endl;
		return 0.0;
	}
	
	double correctPredictions = 0; 
	int dataSize = modelOutput.size();
	Eigen::Index predictedClass, trueClass;

	for (size_t i = 0; i < dataSize; ++i) {
		modelOutput[i].maxCoeff(&predictedClass);
		oneHotTargets[i].maxCoeff(&trueClass);
		if (predictedClass == trueClass) {
			++correctPredictions;
		}
		/*else { // Debug!
			std::cout << modelOutput[i] << std::endl;
			std::cout << oneHotTargets[i] <<  std::endl;
			std::cout << "trueClass : " << trueClass << " ; predictedClass : " << predictedClass << std::endl << std::endl;
		} // Debug!*/
	}

	return static_cast<double>(correctPredictions) / static_cast<double>(dataSize) * 100.0;
}

void exportTrainingDataToCSV(void) {}
void saveModelState(void) {}
void loadModelState(void) {}

