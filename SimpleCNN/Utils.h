#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>


double accuracyCalculation(std::vector<Eigen::VectorXd>& modelOutput,
	const std::vector<Eigen::VectorXd>& oneHotTargets) {
	
	if (modelOutput.size() != oneHotTargets.size()) {
		std::cerr << "Error: Input vectors have different sizes." << std::endl;
		return 0.0;
	}
	
	double correctPredictions = 0; 
	int dataSize = modelOutput.size();
	Eigen::Index predictedClass, trueClass;

	for (int i = 0; i < dataSize; i++) {
		modelOutput[i].maxCoeff(&predictedClass);
		oneHotTargets[i].maxCoeff(&trueClass);
		if (predictedClass == trueClass)
			correctPredictions++;
		/*else { //TESTING
			std::cout << modelOutput[i] << std::endl;
			std::cout << oneHotTargets[i] <<  std::endl;
			std::cout << "trueClass : " << trueClass << " ; predictedClass : " << predictedClass << std::endl << std::endl;
		} //TESTING*/
	}

	return correctPredictions / static_cast<double>(dataSize) * 100.0;
}

void exportTrainingData(void) {}

void storeModelData(void) {}
