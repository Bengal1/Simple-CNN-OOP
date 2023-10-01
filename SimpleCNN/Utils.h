#pragma once

#include <vector>
#include <Eigen/Dense>


double accuracyCalculation(std::vector<Eigen::VectorXd>& modelOutput,
	const std::vector<Eigen::VectorXd>& oneHotTargets) {
	double correct = 0, dataSize = modelOutput.size();
	Eigen::Index maxIndex, correctIndex;

	for (int i = 0; i < dataSize; i++) {
		modelOutput[i].maxCoeff(&maxIndex);
		oneHotTargets[i].maxCoeff(&correctIndex);
		if (maxIndex == correctIndex)
			correct++;
	}

	return correct / dataSize * 100;
}

void exportTrainingData(void) {}

void storeModelData(void) {}
