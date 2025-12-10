// src/SimpleANN.h

#ifndef SIMPLE_ANN_H
#define SIMPLE_ANN_H
#include <Eigen/Dense>

using MatrixFloat = Eigen::MatrixXf;
using VectorInt = Eigen::VectorXi;

// Helper functions
MatrixFloat softmax(const MatrixFloat &logits);
float crossEntropyLoss(const MatrixFloat &predictions, const MatrixFloat &targets);
float computeAccuracy(const MatrixFloat &predictions, const VectorInt &trueLabels);
Eigen::MatrixXi computeConfusionMatrix(const MatrixFloat &predictions, const VectorInt &trueLabels, int numClasses);
void printConfusionMatrix(const Eigen::MatrixXi &confusionMatrix);
void printPerformanceMetrics(const Eigen::MatrixXi &confusionMatrix);

class SimpleANN {
public:
    MatrixFloat weightsInputToHidden;
    Eigen::VectorXf biasHidden;
    MatrixFloat weightsHiddenToOutput;
    Eigen::VectorXf biasOutput;

    SimpleANN(int numFeatures, int numHiddenUnits, int numClasses);
    MatrixFloat predict(const MatrixFloat& images);
    void train(const MatrixFloat& trainImages, const MatrixFloat& trainLabelsOneHot, int numEpochs, float learningRate, int batchSize);
};

#endif