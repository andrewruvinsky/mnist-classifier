#ifndef SOFTMAX_REGRESSION_H
#define SOFTMAX_REGRESSION_H
#include <Eigen/Dense>

using MatrixFloat = Eigen::MatrixXf;
using VectorInt = Eigen::VectorXi;

// Helper functions
MatrixFloat softmax(const MatrixFloat &logits);
float crossEntropyLoss(const MatrixFloat &predictions, const MatrixFloat &targets);
float computeAccuracy(const MatrixFloat &predictions, const VectorInt &trueLabels);

class SoftmaxRegression {
public:
    MatrixFloat weights;  // numFeatures x numClasses
    Eigen::VectorXf bias; // numClasses
    
    SoftmaxRegression(int numFeatures, int numClasses);
    MatrixFloat predict(const MatrixFloat &images);
    void train(const MatrixFloat &trainImages, const MatrixFloat &trainLabelsOneHot, int numEpochs, float learningRate, int batchSize);
};

#endif