// src/SimpleANN.cpp

#include "SimpleANN.h"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <iomanip>
using namespace std;

using MatrixFloat = Eigen::MatrixXf;
using VectorInt = Eigen::VectorXi;

/*********** Helper Functions ***********/
// Softmax function: converts logits to probabilities
// Applied row-wise: each row is a sample, each column is a class
MatrixFloat softmax(const MatrixFloat &logits)
{
    MatrixFloat probabilities(logits.rows(), logits.cols());

    for (int i = 0; i < logits.rows(); i++)
    {
        // Subtract max for numerical stability
        float maxLogit = logits.row(i).maxCoeff();
        Eigen::VectorXf expValues = (logits.row(i).array() - maxLogit).exp();
        float sumExp = expValues.sum();
        probabilities.row(i) = expValues / sumExp;
    }

    return probabilities;
}

float crossEntropyLoss(const MatrixFloat &predictions, const MatrixFloat &targets) {
    // Add small epsilon to avoid log(0)
    const float epsilon = 1e-10f;
    MatrixFloat clippedPredictions = predictions.array().max(epsilon);

    // Cross-entropy: -sum(y_true * log(y_pred)) / numSamples
    float loss = -(targets.array() * clippedPredictions.array().log()).sum() / predictions.rows();
    return loss;
}

float computeAccuracy(const MatrixFloat &predictions, const VectorInt &trueLabels) {
    int correct = 0;
    for (int i = 0; i < predictions.rows(); i++) {
        int predictedClass;
        predictions.row(i).maxCoeff(&predictedClass);
        if (predictedClass == trueLabels(i))
            correct++;
    }

    return static_cast<float>(correct) / predictions.rows();
}

/*********** SimpleANN Member Functions ***********/

SimpleANN::SimpleANN(int numFeatures, int numHiddenUnits, int numClasses)
{
    // Initialize weights and biases with small random values
    weightsInputToHidden = MatrixFloat::Random(numFeatures, numHiddenUnits) * 0.01f;
    biasHidden = Eigen::VectorXf::Zero(numHiddenUnits);
    weightsHiddenToOutput = MatrixFloat::Random(numHiddenUnits, numClasses) * 0.01f;
    biasOutput = Eigen::VectorXf::Zero(numClasses);
}

// Returns predicted probabilities for each class
MatrixFloat SimpleANN::predict(const MatrixFloat &images)
{
    /***** Forward pass *****/
    // Input -> Hidden
    // Z1 = X * W1 + b1
    MatrixFloat hiddenPreActivation = (images * weightsInputToHidden).rowwise() + biasHidden.transpose();

    // Apply ReLU activation function
    // A1 = max(0, Z1)
    MatrixFloat hiddenActivation = hiddenPreActivation.cwiseMax(0.0f);

    // Output layer
    // Z2 = A1 * W2 + b2
    MatrixFloat logits = (hiddenActivation * weightsHiddenToOutput).rowwise() + biasOutput.transpose();

    // Apply softmax to get probabilities for each output class
    // P = softmax(Z2)
    MatrixFloat probabilities = softmax(logits);

    return probabilities;
}

void SimpleANN::train(const MatrixFloat &trainImages, const MatrixFloat &trainLabelsOneHot, int numEpochs, float learningRate, int batchSize) {
    int numSamples = trainImages.rows();
    int numBatches = (numSamples + batchSize - 1) / batchSize;

    auto trainingStartTime = chrono::high_resolution_clock::now();
    double totalEpochTime = 0.0;

    for (int epoch = 0; epoch < numEpochs; epoch++) {
        auto epochStartTime = chrono::high_resolution_clock::now();
        float totalLoss = 0.0f;

        for (int batch = 0; batch < numBatches; batch++) {
            // Grabs the first batchSize samples, then the next
            // batchSize on the next iterations, and so on.
            int startIdx = batch * batchSize;
            int endIdx = min(startIdx + batchSize, numSamples);
            int currentBatchSize = endIdx - startIdx;

            // Get batch
            MatrixFloat batchImages = trainImages.middleRows(startIdx, currentBatchSize);
            // One-hot encod labels for the batch
            MatrixFloat batchLabels = trainLabelsOneHot.middleRows(startIdx, currentBatchSize);

            /***** FORWARD PASS: Make predictions based on current weights and bias *****/
            // 1) Input -> Hidden
            MatrixFloat hiddenPreActivation = (batchImages * weightsInputToHidden).rowwise() + biasHidden.transpose();
            MatrixFloat hiddenActivation = hiddenPreActivation.cwiseMax(0.0f);

            // 2) Hidden -> Output
            MatrixFloat logits = (hiddenActivation * weightsHiddenToOutput).rowwise() + biasOutput.transpose();

            // 3) Apply softmax to get probabilities for each class
            MatrixFloat predictions = softmax(logits);

            float batchLoss = crossEntropyLoss(predictions, batchLabels);
            totalLoss += batchLoss * currentBatchSize;

            /***** BACKWARD PASS: Compute gradients *****/
            // Output layer gradients (W2, b2)
            MatrixFloat gradientLogits = (predictions - batchLabels) / currentBatchSize;
            MatrixFloat gradientWeightsHiddenToOutput = hiddenActivation.transpose() * gradientLogits;
            Eigen::VectorXf gradientBiasOutput = gradientLogits.colwise().sum();

            // Next backprop error into hidden layer through W2, ReLU to get gradients for W1, b1

            // dA1 = dZ2 * W2ᵀ
            MatrixFloat gradientHiddenActivation = gradientLogits * weightsHiddenToOutput.transpose();

            // ReLU derivative: 1 where Z1 > 0, else 0
            MatrixFloat reluMask = (hiddenPreActivation.array() > 0.0f).cast<float>();
            MatrixFloat gradientHiddenPreActivation = gradientHiddenActivation.array() * reluMask.array();

            // Input layer gradients (W1, b1)
            MatrixFloat gradientWeightsInputToHidden = batchImages.transpose() * gradientHiddenPreActivation;
            Eigen::VectorXf gradientBiasHidden = gradientHiddenPreActivation.colwise().sum();

            //***** UPDATE PARAMETERS *****/
            // ...to minimize loss
            weightsInputToHidden -= learningRate * gradientWeightsInputToHidden;
            biasHidden -= learningRate * gradientBiasHidden;

            weightsHiddenToOutput -= learningRate * gradientWeightsHiddenToOutput;
            biasOutput -= learningRate * gradientBiasOutput;
        }

        auto epochEndTime = chrono::high_resolution_clock::now();
        chrono::duration<double> epochDuration = epochEndTime - epochStartTime;
        double epochTime = epochDuration.count();
        totalEpochTime += epochTime;

        // Report progress each epoch
        float avgLoss = totalLoss / numSamples;

        MatrixFloat trainPredictions = predict(trainImages);
        
        // Get actual labels from one-hot encoding for accuracy calculation
        VectorInt actualLabels(trainLabelsOneHot.rows());
        for (int i = 0; i < trainLabelsOneHot.rows(); i++) {
            trainLabelsOneHot.row(i).maxCoeff(&actualLabels(i));
        }
        
        float trainAccuracy = computeAccuracy(trainPredictions, actualLabels);

        // TODO: Set aside 20% of training data for validation accuracy
        // to avoid reporting training accuracy here (which can 
        // inaccurately represent the model's accuracy).

        cout << "Epoch " << (epoch + 1) << "/" << numEpochs
             << " | Loss: " << avgLoss
             << " | Accuracy: " << (trainAccuracy * 100) << "%"
             << " | Time: " << fixed << setprecision(1) << epochTime << "s\n";
    }

    auto trainingEndTime = chrono::high_resolution_clock::now();
    chrono::duration<double> trainingDuration = trainingEndTime - trainingStartTime;
    double totalTrainingTime = trainingDuration.count();
    double avgEpochTime = totalEpochTime / numEpochs;

    cout << "\nTraining complete!\n";
    cout << "Total training time: " << fixed << setprecision(1) << totalTrainingTime << "s | ";
    cout << "Avg. time per epoch: " << fixed << setprecision(1) << avgEpochTime << "s\n\n";
}
