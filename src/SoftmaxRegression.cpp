// src/SoftmaxRegression.cpp

#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include "SoftmaxRegression.h"
using namespace std;

using MatrixFloat = Eigen::MatrixXf;
using VectorInt = Eigen::VectorXi;

// Softmax regression constructor
SoftmaxRegression::SoftmaxRegression(int numFeatures, int numClasses) {
    // Initialize weights with small random values
    weights = MatrixFloat::Random(numFeatures, numClasses) * 0.01f;
    bias = Eigen::VectorXf::Zero(numClasses);
}

// Forward pass: compute predictions
MatrixFloat SoftmaxRegression::predict(const MatrixFloat &images) {
    // z = X * W + b
    MatrixFloat logits = images * weights;
    logits.rowwise() += bias.transpose();
    return softmax(logits);
}

// Train the model using gradient descent
void SoftmaxRegression::train(const MatrixFloat &trainImages, const MatrixFloat &trainLabelsOneHot, int numEpochs, float learningRate, int batchSize) {
    int numSamples = trainImages.rows();
    int numBatches = (numSamples + batchSize - 1) / batchSize;

    cout << "Training softmax regression...\n";
    cout << "Epochs: " << numEpochs << ", Learning rate: " << learningRate << ", Batch size: " << batchSize << "\n\n";

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

            /***** FORWARD PASS *****/ 
            // Make predictions based on current weights and bias
            MatrixFloat predictions = predict(batchImages);

            // Compute average loss for this batch
            // "How far off was I this time?"
            float loss = crossEntropyLoss(predictions, batchLabels);

            // totalLoss measures loss for each epoch. Need to multiply
            // average loss by currentBatchSize to scale it up from per-batch
            // average to eventually get total loss for epoch.
            totalLoss += loss * currentBatchSize;
            // Additional notes: Each batch gives us the average loss per-batch,
            // but we need the average across per-epoch. To do this, we
            // accumulate the average loss scaled by the number of samples over
            // each batch in the epoch.

            /***** BACKWARD PASS: Compute gradients *****/ 
            // Gradient of cross-entropy + softmax: predictions - targets
            // Gets mean gradient for the batch
            // The gradient logits are needed to update the weights and bias on the next step
            MatrixFloat gradientLogits = (predictions - batchLabels) / currentBatchSize;

            // Here's the equation for calculating logits: z = XW + b.
            // Take the derivative wrt W to know how to update W (weights).
            // Gradient for weights: X^T * gradientLogits
            // “How much does changing pixel i affect class j’s error?”
            MatrixFloat gradientWeights = batchImages.transpose() * gradientLogits;

            // Gradient for bias: sum over samples
            // "For each column (class), sum all values in that column"
            Eigen::VectorXf gradientBias = gradientLogits.colwise().sum();

            //***** UPDATE PARAMETERS *****/
            // The goal is to minimize loss
            // SUBTRACTING moves us in the direction of LOWER loss
            weights -= learningRate * gradientWeights;
            bias -= learningRate * gradientBias;
        }

        // Timekeeping for reporting performance
        auto epochEndTime = chrono::high_resolution_clock::now();
        chrono::duration<double> epochDuration = epochEndTime - epochStartTime;
        double epochTime = epochDuration.count();
        totalEpochTime += epochTime;

        // Report progress each epoch
        float avgLoss = totalLoss / numSamples;
        MatrixFloat trainPredictions = predict(trainImages);
        float trainAccuracy = computeAccuracy(trainPredictions, Eigen::Map<const VectorInt>(reinterpret_cast<const int *>(trainLabelsOneHot.data()), trainLabelsOneHot.rows()));

        // Get actual labels from one-hot encoding for accuracy calculation
        VectorInt actualLabels(trainLabelsOneHot.rows());
        for (int i = 0; i < trainLabelsOneHot.rows(); i++) {
            trainLabelsOneHot.row(i).maxCoeff(&actualLabels(i));
        }
        
        // TODO: Set aside 20% of training data for validation accuracy
        // to avoid reporting training accuracy here (which can 
        // inaccurately represent the model's accuracy).
        trainAccuracy = computeAccuracy(trainPredictions, actualLabels);

        cout << "Epoch " << (epoch + 1) << "/" << numEpochs
             << " - Loss: " << avgLoss
             << " - Accuracy: " << (trainAccuracy * 100) << "%"
             << " - Time: " << fixed << setprecision(1) << epochTime << "s\n";
    }

    auto trainingEndTime = chrono::high_resolution_clock::now();
    chrono::duration<double> trainingDuration = trainingEndTime - trainingStartTime;
    double totalTrainingTime = trainingDuration.count();
    double avgEpochTime = totalEpochTime / numEpochs;

    cout << "\nTraining complete!\n";
    cout << "Total training time: " << fixed << setprecision(1) << totalTrainingTime << "s | ";
    cout << "Avg. time per epoch: " << fixed << setprecision(1) << avgEpochTime << "s\n\n";
}

// Softmax function: converts logits to probabilities
// Applied row-wise: each row is a sample, each column is a class
MatrixFloat softmax(const MatrixFloat &logits) {
    MatrixFloat probabilities(logits.rows(), logits.cols());

    for (int i = 0; i < logits.rows(); i++) {
        // Subtract max for numerical stability
        float maxLogit = logits.row(i).maxCoeff();
        Eigen::VectorXf expValues = (logits.row(i).array() - maxLogit).exp();
        float sumExp = expValues.sum();
        probabilities.row(i) = expValues / sumExp;
    }

    return probabilities;
}

// Compute cross-entropy loss
float crossEntropyLoss(const MatrixFloat &predictions, const MatrixFloat &targets) {
    // Add small epsilon to avoid log(0)
    const float epsilon = 1e-10f;
    MatrixFloat clippedPredictions = predictions.array().max(epsilon);

    // Cross-entropy: -sum(y_true * log(y_pred)) / numSamples
    float loss = -(targets.array() * clippedPredictions.array().log()).sum() / predictions.rows();
    return loss;
}

// Compute accuracy
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
