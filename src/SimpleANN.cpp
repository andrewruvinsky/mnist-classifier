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

// Compute confusion matrix where:
// rows = true labels, columns = predicted labels
// confusionMatrix(i, j) = number of samples with true label i predicted as j
Eigen::MatrixXi computeConfusionMatrix(const MatrixFloat &predictions, const VectorInt &trueLabels, int numClasses) {
    Eigen::MatrixXi confusionMatrix = Eigen::MatrixXi::Zero(numClasses, numClasses);
    
    for (int i = 0; i < predictions.rows(); i++) {
        int predictedClass;
        predictions.row(i).maxCoeff(&predictedClass);
        int trueClass = trueLabels(i);
        confusionMatrix(trueClass, predictedClass)++;
    }
    
    return confusionMatrix;
}

void printConfusionMatrix(const Eigen::MatrixXi &confusionMatrix) {
    int numClasses = confusionMatrix.rows();
    
    cout << "\nConfusion Matrix:\n";
    cout << "                      Predicted\n";
    cout << "     ";
    for (int i = 0; i < numClasses; i++) {
        cout << setw(5) << i;
    }
    cout << "\n";
    cout << "    " << string(numClasses * 5 + 1, '-') << "\n";
    
    for (int i = 0; i < numClasses; i++) {
        if (i == numClasses / 2) cout << "T";
        else cout << " ";

        if (i == numClasses / 2) cout << "r";
        else cout << " ";

        if (i == numClasses / 2) cout << "u";
        else cout << " ";

        if (i == numClasses / 2) cout << "e";
        else cout << " ";

        cout << setw(3) << i << " |";
        for (int j = 0; j < numClasses; j++) {
            cout << setw(5) << confusionMatrix(i, j);
        }
        cout << "\n";
    }
    cout << "\n";
}

void printPerformanceMetrics(const Eigen::MatrixXi &confusionMatrix) {
    int numClasses = confusionMatrix.rows();
    
    cout << "Performance Metrics per Class:\n";
    cout << "Class | Precision | Recall\n";
    cout << "------|-----------|--------\n";
    
    float totalPrecision = 0.0f;
    float totalRecall = 0.0f;
    
    for (int i = 0; i < numClasses; i++) {
        // True Positives: correctly predicted as class i
        int truePositives = confusionMatrix(i, i);
        
        // False Positives: incorrectly predicted as class i
        int falsePositives = 0;
        for (int j = 0; j < numClasses; j++) {
            if (j != i) falsePositives += confusionMatrix(j, i);
        }
        
        // False Negatives: true class i but predicted as something else
        int falseNegatives = 0;
        for (int j = 0; j < numClasses; j++) {
            if (j != i) falseNegatives += confusionMatrix(i, j);
        }
        
        // Precision: TP / (TP + FP)
        float precision = (truePositives + falsePositives > 0) ? 
            static_cast<float>(truePositives) / (truePositives + falsePositives) : 0.0f;
        
        // Recall: TP / (TP + FN)
        float recall = (truePositives + falseNegatives > 0) ? 
            static_cast<float>(truePositives) / (truePositives + falseNegatives) : 0.0f;
        
        cout << setw(5) << i << " | "
             << setw(9) << fixed << setprecision(4) << precision << " | "
             << setw(6) << fixed << setprecision(4) << recall << "\n";
        
        totalPrecision += precision;
        totalRecall += recall;
    }
    
    // Average precision and recall across all classes
    cout << "------|-----------|--------\n";
    cout << " Avg  | "
         << setw(9) << fixed << setprecision(4) << (totalPrecision / numClasses) << " | "
         << setw(6) << fixed << setprecision(4) << (totalRecall / numClasses) << "\n\n";
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
            // One-hot encoded labels for the batch
            MatrixFloat batchLabels = trainLabelsOneHot.middleRows(startIdx, currentBatchSize);

            /***** FORWARD PASS: Make predictions based on current weights and biases *****/
            // 1. Input -> Hidden
            MatrixFloat hiddenPreActivation = (batchImages * weightsInputToHidden).rowwise() + biasHidden.transpose();
            MatrixFloat hiddenActivation = hiddenPreActivation.cwiseMax(0.0f);

            // 2. Hidden -> Output
            MatrixFloat logits = (hiddenActivation * weightsHiddenToOutput).rowwise() + biasOutput.transpose();

            // 3. Apply softmax to get probabilities for each class
            MatrixFloat predictions = softmax(logits);

            float batchLoss = crossEntropyLoss(predictions, batchLabels);
            totalLoss += batchLoss * currentBatchSize;

            /***** BACKWARD PASS: Compute gradients *****/
            // Output layer gradients (W2, b2)
            
            // Error is computed here vvvvvvvvvvvvvvv
            MatrixFloat gradientLogits = (predictions - batchLabels) / currentBatchSize;

            // dW2 = A1^T * dZ2
            MatrixFloat gradientWeightsHiddenToOutput = hiddenActivation.transpose() * gradientLogits;
            
            // db2 = sum(dZ2, axis=0)
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
             << " | Loss: " << fixed << setprecision(2) << avgLoss
             << " | Accuracy: " << fixed << setprecision(2) << (trainAccuracy * 100) << "%"
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
