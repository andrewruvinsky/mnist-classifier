// Author: Andrew Ruvinsky
// src/main.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include "SimpleANN.h"
using namespace std;

using MatrixFloat = Eigen::MatrixXf;
using VectorInt = Eigen::VectorXi;

uint32_t readBigEndianUInt32(ifstream& fileStream) {
    uint32_t value;
    // Reads 4 bytes/32-bit big-endian unsigned int from file
    fileStream.read(reinterpret_cast<char*>(&value), 4);
    // Convert from big-endian to host endianness (whatever the host machine uses)
    return __builtin_bswap32(value);
}

// Load MNIST image file into an Eigen matrix (numImages x numPixels)
MatrixFloat loadMnistImages(const string& filePath) {
    //ios::binary ensures file is read as binary (NOT text)
    ifstream fileStream(filePath, ios::binary);
    if (!fileStream)
        throw runtime_error("Could not open file: " + filePath + "\nEnsure you're building from the root directory.");

    uint32_t magicNumber = readBigEndianUInt32(fileStream);
    uint32_t numImages   = readBigEndianUInt32(fileStream);
    uint32_t numRows     = readBigEndianUInt32(fileStream);
    uint32_t numCols     = readBigEndianUInt32(fileStream);

    if (magicNumber != 2051)
        throw runtime_error("Invalid MNIST image file magic number in: " + filePath);

    size_t numPixels = static_cast<size_t>(numRows) * numCols;
    // 60,000 x 784 matrix for training set; 28x28 pixels
    MatrixFloat images(numImages, numPixels);
    vector<unsigned char> pixelBuffer(numPixels);

    for (uint32_t i = 0; i < numImages; i++) {
        fileStream.read(reinterpret_cast<char*>(pixelBuffer.data()), pixelBuffer.size());

        for (size_t pixelIndex = 0; pixelIndex < numPixels; pixelIndex++) {
            // Normalize to [0,1]
            images(i, pixelIndex) = pixelBuffer[pixelIndex] / 255.0f;
        }
    }

    return images;
}

// Load MNIST label file into an Eigen vector (numLabels)
VectorInt loadMnistLabels(const string& filePath) {
    ifstream fileStream(filePath, ios::binary);
    
    if (!fileStream) throw runtime_error("Could not open file: " + filePath);

    uint32_t magicNumber = readBigEndianUInt32(fileStream);
    uint32_t numLabels   = readBigEndianUInt32(fileStream);

    if (magicNumber != 2049) throw runtime_error("Invalid MNIST label file magic number in: " + filePath);

    VectorInt labels(numLabels);

    for (uint32_t i = 0; i < numLabels; i++) {
        unsigned char label;
        // Reads a single byte/unsigned char from fileStream
        fileStream.read(reinterpret_cast<char*>(&label), 1);
        labels(i) = static_cast<int>(label);
    }

    return labels;
}

// Convert label vector to one-hot encoded matrix (numSamples x numClasses)
MatrixFloat oneHotEncode(const VectorInt &labels, int numClasses) {
    MatrixFloat oneHot = MatrixFloat::Zero(labels.size(), numClasses);
    for (int i = 0; i < labels.size(); i++) {
        oneHot(i, labels(i)) = 1.0f;
    }

    return oneHot;
}

int main() {
    try {
        // Load training data
        cout << "Loading MNIST dataset...\n";
        MatrixFloat trainImages = loadMnistImages("data/train-images.idx3-ubyte");
        VectorInt trainLabels   = loadMnistLabels("data/train-labels.idx1-ubyte");
        cout << "Loaded " << trainImages.rows() << " training images\n";
        
        // Load test data
        MatrixFloat testImages = loadMnistImages("data/t10k-images.idx3-ubyte");
        VectorInt testLabels   = loadMnistLabels("data/t10k-labels.idx1-ubyte");
        cout << "Loaded " << testImages.rows() << " test images\n\n";

        // Number of output classes for [0-9] digits
        const int numClasses = 10;
        // One-hot encode labels for training
        MatrixFloat trainLabelsOneHot = oneHotEncode(trainLabels, numClasses);
        int numFeatures = trainImages.cols(); // 784 (28x28 pixels)

        /***** Hyperparameters *****/ 
        int numHiddenNeurons = 64; // Number of neurons in hidden layer
        int numEpochs = 10; // # full passes through the training set
        int batchSize = 128; // Determines # samples to process before updating weights
        float learningRate = 0.1f; // When updating gradients, how "big" of a step to take
        /***************************/

        // Create and train ANN model
        SimpleANN model(numFeatures, numHiddenNeurons, numClasses);

        cout << "Training ANN...\n";
        cout << "# Hidden neurons: " << numHiddenNeurons
        << ", Epochs: " << numEpochs 
        << ", Batch size: " << batchSize 
        << ", Learning rate: " << learningRate << "\n\n";
        model.train(trainImages, trainLabelsOneHot, numEpochs, learningRate, batchSize);
        
        // Evaluate on test set
        cout << "Evaluating on test set...\n";
        MatrixFloat testPredictions = model.predict(testImages);
        float testAccuracy = computeAccuracy(testPredictions, testLabels);
        
        cout << "Test Accuracy: " << (testAccuracy * 100) << "%\n";

    } catch (const exception& error) {
        cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
