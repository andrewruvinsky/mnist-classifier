#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
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
        throw runtime_error("Could not open file: " + filePath);

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
    if (!fileStream)
        throw runtime_error("Could not open file: " + filePath);

    uint32_t magicNumber = readBigEndianUInt32(fileStream);
    uint32_t numLabels   = readBigEndianUInt32(fileStream);

    if (magicNumber != 2049)
        throw runtime_error("Invalid MNIST label file magic number in: " + filePath);

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
MatrixFloat oneHotEncode(const VectorInt& labels, int numClasses) {
    MatrixFloat oneHot = MatrixFloat::Zero(labels.size(), numClasses);
    for (int i = 0; i < labels.size(); i++) {
        oneHot(i, labels(i)) = 1.0f;
    }
    return oneHot;
}

// Softmax function: converts logits to probabilities
// Applied row-wise: each row is a sample, each column is a class
MatrixFloat softmax(const MatrixFloat& logits) {
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
float crossEntropyLoss(const MatrixFloat& predictions, const MatrixFloat& targets) {
    // Add small epsilon to avoid log(0)
    const float epsilon = 1e-10f;
    MatrixFloat clippedPredictions = predictions.array().max(epsilon);
    
    // Cross-entropy: -sum(y_true * log(y_pred)) / numSamples
    float loss = -(targets.array() * clippedPredictions.array().log()).sum() / predictions.rows();
    return loss;
}

// Compute accuracy
float computeAccuracy(const MatrixFloat& predictions, const VectorInt& trueLabels) {
    int correct = 0;
    for (int i = 0; i < predictions.rows(); i++) {
        int predictedClass;
        predictions.row(i).maxCoeff(&predictedClass);
        if (predictedClass == trueLabels(i)) {
            correct++;
        }
    }
    return static_cast<float>(correct) / predictions.rows();
}

// Softmax regression model
class SoftmaxRegression {
public:
    MatrixFloat weights;  // numFeatures x numClasses
    Eigen::VectorXf bias; // numClasses
    
    SoftmaxRegression(int numFeatures, int numClasses) {
        // Initialize weights with small random values
        weights = MatrixFloat::Random(numFeatures, numClasses) * 0.01f;
        bias = Eigen::VectorXf::Zero(numClasses);
    }
    
    // Forward pass: compute predictions
    MatrixFloat predict(const MatrixFloat& images) {
        // logits = X * W + b
        MatrixFloat logits = images * weights;
        logits.rowwise() += bias.transpose();
        return softmax(logits);
    }
    
    // Train the model using gradient descent
    void train(const MatrixFloat& trainImages, const MatrixFloat& trainLabelsOneHot,
               int numEpochs, float learningRate, int batchSize = 100) {
        int numSamples = trainImages.rows();
        int numBatches = (numSamples + batchSize - 1) / batchSize;
        
        cout << "Training softmax regression...\n";
        cout << "Epochs: " << numEpochs << ", Learning rate: " << learningRate << ", Batch size: " << batchSize << "\n\n";
        
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            float totalLoss = 0.0f;
            
            for (int batch = 0; batch < numBatches; batch++) {
                int startIdx = batch * batchSize;
                int endIdx = min(startIdx + batchSize, numSamples);
                int currentBatchSize = endIdx - startIdx;
                
                // Get batch
                MatrixFloat batchImages = trainImages.middleRows(startIdx, currentBatchSize);
                MatrixFloat batchLabels = trainLabelsOneHot.middleRows(startIdx, currentBatchSize);
                
                // Forward pass
                MatrixFloat predictions = predict(batchImages);
                
                // Compute loss
                float loss = crossEntropyLoss(predictions, batchLabels);
                totalLoss += loss * currentBatchSize;
                
                // Backward pass: compute gradients
                // Gradient of cross-entropy + softmax: predictions - targets
                MatrixFloat gradientLogits = (predictions - batchLabels) / currentBatchSize;
                
                // Gradient for weights: X^T * gradientLogits
                MatrixFloat gradientWeights = batchImages.transpose() * gradientLogits;
                
                // Gradient for bias: sum over samples
                Eigen::VectorXf gradientBias = gradientLogits.colwise().sum();
                
                // Update parameters
                weights -= learningRate * gradientWeights;
                bias -= learningRate * gradientBias;
            }
            
            // Report progress every few epochs
            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                float avgLoss = totalLoss / numSamples;
                MatrixFloat trainPredictions = predict(trainImages);
                float trainAccuracy = computeAccuracy(trainPredictions, 
                    Eigen::Map<const VectorInt>(reinterpret_cast<const int*>(trainLabelsOneHot.data()), 
                                                 trainLabelsOneHot.rows()));
                
                // Get actual labels from one-hot encoding for accuracy calculation
                VectorInt actualLabels(trainLabelsOneHot.rows());
                for (int i = 0; i < trainLabelsOneHot.rows(); i++) {
                    trainLabelsOneHot.row(i).maxCoeff(&actualLabels(i));
                }
                trainAccuracy = computeAccuracy(trainPredictions, actualLabels);
                
                cout << "Epoch " << (epoch + 1) << "/" << numEpochs 
                     << " - Loss: " << avgLoss 
                     << " - Accuracy: " << (trainAccuracy * 100) << "%\n";
            }
        }
        cout << "\nTraining complete!\n\n";
    }
};

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

        // One-hot encode labels
        const int numClasses = 10;
        MatrixFloat trainLabelsOneHot = oneHotEncode(trainLabels, numClasses);
        
        // Create and train model
        int numFeatures = trainImages.cols(); // 784 (28x28 pixels)
        SoftmaxRegression model(numFeatures, numClasses);
        
        // Hyperparameters
        int numEpochs = 1;
        int batchSize = 128;
        float learningRate = 0.1f;
        
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
