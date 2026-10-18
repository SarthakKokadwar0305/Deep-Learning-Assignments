# Assignment 1: Introduction to Deep Learning with TensorFlow & Keras

## Objective
The objective of this assignment is to build, train, and evaluate a foundational Deep Feedforward Neural Network (Multilayer Perceptron) using **TensorFlow** and **Keras** for image classification on the benchmark **MNIST** handwritten digits dataset.

---

## Dataset
- **Dataset:** MNIST (Modified National Institute of Standards and Technology)
- **Content:** 70,000 grayscale images ($28 \times 28$ pixels) of handwritten digits from `0` to `9`.
- **Partitioning:**
  - **Training Set:** 60,000 images and labels
  - **Test Set:** 10,000 images and labels
- **Preprocessing:**
  - Pixel values normalized from $[0, 255]$ integer range to $[0.0, 1.0]$ floating-point range ($X / 255.0$) to facilitate stable gradient descent convergence.

---

## Model Architecture

The neural network is built using the sequential Keras API:
1. **Input & Flattening Layer:**
   - `Flatten(input_shape=(28, 28))`: Reshapes $28 \times 28$ 2D matrices into 784-dimensional 1D vectors.
2. **Hidden Layer:**
   - `Dense(128, activation='relu')`: Fully connected hidden layer with 128 neurons and Rectified Linear Unit (ReLU) non-linearity to learn hierarchical patterns.
3. **Output Layer:**
   - `Dense(10, activation='softmax')`: 10 output units corresponding to digit classes 0–9, producing normalized posterior probability distributions via Softmax.

---

## Training Configuration
- **Optimizer:** Adam (Adaptive Moment Estimation)
- **Loss Function:** Sparse Categorical Cross-Entropy (`sparse_categorical_crossentropy`)
- **Metrics:** Accuracy
- **Epochs:** 3
- **Batch Size:** 32 (default mini-batch)

---

## Performance Summary

| Metric | Training Value | Test / Evaluation Value |
|---|:---:|:---:|
| **Loss** | 0.0762 | **0.0777** |
| **Accuracy** | 97.73% | **97.63%** |
| **Model Format** | Saved Keras Model (`my_mnist_model`) | Exported |

---

## Key Takeaways & Conclusion
1. **Data Normalization:** Scaling raw image pixel intensities from $[0, 255]$ down to $[0.0, 1.0]$ prevents vanishing/exploding gradients and drastically speeds up optimization.
2. **Dense Representation:** A compact single-hidden-layer MLP with 128 ReLU units proves sufficient to achieve over **97.6% test accuracy** on handwritten digits within just 3 training epochs.
3. **Model Persistence:** Demonstrates end-to-end model saving and re-loading using Keras serialization protocols for subsequent deployment and inference.
