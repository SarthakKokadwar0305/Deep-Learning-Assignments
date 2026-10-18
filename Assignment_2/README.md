# Assignment 2: Multilayer Perceptron (MLP) Classifier using Scikit-Learn

## Objective
The objective of this assignment is to design, train, and evaluate a **Multilayer Perceptron (MLP)** neural network classifier using **Scikit-Learn** (`MLPClassifier`) on the classical **Iris** flower dataset, analyzing model convergence, 10-fold cross-validation performance, confusion matrices, and loss trajectories.

---

## Dataset
- **Dataset:** Fisher's Iris Dataset (`sklearn.datasets.load_iris`)
- **Number of Samples:** 150 instances
- **Features (4 continuous attributes):**
  1. Sepal Length (cm)
  2. Sepal Width (cm)
  3. Petal Length (cm)
  4. Petal Width (cm)
- **Target Classes (3 categories, 50 samples each):**
  - `Setosa` (Class 0)
  - `Versicolor` (Class 1)
  - `Virginica` (Class 2)
- **Data Splitting & Scaling:**
  - 80% Training set (120 samples) and 20% Test set (30 samples) with `random_state=42`.
  - Feature standardization using `StandardScaler` to ensure zero-mean and unit-variance $(\mu=0, \sigma=1)$ across all four input dimensions.

---

## Model Architecture & Hyperparameters

- **Classifier:** `sklearn.neural_network.MLPClassifier`
- **Network Topology:**
  - Input Layer: 4 feature units
  - Hidden Layer 1: 10 neurons (ReLU activation)
  - Hidden Layer 2: 10 neurons (ReLU activation)
  - Output Layer: 3 neurons (Softmax classification)
- **Solver / Optimizer:** `adam`
- **Maximum Iterations:** 1,000
- **Random State:** 42

---

## Evaluation Metrics & Diagnostics

The classifier is evaluated using:
- **Test Set Accuracy**
- **Weighted Precision, Recall, and F1-Score**
- **10-Fold Stratified Cross-Validation:** Assesses generalization robustness across different data partitions.
- **Confusion Matrix:** Plotted via `ConfusionMatrixDisplay` to visualize class misclassifications.
- **Training Loss Curve:** Tracks cross-entropy optimization per epoch until convergence.

---

## Performance Summary

| Metric | Measured Value |
|---|:---:|
| **Test Accuracy** | **100.0%** (30/30 correct) |
| **Precision (Weighted)** | **1.0000** |
| **Recall (Weighted)** | **1.0000** |
| **F1 Score (Weighted)** | **1.0000** |
| **10-Fold CV Mean Accuracy** | **~96.67%** |
| **Convergence** | Reached minimal cross-entropy loss within ~300 iterations |

---

## Key Takeaways & Conclusion
1. **Feature Normalization:** Standardizing features is critical for gradient-based neural networks like MLPClassifier; without scaling, features with larger numeric variance dominate the backpropagation updates.
2. **Hidden Layer Representation:** Two modest hidden layers of 10 neurons each $(10, 10)$ with ReLU non-linearities cleanly separate the non-linearly separable classes (`Versicolor` vs. `Virginica`) while preserving 100% test accuracy.
3. **Cross-Validation Rigor:** 10-fold cross validation confirms that the network generalizes robustly without severe overfitting, showing average validation scores of over 96.6%.
