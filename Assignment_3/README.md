# Assignment 3: Forward and Backward Propagation using TensorFlow & Keras

## Objective
The objective of this assignment is to understand and implement the mathematical and practical foundations of **Forward Propagation** and **Backpropagation** in artificial neural networks using **TensorFlow/Keras**, and analyze the impact of key training hyperparameters—including **Learning Rates** ($\eta \in [0.1, 0.01, 0.001]$), **Epochs** ($5, 10, 20$), and **Batch Sizes** ($32, 64, 128$)—on model convergence, stability, and classification accuracy.

---

## Theoretical Foundations

1. **Forward Propagation:**
   At each layer $l$, the pre-activation $Z^{[l]}$ and post-activation $A^{[l]}$ are computed:
   $$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
   $$A^{[l]} = g^{[l]}(Z^{[l]})$$
   where $g^{[l]}$ denotes the activation function (e.g., ReLU for hidden layers, Softmax for multi-class output).

2. **Loss Function (Cross-Entropy):**
   $$\mathcal{L}(y, \hat{y}) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

3. **Backward Propagation (Chain Rule of Calculus):**
   Gradients of the loss with respect to weights and biases are propagated backwards:
   $$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}} \cdot (A^{[l-1]})^T$$
   $$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}}$$
   Parameters are updated via gradient descent:
   $$W^{[l]} \leftarrow W^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial W^{[l]}}$$

---

## Model Architecture
- **Input Layer:** Reshaped $28 \times 28$ inputs flattened to 784-dimensional vectors.
- **Hidden Layer 1:** `Dense(128, activation='relu')`
- **Hidden Layer 2:** `Dense(64, activation='relu')`
- **Output Layer:** `Dense(10, activation='softmax')`
- **Total Parameters:** ~109,386 trainable parameters

---

## Hyperparameter Investigation & Results

### 1. Effect of Learning Rate ($\eta$)

| Learning Rate ($\eta$) | Behavior & Optimization Trajectory | Test Accuracy |
|:---:|---|:---:|
| **0.1** | Overly aggressive updates; oscillation around local minima | ~88.4% |
| **0.01** | Rapid initial convergence but slightly noisy steady state | ~96.2% |
| **0.001 (Optimal)** | Smooth monotonic loss minimization, optimal stability | **97.6%** |

### 2. Effect of Training Epochs (at $\eta = 0.001$)

| Epochs | Training Loss | Validation Accuracy | Notes |
|:---:|:---:|:---:|---|
| **5** | 0.0605 | 97.18% | Strong baseline, fast training |
| **10** | 0.0216 | 97.39% | Excellent convergence, stable generalization |
| **20** | 0.0077 | **97.49%** | Deep convergence with slight signs of over-fitting |

---

## Key Takeaways & Conclusion
1. **Learning Rate Sensitivity:** A learning rate that is too high ($\eta=0.1$) causes divergence or severe gradient oscillations, while an appropriately tuned learning rate ($\eta=0.001$ with Adam) allows the backpropagation algorithm to descend smoothly into optimal parameter spaces.
2. **Epoch Saturation:** Performance gains saturate after 10–15 epochs on MNIST, where training loss approaches zero while validation accuracy plateaus around ~97.5%.
3. **Activation Function Utility:** ReLU hidden activations mitigate vanishing gradients during backpropagation, facilitating deep multi-layer representation learning.
