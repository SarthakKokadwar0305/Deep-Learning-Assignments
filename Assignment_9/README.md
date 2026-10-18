# Assignment 9: Image Classification using Pre-trained Vision Transformer (ViT) vs. CNN

## Objective
The objective of this assignment is to perform image classification using a pre-trained **Vision Transformer (ViT)** model and compare its performance with a **Convolutional Neural Network (CNN)** trained from scratch on the **Fashion-MNIST** dataset.

## Dataset
- **Dataset:** Fashion-MNIST
- **Number of Classes:** 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Input Preprocessing:**
  - **CNN:** Normalized grayscale images of size $28 \times 28 \times 1$.
  - **ViT:** Normalized 3-channel (RGB) images resized to $224 \times 224 \times 3$ to match the pre-trained ViT input resolution.
- **Data Subsetting:** Class-balanced stratified subset (1,000 training images, 250 testing images) to maintain CPU-friendly execution with total notebook runtime under 3–5 minutes.

## Models Used

### 1. Lightweight Convolutional Neural Network (CNN)
Built from scratch with the following sequential pipeline:
- `Conv2D` (32 filters, $3 \times 3$ kernel, ReLU, padding='same')
- `MaxPooling2D` ($2 \times 2$ pool size)
- `Conv2D` (64 filters, $3 \times 3$ kernel, ReLU, padding='same')
- `MaxPooling2D` ($2 \times 2$ pool size)
- `Flatten`
- `Dense` (128 units, ReLU)
- `Dropout` (0.3)
- `Dense` (10 units, Softmax)
- **Training:** 5 Epochs, Batch Size = 64

### 2. Vision Transformer (ViT-S/16)
- **Backbone:** Pre-trained Vision Transformer (ViT-S/16) from TensorFlow Hub (`sayakpaul/vit_s16_fe/1`), pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k.
- **Frozen Backbone:** Backbone weights frozen (`trainable=False`, gradients detached) to leverage pre-trained representations.
- **Classification Head:**
  - `Dense` (128 units, ReLU)
  - `Dropout` (0.3)
  - `Dense` (10 units, Softmax)
- **Training:** 3 Epochs, Batch Size = 64 (only classifier head weights updated).

## Evaluation Metrics
Both models are evaluated on the test set using:
- **Accuracy**
- **Precision** (Weighted)
- **Recall** (Weighted)
- **F1 Score** (Weighted)
- **Confusion Matrix**
- **Classification Report**
- **Training and Validation Accuracy/Loss Curves**
- **Training Wall-Clock Runtime**

## Performance Summary

| Metric | Lightweight CNN (From Scratch) | Vision Transformer (ViT-S/16 Transfer Learning) |
|---|:---:|:---:|
| **Input Resolution** | $28 \times 28 \times 1$ | $224 \times 224 \times 3$ |
| **Trained Epochs** | 5 | 3 |
| **Accuracy** | **81.20%** | **85.60%** |
| **Precision (Weighted)** | **80.52%** | **85.97%** |
| **Recall (Weighted)** | **81.20%** | **85.60%** |
| **F1 Score (Weighted)** | **80.19%** | **85.67%** |
| **Training Time (CPU)** | **2.09 seconds** | **110.26 seconds** |

## Conclusion
1. **Transfer Learning Superiority:** The pre-trained Vision Transformer demonstrates superior representation power and sample efficiency. Even with a modest training budget of only 1,000 samples and 3 training epochs for the classifier head, ViT achieves higher accuracy (85.60%) and F1 score (85.67%) compared to the CNN trained from scratch (81.20% and 80.19%).
2. **Inductive Bias vs. Global Attention:** The CNN's localized inductive bias enables fast convergence and respectable baseline accuracy on small datasets, but it exhibits higher confusion among visually analogous clothing classes (e.g., Pullovers, Coats, and Shirts). In contrast, the ViT's global Multi-Head Self-Attention effectively captures long-range spatial context and fine garment silhouettes.
3. **Computational Trade-off:** The CNN is exceptionally lightweight, training in ~2 seconds on CPU with native $28 \times 28$ inputs. The ViT requires resizing images to $224 \times 224 \times 3$ and computing multi-head self-attention across 196 image patches, taking ~110 seconds on CPU. Therefore, CNNs remain ideal for resource-constrained edge deployments, while pre-trained ViTs provide higher classification accuracy.
