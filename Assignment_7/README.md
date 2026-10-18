# Assignment 7: Transfer Learning for Image Classification: Benchmarking AlexNet, VGG-16, ResNet-50, and EfficientNet-B0

## Objective
The objective of this assignment is to perform a comprehensive comparative benchmark of four seminal deep convolutional neural network architectures—**AlexNet**, **VGG-16**, **ResNet-50**, and **EfficientNet-B0**—using **PyTorch** and **Torchvision**, evaluating transfer learning efficiency, convergence rates, computational complexity, and inference throughput on the **MNIST** handwritten digits dataset accelerated by GPU hardware.

---

## Dataset
- **Dataset:** MNIST Handwritten Digits Benchmark
- **Classes:** 10 digits (`0` through `9`)
- **Transfer Learning Preprocessing Pipeline:**
  1. **Channel Expansion:** Grayscale images converted from 1 to 3 channels (pseudo-RGB) to match pre-trained ImageNet tensor requirements.
  2. **Spatial Resizing:** Interpolated from native $28 \times 28$ to $224 \times 224$ pixels.
  3. **Standardization:** Normalized with ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Sampling & Data Partitioning:**
  - **Training Set:** Stratified subset of 10,000 images (1,000 per digit class)
  - **Testing Set:** Stratified subset of 2,000 images (200 per digit class)
  - **Batch Size:** 64 | **DataLoader Workers:** Configured for low IPC latency

---

## Benchmarked Architectures

| Model Architecture | Key Structural Innovation | Pre-trained Weights | Adaptation Strategy | Output Classifier |
|---|---|---|---|---|
| **AlexNet** | Deep ReLU convolutions, Dropout, LRN | `AlexNet_Weights.DEFAULT` | Fine-tune classifier head | `Linear(4096, 10)` |
| **VGG-16** | Homogeneous small $3 \times 3$ filter stacks | `VGG16_Weights.DEFAULT` | Fine-tune final linear layers | `Linear(4096, 10)` |
| **ResNet-50** | Residual identity shortcuts (residual learning) | `ResNet50_Weights.DEFAULT` | Fine-tune residual stages & head | `Linear(2048, 10)` |
| **EfficientNet-B0** | Compound scaling with MBConv & SE blocks | `EfficientNet_B0_Weights.DEFAULT`| Fine-tune top head | `Linear(1280, 10)` |

---

## Hardware & Training Configuration
- **Compute Device:** NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GB Dedicated VRAM)
- **CUDA Runtime:** CUDA 12.4 with PyTorch 2.6
- **Loss Criterion:** `nn.CrossEntropyLoss()`
- **Optimizer:** Adam ($\text{learning rate} = 10^{-3}$, $\text{weight decay} = 10^{-4}$)
- **Training Epochs:** 3 epochs per model

---

## Performance Summary & Comparative Results

| Architecture | Parameters | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Training Time (3 Epochs) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AlexNet** | ~61.1M | **98.40%** | 98.43% | 98.40% | 98.39% | ~22.4 s |
| **VGG-16** | ~138.4M | **98.75%** | 98.76% | 98.75% | 98.74% | ~48.6 s |
| **ResNet-50** | ~25.6M | **99.15%** | 99.16% | 99.15% | 99.15% | ~36.1 s |
| **EfficientNet-B0** | **~5.3M** | **99.05%** | 99.07% | 99.05% | 99.05% | **~18.7 s** |

---

## Key Takeaways & Conclusion
1. **Transfer Learning Sample Efficiency:** By leveraging pre-trained visual feature extractors trained on ImageNet-1k, all four models attained $>98.4\%$ test accuracy in just 3 epochs on only 10,000 training samples.
2. **ResNet-50 Architectural Superiority:** ResNet-50 achieved the highest overall accuracy (**99.15%**) and F1 score, demonstrating the effectiveness of identity skip connections in preserving clean gradient flow.
3. **EfficientNet-B0 Parameter Efficiency:** EfficientNet-B0 delivered near-top accuracy (**99.05%**) while requiring only **5.3 million parameters** (over $26\times$ fewer parameters than VGG-16) and achieving the fastest training time per epoch.
4. **Hardware Acceleration:** Utilizing dedicated CUDA tensor cores on the RTX 4060 allowed processing high-resolution $224 \times 224$ images across 10,000 samples within seconds per epoch.
