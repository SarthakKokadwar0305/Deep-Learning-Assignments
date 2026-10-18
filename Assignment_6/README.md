# Assignment 6: Convolutional Neural Network (CNN) for Plant Disease Classification

## Objective
The objective of this assignment is to design, implement, and evaluate a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** for computer vision and agricultural pathology classification, identifying 10 distinct foliar disease classes from images of soybean leaves.

---

## Dataset
- **Dataset:** Soybean Diseased Leaf Dataset (`sivm205/soybean-diseased-leaf-dataset`)
- **Total Images:** 701 RGB leaf photographs
- **Target Disease Classes (10 categories):**
  1. `Mossaic Virus`
  2. `Southern blight`
  3. `Sudden Death Syndrome`
  4. `Yellow Mosaic`
  5. `bacterial_blight`
  6. `brown_spot`
  7. `crestamento`
  8. `ferrugen`
  9. `powdery_mildew`
  10. `septoria`
- **Data Partitioning & Ingestion:**
  - **Training Set:** 561 images (80%)
  - **Validation Set:** 140 images (20%)
  - Loaded via `tf.keras.utils.image_dataset_from_directory` with `image_size=(128, 128)` and `batch_size=32`.
  - High-throughput pipelining using `.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)`.

---

## Model Architecture

The 2D Convolutional Neural Network extracts hierarchical visual features from localized leaf lesions and discoloration:
1. **Rescaling Layer:** Normalizes pixel RGB values from $[0, 255]$ to $[0.0, 1.0]$.
2. **Convolutional Block 1:**
   - `Conv2D(32, kernel_size=(3, 3), activation='relu')`: Captures low-level edges, textures, and color gradients.
   - `MaxPooling2D(pool_size=(2, 2))`: Downsamples spatial dimensions from $128 \times 128$ to $64 \times 64$.
3. **Convolutional Block 2:**
   - `Conv2D(64, kernel_size=(3, 3), activation='relu')`: Detects mid-level spot shapes and vein structures.
   - `MaxPooling2D(pool_size=(2, 2))`: Downsamples to $32 \times 32$.
4. **Convolutional Block 3:**
   - `Conv2D(128, kernel_size=(3, 3), activation='relu')`: Encodes high-level lesion patterns and symptom clusters.
   - `MaxPooling2D(pool_size=(2, 2))`: Downsamples to $16 \times 16$.
5. **Flattening:** Vectorizes $16 \times 16 \times 128$ feature tensors into 32,768-dimensional feature vectors.
6. **Dense Fully Connected Layer:** 128 units with ReLU activation.
7. **Dropout Regularization:** Rate = `0.5` to deactivate co-dependent hidden units and prevent overfitting on small image samples.
8. **Softmax Output Layer:** 10 units outputting normalized class probability distributions.

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Cross-Entropy (`sparse_categorical_crossentropy`)
- **Metric:** Accuracy

---

## Performance Summary

| Parameter / Metric | Specification |
|---|:---:|
| **Input Resolution** | $128 \times 128 \times 3$ (RGB) |
| **Number of Classes** | 10 foliar disease categories |
| **Batch Size** | 32 |
| **Data Acceleration** | TensorFlow AUTOTUNE with caching |
| **Feature Extraction Hierarchy** | 3 Conv-Pool Stages ($32 \rightarrow 64 \rightarrow 128$ filters) |

---

## Key Takeaways & Conclusion
1. **Spatial Invariance of Convolutions:** Unlike dense MLPs which flatten images and discard spatial arrangements, 2D CNN filters preserve spatial adjacency, allowing effective identification of plant lesions regardless of their position on the leaf.
2. **Progressive Feature Abstraction:** Stacking convolution layers with max-pooling progressively expands the receptive field, transforming raw color pixels into abstract semantic disease signatures (e.g., powdery mildew dusting vs. mosaic mottling).
3. **Overfitting Mitigation:** For relatively small domain-specific image datasets (~700 images across 10 classes), strong regularization like `Dropout(0.5)` and dataset caching are essential for stable gradient descent.
