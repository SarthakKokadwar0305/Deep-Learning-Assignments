# Assignment 10: Deep Learning for Audio Classification with Speech-to-Text (STT) and Text-to-Speech (TTS)

## Objective
The objective of this assignment is to design, implement, and evaluate an end-to-end deep learning audio pipeline that integrates:
1. **Audio Feature Extraction**: Transforming 1D raw acoustic waveforms into 2D time-frequency representations (**Mel Spectrograms** and **MFCCs**).
2. **Lightweight Convolutional Neural Network (CNN)**: Building a CPU-friendly 2D CNN model to accurately classify spoken navigational commands.
3. **Speech-to-Text (STT)**: Converting spoken audio samples into text using OpenAI's pre-trained **Whisper Tiny** model.
4. **Text-to-Speech (TTS)**: Synthesizing spoken audio responses from recognized commands using **Google Text-to-Speech (gTTS)**.
5. **Comprehensive Evaluation**: Assessing classification accuracy, precision, recall, F1 score, confusion matrix, and training/validation trajectories.

---

## Dataset
- **Dataset:** TensorFlow Speech Commands (Mini) Dataset
- **Selected Classes:** 4 distinct navigational keyword commands:
  - `"yes"`
  - `"no"`
  - `"up"`
  - `"down"`
- **Audio Properties:**
  - Sampling Rate: 16,000 Hz (16 kHz, single-channel mono PCM)
  - Duration: Fixed 1.0 second (16,000 samples) via zero-padding and truncation
- **Data Partitioning:**
  - **Training Set:** 1,600 samples (400 samples per class, stratified and balanced)
  - **Validation Set:** 400 samples (100 samples per class, stratified and balanced)
  - Shuffled and cached via high-throughput `tf.data.Dataset` pipelines with prefetching.

---

## Model Used

### Lightweight 2D Convolutional Neural Network (CNN)
The classifier is built using a sequential convolutional pipeline matching the assignment requirements:
1. **Input Layer**: Shape `(124, 32, 1)` corresponding to 124 time frames and 32 Mel frequency bins.
2. **Normalization Layer**: Adapts to standardize mean and variance across spectrogram bins.
3. **Conv2D (1)**: 32 filters, $3 \times 3$ kernel, ReLU activation, padding `'same'`.
4. **MaxPooling2D (1)**: $2 \times 2$ downsampling.
5. **Conv2D (2)**: 64 filters, $3 \times 3$ kernel, ReLU activation, padding `'same'`.
6. **MaxPooling2D (2)**: $2 \times 2$ downsampling.
7. **Flatten**: Vectorizes feature maps into a 1D representation.
8. **Dense (Hidden)**: 64 units with ReLU activation.
9. **Dropout**: Regularization rate of 0.3 to prevent overfitting.
10. **Dense (Output)**: 4 units with Softmax activation for multiclass probability output.

- **Total Parameters:** 4,082,372 (15.57 MB)
- **Optimizer:** Adam ($\text{learning rate} = 10^{-3}$)
- **Loss Function:** Sparse Categorical Crossentropy
- **Training Hyperparameters:**
  - **Epochs:** 5
  - **Batch Size:** 32

---

## STT Method (Speech-to-Text)
- **Model:** OpenAI **Whisper Tiny** (`whisper.load_model('tiny')`).
- **Architecture:** Sequence-to-sequence Transformer encoder-decoder trained on 680,000 hours of multilingual, multitask supervised audio data (~39M parameters).
- **Execution:** Input audio is read at 16 kHz and converted to normalized float32 format $[-1.0, 1.0]$ in-memory, bypassing external system binary dependencies (`ffmpeg`) and executing fully on CPU in FP32 mode.
- **Evaluation Sample:** Sample test utterance `9712cce0_nohash_0.wav` (Ground truth: `"no"`).
- **Result:** Whisper Tiny accurately transcribed the audio as **`"No."`** with near-instantaneous inference latency.

---

## TTS Method (Text-to-Speech)
- **Engine:** Google Text-to-Speech (**`gTTS`**).
- **Operation:** Translates the recognized command into natural spoken verbal confirmation:
  $$\text{"Speech command recognized as No. Action confirmed."}$$
- **Delivery:** Generates MP3 audio into an in-memory buffer and renders an interactive HTML5 audio player widget directly inside the notebook via `IPython.display.Audio(data=..., embed=True)`.

---

## Evaluation Metrics
The CNN model is evaluated on the 400 validation samples across standard statistical metrics:
- **Accuracy**
- **Precision** (Weighted & Macro)
- **Recall** (Weighted & Macro)
- **F1 Score** (Weighted & Macro)
- **Confusion Matrix**
- **Classification Report**
- **Diagnostic Curves:**
  - Training & Validation Accuracy Graph
  - Training & Validation Loss Graph

---

## Performance Summary

| Metric / Parameter | Value |
|---|:---:|
| **Target Keyword Classes** | 4 (`yes`, `no`, `up`, `down`) |
| **Input Shape (Mel Spectrogram)** | $(124 \times 32 \times 1)$ |
| **Trained Epochs** | 5 |
| **Batch Size** | 32 |
| **Validation Accuracy** | **84.75%** |
| **Precision (Weighted)** | **85.21%** |
| **Recall (Weighted)** | **84.75%** |
| **F1 Score (Weighted)** | **84.90%** |
| **Precision (Macro)** | **85.21%** |
| **Recall (Macro)** | **84.75%** |
| **F1 Score (Macro)** | **84.90%** |
| **CPU Training Duration** | **6.05 seconds** |
| **Whisper Tiny STT Output** | Ground truth: `'no'` $\rightarrow$ Transcribed: `'No.'` |
| **gTTS Synthesis Output** | 31,488 bytes synthesized and embedded |

### Detailed Classification Report

```text
              precision    recall  f1-score   support

         yes     0.9381    0.9100    0.9239       100
          no     0.7523    0.8200    0.7847       100
          up     0.9239    0.8500    0.8854       100
        down     0.7941    0.8100    0.8020       100

    accuracy                         0.8475       400
   macro avg     0.8521    0.8475    0.8490       400
weighted avg     0.8521    0.8475    0.8490       400
```

---

## Conclusion
1. **Acoustic Time-Frequency Representation:** Raw 1D waveforms contain significant temporal variance and phase ambiguity. Extracting 2D Log-Mel Spectrograms compresses acoustic energy into perceptually scaled frequency bins, allowing 2D CNN filters to learn invariant local spectro-temporal features (formants, pitch trajectories, and energy bursts).
2. **CPU-Friendly High Performance:** The lightweight CNN architecture achieved **84.75% validation accuracy** and **84.90% weighted F1 score** in only 5 epochs. With a training time of **6.05 seconds on CPU**, the model provides an ideal balance between low computational footprint and high classification fidelity.
3. **End-to-End Multimodal Pipeline:** Integrating **OpenAI Whisper Tiny** (STT) and **Google Text-to-Speech** (TTS) completes a full voice-driven conversational loop:
   $$\text{Acoustic Waveform} \xrightarrow{\text{STFT + Mel}} \text{Spectrogram} \xrightarrow{\text{CNN Classifier}} \text{Intent Prediction} \xrightarrow{\text{Whisper Tiny}} \text{STT Text} \xrightarrow{\text{gTTS}} \text{Spoken Feedback}$$
4. **Reproducibility & Execution:** The notebook contains automatic package dependency checking and self-contained dataset retrieval, enabling reproducible top-to-bottom execution on any standard workstation or cloud Jupyter environment.
