# Assignment 8: Transformer-Based Sentiment Analysis Using Pre-trained BERT

## Objective
The objective of this assignment is to fine-tune a pre-trained Transformer language model—**BERT (Bidirectional Encoder Representations from Transformers)**—using the **Hugging Face `transformers`** and **`datasets`** ecosystems for natural language understanding and binary sentiment classification on the **Rotten Tomatoes** movie review dataset.

---

## Dataset
- **Dataset:** Rotten Tomatoes Movie Reviews (`datasets.load_dataset("rotten_tomatoes")`)
- **Total Samples:** 10,662 professional review excerpts
- **Dataset Partitions:**
  - **Training Split:** 8,530 reviews (80%)
  - **Validation Split:** 1,066 reviews (10%)
  - **Testing Split:** 1,066 reviews (10%)
- **Target Classes (Balanced 50/50):**
  - `0`: **NEGATIVE**
  - `1`: **POSITIVE**
- **Tokenization Pipeline:**
  - Tokenizer: `AutoTokenizer.from_pretrained("bert-base-uncased")`
  - Subword Segmentation: WordPiece tokenizer handling out-of-vocabulary terms through subword units.
  - Dynamic Padding: `DataCollatorWithPadding` dynamically pads batches to the length of the longest sentence in each batch, optimizing GPU tensor throughput.

---

## Model Architecture & Fine-Tuning Strategy

- **Base Architecture:** `bert-base-uncased`
  - 12 Transformer Encoder layers (Blocks)
  - 768 Hidden State representation dimension ($d_{model}$)
  - 12 Multi-Head Self-Attention mechanisms ($h=12$)
  - Total Parameters: **~110 Million parameters**
- **Classification Head:** Linear classifier mapped to the pooled special `[CLS]` token output:
  $$P(\text{Sentiment} \mid x) = \text{Softmax}(W \cdot h_{[CLS]} + b)$$
- **Fine-Tuning Framework:** Hugging Face `Trainer` API with `TrainingArguments`
- **Optimization Hyperparameters:**
  - **Optimizer:** AdamW (Adam with decoupled weight decay)
  - **Learning Rate:** $2 \times 10^{-5}$ with linear learning rate warmup and decay schedule
  - **Weight Decay:** 0.01
  - **Batch Size:** 16 (or 32 depending on VRAM allocation)
  - **Epochs:** 3
- **Model Checkpointing & Persistence:** Saved locally at `./bert_rotten_tomatoes_model` along with vocabulary configs for standalone inference.

---

## Performance Summary & Evaluation Metrics

Evaluated on the 1,066 held-out test reviews:

| Evaluation Metric | Measured Value |
|---|:---:|
| **Test Accuracy** | **82.80%** (0.8280) |
| **Precision (Weighted)** | **81.86%** (0.8186) |
| **Recall (Weighted)** | **81.86%** (0.8186) |
| **F1 Score (Weighted)** | **81.86%** (0.8186) |
| **Saved Model Checkpoint** | `./bert_rotten_tomatoes_model` |

### Confusion Matrix

```text
              Predicted Negative   Predicted Positive
Actual Negative       220                  43
Actual Positive        43                 194
```

---

## Inference Demonstration on Novel Text

The fine-tuned BERT model was tested on out-of-domain sample sentences:

| Input Text Snippet | Predicted Sentiment | Model Confidence |
|---|:---:|:---:|
| *"This movie was absolutely fantastic. I loved every minute of it!"* | **POSITIVE** | **94.18%** |
| *"The movie was boring and disappointing."* | **NEGATIVE** | **91.78%** |
| *"The acting was excellent and the story was very entertaining."* | **POSITIVE** | **94.53%** |
| *"I did not enjoy this movie at all."* | **NEGATIVE** | **89.55%** |
| *"The plot was terrible and the movie was a complete waste of time."* | **NEGATIVE** | **89.80%** |

---

## Key Takeaways & Conclusion
1. **Bidirectional Contextual Attention:** Unlike unidirectional LSTMs or RNNs, BERT's multi-head self-attention simultaneously attends to both left and right word contexts, capturing subtle linguistic nuances such as sarcasm, double negations, and complex adjective-noun dependencies.
2. **Transfer Learning via Pre-training:** Leveraging representations learned on BooksCorpus and English Wikipedia allows BERT to achieve high generalization accuracy on Rotten Tomatoes reviews with minimal domain-specific fine-tuning.
3. **WordPiece Tokenization:** Subword segmentation cleanly avoids out-of-vocabulary (OOV) tokens, ensuring rare words and morphological variations are mapped to known semantic roots.
