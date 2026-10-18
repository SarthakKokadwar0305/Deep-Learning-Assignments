# Assignment 4: Stock Price Prediction Using Long Short-Term Memory (LSTM) Networks

## Objective
The objective of this assignment is to design, implement, and evaluate a **Long Short-Term Memory (LSTM)** deep recurrent neural network architecture for univariate time-series forecasting, specifically predicting future daily closing prices of **Apple Inc. (AAPL)** stock based on historical financial time-series data.

---

## Dataset
- **Asset Ticker:** AAPL (Apple Inc.)
- **Source:** Yahoo Finance API (`yfinance`)
- **Timeframe:** January 1, 2015 – January 1, 2025 (10-year historical window, 2,518 trading days)
- **Target Variable:** Adjusted Closing Price (`Close`)
- **Data Preprocessing & Windowing:**
  1. **Data Cleaning:** Missing values, non-numeric artifacts, and weekend gaps removed.
  2. **Normalization:** Feature values scaled to $[0.0, 1.0]$ using `MinMaxScaler(feature_range=(0, 1))` to maintain numerical stability in recurrent gating mechanisms.
  3. **Sliding Time Window:** Formatted into overlapping sequences using a **60-day lookback window** (i.e., using trading days $t-59$ through $t$ to predict the price at $t+1$).
  4. **Dataset Splitting:** 80% training partition and 20% hold-out test partition.

---

## Model Architecture

The multi-layer recurrent architecture utilizes gated LSTM cells to capture long-term temporal dependencies:
1. **Input Layer:** Sequence shape of `(60, 1)` corresponding to 60 timesteps with 1 feature per step.
2. **LSTM Layer 1:** 50 recurrent memory units with `return_sequences=True` to preserve sequence representations for the next recurrent layer.
3. **Dropout Regularization 1:** Rate = `0.2` to mitigate overfitting and co-adaptation of memory cells.
4. **LSTM Layer 2:** 50 recurrent units with `return_sequences=False` to condense sequential hidden states into a summary representation.
5. **Dropout Regularization 2:** Rate = `0.2`.
6. **Dense Output Layer:** 1 unit with linear activation predicting the continuous normalized stock price at day $t+1$.

- **Total Parameters:** 30,651 trainable parameters
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (`mean_squared_error`)
- **Callback:** `ModelCheckpoint` saving optimal weights to `stock_lstm.keras` based on minimum validation loss.

---

## Evaluation Metrics & Visualizations

The performance on unscaled test prices is evaluated using:
- **Root Mean Squared Error (RMSE):** Penalizes large deviations in dollar price predictions.
- **Mean Absolute Error (MAE):** Average absolute price divergence in USD ($).
- **Time-Series Trajectory Visualizations:** Plots comparing actual historical Apple closing prices versus model predictions across test timelines.

---

## Performance Summary

| Metric / Parameter | Value |
|---|:---:|
| **Lookback Sequence Window** | 60 trading days |
| **Training Samples** | ~1,950 sequences |
| **Testing Samples** | ~500 sequences |
| **Model Checkpoint** | Saved as `stock_lstm.keras` |
| **Trend Tracking Fidelity** | Captures market momentum, upward rallies, and pullbacks |

---

## Key Takeaways & Conclusion
1. **Long-Term Memory Retention:** Standard RNNs struggle with vanishing gradients over long horizons; the input, forget, and output gates of the LSTM enable effective retention of trends across the 60-day window.
2. **Impact of Normalization:** MinMax scaling is vital for LSTMs because the internal tanh and sigmoid activation functions operate within restricted ranges ($[-1, 1]$ and $[0, 1]$).
3. **Practical Forecasting Constraints:** While the LSTM captures directional market momentum and smooths price trajectories well, sudden volatility shocks and earnings announcements remain challenging without exogenous macroeconomic indicators.
