# Deep learning regressors

The project now ships with lightweight PyTorch-based regressors tailored for
sequence modeling: `LSTMRegressor`, `GRURegressor`, and
`TransformerRegressor`. Each model consumes rolling windows of historical
features and predicts the next value in the series.

All models expose familiar sklearn-style `fit` and `predict` methods, and can
be created through the existing `ModelFactory` using the `lstm`, `gru`, or
`transformer` model types.

## Quickstart

```python
import numpy as np
import pandas as pd

from stock_predictor.core.models import ModelFactory

# Synthetic close-price series
prices = pd.Series(np.linspace(0, 10, 64) + np.random.normal(scale=0.2, size=64))
features = prices.to_frame("close")

# Train an LSTM over 10-step windows
factory = ModelFactory(
    "lstm",
    {"sequence_length": 10, "epochs": 5, "hidden_size": 32, "batch_size": 16},
)
model = factory.create("regression")
model.fit(features, prices)

# Predict for the same horizon (or new data with at least `sequence_length` rows)
predictions = model.predict(features)
print(predictions[-5:])
```

For transformers, the configuration is similar:

```python
factory = ModelFactory(
    "transformer",
    {
        "sequence_length": 20,
        "epochs": 5,
        "nhead": 4,
        "dim_feedforward": 128,
        "hidden_size": 64,
    },
)
transformer = factory.create("regression")
transformer.fit(features, prices)
future_pred = transformer.predict(features.tail(30))
```

> **Note:** PyTorch is an optional dependency. Install `torch>=2.0` to enable
> these models.
