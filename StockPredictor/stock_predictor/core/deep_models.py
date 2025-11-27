"""Lightweight deep learning regressors for time-series price prediction."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore
    nn = None  # type: ignore
    Tensor = Any  # type: ignore
    DataLoader = TensorDataset = None  # type: ignore


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - guarded by runtime check
        raise ImportError(
            "PyTorch is required for deep learning regressors. Install torch>=2.0 to use these models."
        )


def _to_float_array(data: Any) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _prepare_sequences(features: np.ndarray, sequence_length: int) -> np.ndarray:
    """Create padded rolling windows so each sample has a sequence context."""

    if sequence_length <= 1:
        return features[:, None, :]

    sequences = []
    for idx in range(len(features)):
        start = max(0, idx - sequence_length + 1)
        window = features[start : idx + 1]
        if len(window) < sequence_length:
            pad = np.repeat(features[start : start + 1], sequence_length - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        sequences.append(window)
    return np.stack(sequences)


class _BaseSequentialRegressor:
    """Shared utilities for sequence models with an sklearn-like API."""

    def __init__(
        self,
        *,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 1e-3,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        _require_torch()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[union-attr]
        self.verbose = verbose
        self._model: Optional[nn.Module] = None

    # sklearn compatibility
    def get_params(self, deep: bool = True) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "device": self.device_name,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> "_BaseSequentialRegressor":  # pragma: no cover - trivial
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def _build_dataset(self, X: Any, y: Any) -> tuple[TensorDataset, int, int]:
        features = _to_float_array(X)
        targets = np.asarray(y, dtype=np.float32)
        sequences = _prepare_sequences(features, self.sequence_length)
        X_tensor = torch.from_numpy(sequences)  # type: ignore[arg-type]
        y_tensor = torch.from_numpy(targets).float()  # type: ignore[arg-type]
        dataset = TensorDataset(X_tensor, y_tensor)
        n_samples, _, n_features = sequences.shape
        return dataset, n_samples, n_features

    def _train_loop(self, dataloader: DataLoader, model: nn.Module) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            model.train()
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                preds = model(batch_X).squeeze(-1)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            if self.verbose:
                LOGGER.info("Epoch %s | Loss %.6f", epoch + 1, epoch_loss / max(len(dataloader), 1))

    def fit(self, X: Any, y: Any) -> "_BaseSequentialRegressor":
        _require_torch()
        dataset, n_samples, n_features = self._build_dataset(X, y)
        self.device = torch.device(self.device_name)  # type: ignore[arg-type]
        self._model = self._build_model(input_size=n_features)
        self._model.to(self.device)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self._train_loop(dataloader, self._model)
        LOGGER.debug("Trained %s with %s samples and %s features", self.__class__.__name__, n_samples, n_features)
        return self

    def predict(self, X: Any) -> np.ndarray:
        _require_torch()
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        features = _to_float_array(X)
        sequences = _prepare_sequences(features, self.sequence_length)
        tensor = torch.from_numpy(sequences).to(self.device)  # type: ignore[arg-type]
        self._model.eval()
        with torch.no_grad():
            preds = self._model(tensor).squeeze(-1).cpu().numpy()
        return preds

    # Subclasses must implement
    def _build_model(self, *, input_size: int) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError


class _RNNModel(nn.Module):
    def __init__(
        self,
        rnn_layer: type[nn.RNNBase],
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.rnn = rnn_layer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        output, _ = self.rnn(x)
        final_state = output[:, -1, :]
        return self.head(final_state)


class LSTMRegressor(_BaseSequentialRegressor):
    """Sequence regressor backed by an LSTM encoder."""

    def _build_model(self, *, input_size: int) -> nn.Module:
        return _RNNModel(nn.LSTM, input_size, self.hidden_size, self.num_layers, self.dropout)


class GRURegressor(_BaseSequentialRegressor):
    """Sequence regressor backed by a GRU encoder."""

    def _build_model(self, *, input_size: int) -> nn.Module:
        return _RNNModel(nn.GRU, input_size, self.hidden_size, self.num_layers, self.dropout)


class _BaseSequentialClassifier(_BaseSequentialRegressor):
    """Sequence classifier that outputs class probabilities."""

    def _build_dataset(self, X: Any, y: Any) -> tuple[TensorDataset, int, int]:
        features = _to_float_array(X)
        targets = np.asarray(y, dtype=np.int64)
        sequences = _prepare_sequences(features, self.sequence_length)
        X_tensor = torch.from_numpy(sequences)  # type: ignore[arg-type]
        y_tensor = torch.from_numpy(targets).long()  # type: ignore[arg-type]
        dataset = TensorDataset(X_tensor, y_tensor)
        n_samples, _, n_features = sequences.shape
        return dataset, n_samples, n_features

    def _train_loop(self, dataloader: DataLoader, model: nn.Module) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            model.train()
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            if self.verbose:
                LOGGER.info("Epoch %s | Loss %.6f", epoch + 1, epoch_loss / max(len(dataloader), 1))

    def predict_proba(self, X: Any) -> np.ndarray:
        _require_torch()
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        features = _to_float_array(X)
        sequences = _prepare_sequences(features, self.sequence_length)
        tensor = torch.from_numpy(sequences).to(self.device)  # type: ignore[arg-type]
        self._model.eval()
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X: Any) -> np.ndarray:  # type: ignore[override]
        probs = self.predict_proba(X)
        labels = np.argmax(probs, axis=1)
        return labels

    def fit(self, X: Any, y: Any) -> "_BaseSequentialClassifier":  # type: ignore[override]
        _require_torch()
        dataset, n_samples, n_features = self._build_dataset(X, y)
        self.device = torch.device(self.device_name)  # type: ignore[arg-type]
        self._model = self._build_model(input_size=n_features)
        self._model.to(self.device)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self._train_loop(dataloader, self._model)
        self.classes_ = np.array([0, 1])
        LOGGER.debug(
            "Trained %s with %s samples and %s features (classification)",
            self.__class__.__name__,
            n_samples,
            n_features,
        )
        return self


class _RNNClassifier(nn.Module):
    def __init__(
        self,
        rnn_layer: type[nn.RNNBase],
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.rnn = rnn_layer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        output, _ = self.rnn(x)
        final_state = output[:, -1, :]
        return self.head(final_state)


class LSTMClassifier(_BaseSequentialClassifier):
    """Sequence classifier backed by an LSTM encoder."""

    def _build_model(self, *, input_size: int) -> nn.Module:
        return _RNNClassifier(
            nn.LSTM, input_size, self.hidden_size, self.num_layers, self.dropout
        )


class GRUClassifier(_BaseSequentialClassifier):
    """Sequence classifier backed by a GRU encoder."""

    def _build_model(self, *, input_size: int) -> nn.Module:
        return _RNNClassifier(nn.GRU, input_size, self.hidden_size, self.num_layers, self.dropout)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]


class _TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = PositionalEncoding(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        projected = self.input_projection(x)
        encoded = self.encoder(self.positional(projected))
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class TransformerRegressor(_BaseSequentialRegressor):
    """Transformer encoder regressor using simple positional encoding."""

    def __init__(self, *, nhead: int = 4, dim_feedforward: int = 256, **kwargs: Any) -> None:
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        super().__init__(**kwargs)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:  # pragma: no cover - trivial
        params = super().get_params(deep)
        params.update({"nhead": self.nhead, "dim_feedforward": self.dim_feedforward})
        return params

    def set_params(self, **params: Any) -> "TransformerRegressor":  # pragma: no cover - trivial
        for key in ["nhead", "dim_feedforward"]:
            if key in params:
                setattr(self, key, params.pop(key))
        super().set_params(**params)
        return self

    def _build_model(self, *, input_size: int) -> nn.Module:
        d_model = max(self.nhead * 2, input_size)
        return _TransformerModel(
            input_size=input_size,
            d_model=d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
