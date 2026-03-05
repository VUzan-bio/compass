"""Learned discrimination prediction models.

Two complementary approaches:

Approach A — FeatureDiscriminationModel (LightGBM, XGBoost fallback):
  Gradient-boosted trees on 15 thermodynamic features.
  Production uses LightGBM (pyproject.toml dependency); XGBoost used
  if available; sklearn GBR as final fallback.
  Fast inference, interpretable feature importance, no GPU needed.
  Best when paired experimental data is limited (<10K pairs).

Approach B — NeuralDiscriminationHead (PyTorch):
  Paired Compass-ML embeddings → [mut, wt, mut-wt, mut*wt] → MLP → ratio.
  Leverages the pre-trained encoder's learned representations.
  Best when encoder is available and data is sufficient.

Both models predict delta_logk (MUT - WT activity in log space).
Discrimination ratio in linear space = 10^(delta_logk).

References:
  - Huang et al. (2024) iMeta — EasyDesign dataset
  - Chen (2016) XGBoost — gradient boosting framework
  - Zhang et al. (2024) NAR — R-loop energetics
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Approach A: Feature-based discrimination model (LightGBM / XGBoost)
# ======================================================================

class FeatureDiscriminationModel:
    """Gradient-boosted model predicting discrimination from thermodynamic features.

    Input: 15 thermodynamic features (see thermo_discrimination_features.py)
    Output: delta_logk (MUT - WT activity in log space)

    Usage:
        model = FeatureDiscriminationModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model.save("checkpoints/disc_xgb.pkl")

        # Later:
        model = FeatureDiscriminationModel.load("checkpoints/disc_xgb.pkl")
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
        }
        self.model = None
        self.feature_names: list[str] = []
        self.train_metrics: dict = {}
        self._backend = "xgboost"  # or "lightgbm"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict:
        """Train the model.

        Returns dict with training metrics (loss, feature importance).
        """
        if feature_names:
            self.feature_names = feature_names

        try:
            import xgboost as xgb
            self._backend = "xgboost"
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                early_stopping_rounds=20 if X_val is not None else None,
                verbosity=0,
                **self.params,
            )
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(X, y, eval_set=eval_set, verbose=False)

        except ImportError:
            try:
                import lightgbm as lgb
                self._backend = "lightgbm"
                self.model = lgb.LGBMRegressor(
                    objective="regression",
                    metric="rmse",
                    verbosity=-1,
                    **self.params,
                )
                callbacks = None
                eval_set_lgb = None
                if X_val is not None:
                    eval_set_lgb = [(X_val, y_val)]
                    callbacks = [lgb.early_stopping(20, verbose=False)]
                self.model.fit(
                    X, y,
                    eval_set=eval_set_lgb,
                    callbacks=callbacks,
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                self._backend = "sklearn"
                self.model = GradientBoostingRegressor(
                    n_estimators=self.params["n_estimators"],
                    max_depth=self.params["max_depth"],
                    learning_rate=self.params["learning_rate"],
                    subsample=self.params["subsample"],
                    min_samples_leaf=self.params["min_child_weight"],
                    random_state=self.params["random_state"],
                )
                self.model.fit(X, y)

        # Compute training metrics
        y_pred = self.model.predict(X)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        corr = float(np.corrcoef(y, y_pred)[0, 1])

        self.train_metrics = {
            "train_rmse": rmse,
            "train_corr": corr,
            "n_samples": len(y),
            "backend": self._backend,
        }

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            if self.feature_names and len(self.feature_names) == len(importances):
                self.train_metrics["feature_importance"] = dict(
                    zip(self.feature_names, importances.tolist())
                )

        logger.info(
            "Trained %s model: RMSE=%.4f, r=%.4f, n=%d",
            self._backend, rmse, corr, len(y),
        )
        return self.train_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict delta_logk for feature matrix X."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_ratio(self, X: np.ndarray) -> np.ndarray:
        """Predict discrimination ratio in linear space (10^delta_logk)."""
        delta = self.predict(X)
        return np.power(10, delta)

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "train_metrics": self.train_metrics,
            "backend": self._backend,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureDiscriminationModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(**data["params"])
        obj.model = data["model"]
        obj.feature_names = data["feature_names"]
        obj.train_metrics = data.get("train_metrics", {})
        obj._backend = data.get("backend", "unknown")
        logger.info("Loaded %s model from %s", obj._backend, path)
        return obj

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance dict (name → importance)."""
        return self.train_metrics.get("feature_importance", {})


# ======================================================================
# Approach B: Neural discrimination head (PyTorch)
# ======================================================================

class NeuralDiscriminationModel:
    """Neural network predicting discrimination from paired embeddings.

    Wraps the existing DiscriminationHead from compass-net/heads/.
    Input: paired encoder representations (mut_pooled, wt_pooled).
    Output: predicted discrimination ratio (> 0, via Softplus).

    For standalone use (without Compass-ML encoder), can also operate
    on concatenated thermodynamic features + one-hot encoded sequence.

    Usage:
        model = NeuralDiscriminationModel(input_dim=15)
        model.fit(X_train, y_train, epochs=50)
        predictions = model.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.model = None
        self.train_metrics: dict = {}

    def _build_model(self):
        """Build PyTorch model."""
        import torch
        import torch.nn as nn

        class DiscriminationMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, dropout):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.7),
                    nn.Linear(hidden_dim // 2, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        self.model = DiscriminationMLP(self.input_dim, self.hidden_dim, self.dropout)
        return self.model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 128,
        patience: int = 15,
    ) -> dict:
        """Train the neural model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()
        self.model.to(device)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            n_batch = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batch += 1
            scheduler.step()

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
                    yv = torch.tensor(y_val, dtype=torch.float32).to(device)
                    val_pred = self.model(Xv)
                    val_loss = criterion(val_pred, yv).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final metrics
        self.model.eval()
        with torch.no_grad():
            X_all = torch.tensor(X, dtype=torch.float32).to(device)
            y_pred = self.model(X_all).cpu().numpy()
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        corr = float(np.corrcoef(y, y_pred)[0, 1])

        self.train_metrics = {
            "train_rmse": rmse,
            "train_corr": corr,
            "n_samples": len(y),
            "epochs": epoch + 1,
            "best_val_loss": best_val_loss if X_val is not None else None,
        }
        logger.info("Trained neural model: RMSE=%.4f, r=%.4f", rmse, corr)
        return self.train_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict delta_logk."""
        import torch
        if self.model is None:
            raise RuntimeError("Model not trained.")
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            return self.model(X_t).cpu().numpy()

    def predict_ratio(self, X: np.ndarray) -> np.ndarray:
        """Predict discrimination ratio in linear space."""
        delta = self.predict(X)
        return np.power(10, delta)

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "train_metrics": self.train_metrics,
        }, path)
        logger.info("Saved neural model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralDiscriminationModel":
        """Load model from checkpoint."""
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            input_dim=data["input_dim"],
            hidden_dim=data["hidden_dim"],
            dropout=data["dropout"],
        )
        obj._build_model()
        obj.model.load_state_dict(data["model_state"])
        obj.train_metrics = data.get("train_metrics", {})
        obj.model.eval()
        logger.info("Loaded neural model from %s", path)
        return obj


# ======================================================================
# Unified interface
# ======================================================================

class DiscriminationPredictor:
    """Unified interface for discrimination prediction.

    Loads the best available model (XGBoost preferred, neural fallback)
    and provides a consistent predict() API.

    Usage in pipeline:
        predictor = DiscriminationPredictor.from_checkpoint("checkpoints/")
        features = compute_features_for_pair(guide, pos, mm_type)
        ratio = predictor.predict_ratio_single(features)
    """

    def __init__(self, model, model_type: str):
        self._model = model
        self.model_type = model_type  # "xgboost" or "neural"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        prefer: str = "xgboost",
    ) -> Optional["DiscriminationPredictor"]:
        """Load the best available model from checkpoint directory.

        Searches for:
          - disc_xgb.pkl (XGBoost)
          - disc_neural.pt (PyTorch)
        """
        checkpoint_dir = Path(checkpoint_dir)

        xgb_path = checkpoint_dir / "disc_xgb.pkl"
        neural_path = checkpoint_dir / "disc_neural.pt"

        if prefer == "xgboost" and xgb_path.exists():
            model = FeatureDiscriminationModel.load(xgb_path)
            return cls(model, "xgboost")
        elif neural_path.exists():
            model = NeuralDiscriminationModel.load(neural_path)
            return cls(model, "neural")
        elif xgb_path.exists():
            model = FeatureDiscriminationModel.load(xgb_path)
            return cls(model, "xgboost")

        logger.warning("No discrimination model found in %s", checkpoint_dir)
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict delta_logk for feature matrix."""
        return self._model.predict(X)

    def predict_ratio(self, X: np.ndarray) -> np.ndarray:
        """Predict discrimination ratio in linear space."""
        return self._model.predict_ratio(X)

    def predict_ratio_single(self, features: dict[str, float]) -> float:
        """Predict ratio for a single feature dict."""
        from compass_ml_thermo import FEATURE_NAMES
        try:
            from thermo_discrimination_features import FEATURE_NAMES
        except ImportError:
            pass

        # Use the canonical feature order
        feature_names = [
            "spacer_position", "in_seed", "position_sensitivity", "region_code",
            "mismatch_destab", "is_wobble", "is_purine_purine", "is_transition",
            "mismatch_ddg", "cumulative_dg_at_mm", "seed_dg", "total_hybrid_dg",
            "energy_ratio", "gc_content", "local_gc",
        ]
        X = np.array([[features[n] for n in feature_names]], dtype=np.float32)
        ratios = self.predict_ratio(X)
        return float(ratios[0])
