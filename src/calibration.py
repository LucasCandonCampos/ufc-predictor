"""
Isotonic calibration wrapper for the UFC predictor model.

Kept in its own module so both train.py and predict.py can import it,
ensuring pickle can resolve the class path regardless of which file is __main__.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb


class _CalibratedModel:
    """
    XGBoost base model + isotonic regression calibrator.
    Exposes .estimator so SHAP can reach the underlying tree model.
    Pickle-friendly: both components are independently serialisable.
    """

    def __init__(self, base: xgb.XGBClassifier, cal: IsotonicRegression) -> None:
        self.estimator  = base
        self.calibrator = cal

    def predict_proba(self, X) -> np.ndarray:
        raw      = self.estimator.predict_proba(X)[:, 1]
        adjusted = np.clip(self.calibrator.transform(raw), 0.0, 1.0)
        return np.column_stack([1.0 - adjusted, adjusted])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
