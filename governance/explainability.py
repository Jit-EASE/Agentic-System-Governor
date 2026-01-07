
from typing import Dict
import pandas as pd
from sklearn.linear_model import LogisticRegression


def linear_explanation(
    model: LogisticRegression,
    feature_vector: pd.Series,
) -> Dict[str, float]:
    """
    Simple feature contribution explanation for linear/logistic model:
    contribution_i ≈ coef_i * x_i
    """
    if not hasattr(model, "coef_"):
        return {}

    coefs = model.coef_[0]
    contribs = {}
    for idx, feature_name in enumerate(feature_vector.index):
        x_i = feature_vector.iloc[idx]
        coef_i = coefs[idx]
        contribs[feature_name] = float(coef_i * x_i)
    return contribs
