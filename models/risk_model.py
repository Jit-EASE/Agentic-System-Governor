
from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import SYSTEM_NAME, SYSTEM_VERSION, RISK_LEVEL
from governance.audit import InferenceLogRecord, log_inference
from governance.energy import energy_tracker
from governance.explainability import linear_explanation


@dataclass
class StressRiskModel:
    model: LogisticRegression
    feature_columns: list

    @staticmethod
    def generate_synthetic_data(n: int = 1000) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "milk_yield": rng.normal(22, 5, n),
            "soil_moisture": rng.uniform(0.2, 0.9, n),
            "grass_growth_index": rng.normal(1.0, 0.3, n),
            "financial_stress_index": rng.uniform(0, 1, n),
            "herd_size": rng.integers(30, 300, n),
            "age_of_farmer": rng.integers(25, 75, n),
            "gender": rng.choice(["M", "F"], n),
            "farm_size_class": rng.choice(["small", "medium", "large"], n),
        })
        logits = (
            -0.08 * df["milk_yield"]
            + 2.5 * df["financial_stress_index"]
            + 0.005 * df["herd_size"]
            + 0.01 * (df["age_of_farmer"] - 50)
        )
        prob = 1 / (1 + np.exp(-logits))
        df["stress_high"] = (prob > 0.5).astype(int)
        return df

    @classmethod
    def train_on_synthetic(cls) -> "StressRiskModel":
        df = cls.generate_synthetic_data()
        df_enc = pd.get_dummies(df, columns=["gender", "farm_size_class"], drop_first=True)
        X = df_enc.drop(columns=["stress_high"])
        y = df_enc["stress_high"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        with energy_tracker("TRAIN_STRESS_MODEL"):
            model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        print(f"[Training] Stress model test accuracy: {acc:.3f}")

        return cls(model=model, feature_columns=list(X.columns))

    def _hash_inputs(self, row: pd.Series) -> str:
        return sha256(row.to_json().encode("utf-8")).hexdigest()

    def predict_with_governance(
        self,
        user_id: str,
        feature_row: pd.Series,
        is_child: bool,
        consent_granted: bool,
    ) -> Tuple[int, float, Dict[str, float]]:
        if not consent_granted:
            raise PermissionError("Consent not granted – inference blocked.")

        x_vec = feature_row.reindex(self.feature_columns, fill_value=0.0)
        x_mat = x_vec.values.reshape(1, -1)

        with energy_tracker("INFER_STRESS_MODEL"):
            proba = float(self.model.predict_proba(x_mat)[0, 1])
            pred = int(proba >= 0.5)
            expl = linear_explanation(self.model, x_vec)

        record = InferenceLogRecord(
            timestamp=pd.Timestamp.utcnow().isoformat(),
            user_id=user_id,
            is_child=is_child,
            consent_granted=consent_granted,
            model_name=SYSTEM_NAME,
            model_version=SYSTEM_VERSION,
            input_hash=self._hash_inputs(x_vec),
            prediction={"label": pred, "probability_high_stress": proba},
            risk_level=RISK_LEVEL,
            explanation=expl,
        )
        log_inference(record)

        return pred, proba, expl

    def get_feature_template(self) -> pd.Series:
        return pd.Series(0.0, index=self.feature_columns)
