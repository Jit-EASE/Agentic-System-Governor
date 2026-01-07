
from typing import Dict, Any
import pandas as pd


def compute_group_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_attr: pd.Series,
) -> Dict[str, Any]:
    """
    Basic fairness diagnostics per group:
    - positive rate
    - accuracy
    - positive rate ratio vs first group
    """
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "group": sensitive_attr.astype(str),
        }
    )

    results = {}
    for grp, sub in df.groupby("group"):
        positive_rate = (sub["y_pred"] == 1).mean()
        accuracy = (sub["y_pred"] == sub["y_true"]).mean()
        results[grp] = {
            "n": int(len(sub)),
            "positive_rate": float(positive_rate),
            "accuracy": float(accuracy),
        }

    groups = list(results.keys())
    if groups:
        ref = groups[0]
        for grp in groups[1:]:
            pr_ref = results[ref]["positive_rate"] + 1e-9
            pr_grp = results[grp]["positive_rate"]
            results[grp]["positive_rate_ratio_vs_" + ref] = float(pr_grp / pr_ref)

    return results
