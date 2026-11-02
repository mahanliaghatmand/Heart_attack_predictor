"""Model-agnostic "what-if" explanation for a single patient.

For every input feature we replace the patient's value with a *reference* value
(the dataset median/mode stored in ``schema.FEATURES[*].default``) and measure
how the predicted probability moves. A positive delta means "this value pushes
the estimated risk up compared to a typical patient".

This is an approximation (one-at-a-time perturbation, no interactions) and is
NOT a causal statement.
"""

from __future__ import annotations

import pandas as pd

from .schema import FEATURES


def what_if_contributions(predictor, patient: dict) -> pd.DataFrame:
    """Return a DataFrame ``[feature, delta]`` sorted by ``|delta|`` (descending).

    ``delta`` is expressed in probability units (0-1), computed with one batched
    model call.
    """
    base = {f.name: patient[f.name] for f in FEATURES}
    rows = [dict(base)]
    for feature in FEATURES:
        variant = dict(base)
        variant[feature.name] = feature.default
        rows.append(variant)

    probabilities = predictor.predict_proba(pd.DataFrame(rows))
    baseline = float(probabilities[0])
    result = pd.DataFrame(
        {
            "feature": [f.name for f in FEATURES],
            "delta": baseline - probabilities[1:],
        }
    )
    order = result["delta"].abs().sort_values(ascending=False).index
    return result.loc[order].reset_index(drop=True)
