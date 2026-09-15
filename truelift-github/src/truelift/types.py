from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CausalFrame:
    """Canonical table the whole engine speaks."""

    ids: np.ndarray
    X: np.ndarray
    feature_names: list[str]
    t: np.ndarray
    y: np.ndarray
    arm_names: list[str]
    split: np.ndarray
    exposure: np.ndarray | None = None
    true_tau: np.ndarray | None = None
    group: np.ndarray | None = None
    group_names: list[str] = field(default_factory=list)
    cost: np.ndarray | None = None
    dose: np.ndarray | None = None
    spend: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_arms(self) -> int:
        return len(self.arm_names)

    @property
    def n_treat(self) -> int:
        return max(self.n_arms - 1, 1)

    def mask(self, name: str) -> np.ndarray:
        return self.split == name

    def subset(self, mask: np.ndarray) -> "CausalFrame":
        def take(a):
            return None if a is None else a[mask]

        return CausalFrame(
            ids=self.ids[mask],
            X=self.X[mask],
            feature_names=self.feature_names,
            t=self.t[mask],
            y=self.y[mask],
            arm_names=self.arm_names,
            split=self.split[mask],
            exposure=take(self.exposure),
            true_tau=take(self.true_tau),
            group=take(self.group),
            group_names=self.group_names,
            cost=take(self.cost),
            dose=take(self.dose),
            spend=take(self.spend),
            meta=dict(self.meta),
        )


def encode_categoricals(df, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    import pandas as pd

    parts = []
    names = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            parts.append(df[c].astype(float).to_numpy()[:, None])
            names.append(c)
        else:
            dummies = pd.get_dummies(df[c], prefix=c, dummy_na=False)
            parts.append(dummies.to_numpy(dtype=float))
            names.extend(list(dummies.columns))
    X = np.hstack(parts) if parts else np.zeros((len(df), 1))
    return X, names
