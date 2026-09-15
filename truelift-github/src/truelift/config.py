from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRUELIFT_", extra="ignore")

    dataset: str = "synthetic"
    outcome: str = "visit"
    treatment: str = "auto"
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.15
    host: str = "127.0.0.1"
    port: int = 8000
    web_port: int = 3000
    web_origin: str = "http://127.0.0.1:3000"
    serve_web: bool = True
    n_estimators: int = 200
    neural_epochs: int = 40
    neural_hidden: int = 64
    conformal_alpha: float = 0.1
    bootstrap: int = 200
    criteo_sample: int = 200_000
    revenue_per_outcome: float = 12.0
    contact_cost: float = 1.0
    treatment_col: str = "treatment"
    csv_max_rows: int = 80_000
    fairness_slack: float = 0.05


DEFAULT = Settings()
