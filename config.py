"""FraudGraph application configuration — Pydantic V2 / pydantic-settings V2."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Data
    data_dir: str = Field(default="./data/raw", alias="DATA_DIR")
    kaggle_username: str = Field(default="", alias="KAGGLE_USERNAME")
    kaggle_key: str = Field(default="", alias="KAGGLE_KEY")

    # Model
    fraud_threshold: float = Field(default=0.5, alias="FRAUD_THRESHOLD")
    tabular_model_weight: float = Field(default=0.6, alias="TABULAR_MODEL_WEIGHT")

    # MLflow
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db", alias="MLFLOW_TRACKING_URI")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8001, alias="API_PORT")

    # Drift monitoring
    drift_window_size: int = Field(default=1000, alias="DRIFT_WINDOW_SIZE")
    drift_threshold: float = Field(default=0.15, alias="DRIFT_THRESHOLD")

    # Streaming
    simulator_tps: int = Field(default=10, alias="SIMULATOR_TPS")


settings = Settings()
