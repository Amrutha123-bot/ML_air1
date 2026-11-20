"""Model registry for AeroSense AI Delhi Forecasting System."""
from src.models.baseline import PersistenceBaseline
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm import LSTMModel
from src.models.cnn_lstm import CNNLSTMModel

__all__ = [
    "PersistenceBaseline",
    "RandomForestModel",
    "XGBoostModel",
    "LSTMModel",
    "CNNLSTMModel",
    "MODEL_MAP",
]

MODEL_MAP = {
    "persistence": PersistenceBaseline,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lstm": LSTMModel,
    "cnn_lstm": CNNLSTMModel,
}
