"""Model registry for Beijing Air Quality Forecasting System."""
from src.models.baseline import PersistenceBaseline
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm import LSTMModel
from src.models.cnn_lstm import CNNLSTMModel
from src.models.stcnn_gru import STCNNGRUModel
from src.models.ga_kelm import GAKELMModel
from src.models.hybrid_forecaster import HybridForecaster

__all__ = [
    "PersistenceBaseline",
    "RandomForestModel",
    "XGBoostModel",
    "LSTMModel",
    "CNNLSTMModel",
    "STCNNGRUModel",
    "GAKELMModel",
    "HybridForecaster",
    "MODEL_MAP",
]

MODEL_MAP = {
    "persistence": PersistenceBaseline,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lstm": LSTMModel,
    "cnn_lstm": CNNLSTMModel,
    "stcnn_gru": STCNNGRUModel,
    "ga_kelm": GAKELMModel,
}

