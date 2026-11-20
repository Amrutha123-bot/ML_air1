"""FastAPI application entry point for AeroSense AI Delhi Air Quality System."""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import (
    API_PREFIX, API_VERSION, CORS_ORIGINS, API_HOST, API_PORT
)
from src.api.routes import (
    health,
    stations,
    forecasts,
    warnings,
    metrics,
    analytics,
)

app = FastAPI(
    title="AeroSense AI — Delhi Air Quality Forecasting & Early-Warning API",
    description=(
        "Production-ready REST API for Delhi National Capital Region (Anand Vihar Hub), "
        "providing multi-horizon PM2.5 forecasting (1h to 72h ahead) using trained ML models "
        "(Random Forest Champion, XGBoost, LSTM, GRU, CNN-LSTM), real evaluation metrics, "
        "and CPCB regulatory early-warning alerts."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in CORS_ORIGINS else CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 route modules
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(stations.router, prefix=API_PREFIX)
app.include_router(forecasts.router, prefix=API_PREFIX)
app.include_router(warnings.router, prefix=API_PREFIX)
app.include_router(metrics.router, prefix=API_PREFIX)
app.include_router(analytics.router, prefix=API_PREFIX)


@app.get("/", include_in_schema=False)
def root():
    return {
        "system": "AeroSense AI Delhi Air Quality Forecasting & Early-Warning System",
        "version": "2.1.0",
        "docs_url": "/docs",
        "api_v1_base": API_PREFIX,
        "status": "online",
        "scope": "Delhi NCR Monitoring Network (DEL_01 Anand Vihar Primary Hub)",
        "dataset": "Delhi CPCB-Calibrated Benchmark Dataset (35,064 records)",
    }


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=False)
