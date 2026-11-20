"""FastAPI application entry point for Beijing Air Quality Forecasting System."""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.config.settings import (
    API_PREFIX, API_VERSION, CORS_ORIGINS, API_HOST, API_PORT
)
from src.api.routes import health, stations, forecasts, warnings, metrics

app = FastAPI(
    title="Beijing Multi-Station Air Quality Forecasting & Early-Warning API",
    description=(
        "Production-ready REST API for multi-horizon PM2.5 forecasting (1h to 72h ahead), "
        "spatiotemporal air pollution modeling, and regulatory early-warning alerts across "
        "12 Beijing monitoring stations based on the UCI Beijing Multi-Site dataset."
    ),
    version=API_VERSION,
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


@app.get("/", include_in_schema=False)
def root():
    return {
        "system": "Beijing Air Quality Forecasting & Early-Warning System",
        "version": API_VERSION,
        "docs_url": "/docs",
        "api_v1_base": API_PREFIX,
        "status": "online"
    }


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=False)
