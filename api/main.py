from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.predictor import predictor
from api.schemas import (
    IrrigationInput,
    IrrigationBatchInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify model is properly loaded on startup
    if not predictor.is_ready():
        raise RuntimeError("Failed to initialize model artifacts or feature column definitions.")
    yield


app = FastAPI(
    title="Smart Irrigation Scheduling API",
    description=(
        "Production-grade decision-support API for agricultural irrigation scheduling. "
        "Transforms 24 raw field and weather telemetry parameters into 37 engineered features "
        "to predict actionable irrigation requirements (No_Irrigation, Low, Medium, High) "
        "via an ordinal Random Forest Regressor."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web and dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Clean JSON response on invalid schema/inputs."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": errors,
        },
    )


@app.get("/", tags=["General"])
async def root():
    """Service metadata and routing directory."""
    return {
        "service": "Smart Irrigation Scheduling API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "features_metadata": "GET /features",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Verify inference engine health and model status."""
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts or feature definitions are not properly loaded",
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        total_features=len(predictor.feature_columns),
        model_type=predictor.model.__class__.__name__,
        scaling_applied=False,
    )


@app.get("/features", response_model=List[str], tags=["Model Info"])
async def get_feature_columns():
    """Retrieve ordered list of the 37 engineered feature columns expected by the model."""
    return predictor.feature_columns


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_irrigation(input_data: IrrigationInput):
    """
    Generate an irrigation decision for a single field telemetry observation.
    
    Accepts 24 raw features (categorical strings, Julian calendar days, and sensor measurements)
    and internally transforms them into the 37-feature vector before executing inference.
    """
    try:
        return predictor.predict_single(input_data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failure: {str(exc)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_irrigation_batch(batch_data: IrrigationBatchInput):
    """
    Generate irrigation recommendations for multiple field monitoring stations in batch.
    """
    try:
        return predictor.predict_batch(batch_data.items)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference pipeline failure: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
