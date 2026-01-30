"""
FastAPI application for Breast Cancer Classification.

This module defines the REST API endpoints for making predictions
using the trained Random Forest model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import uvicorn

from src.data import get_feature_names, get_target_names, get_sample_data, get_dataset_info
from src.predict import get_predictor, BreastCancerPredictor


# =============================================================================
# Pydantic Models for Request/Response Validation
# =============================================================================

class TumorFeatures(BaseModel):
    """
    Input features for breast tumor classification.
    
    All 30 features are required for prediction. Features describe
    characteristics of cell nuclei present in the digitized image
    of a fine needle aspirate (FNA) of a breast mass.
    """
    # Mean values
    mean_radius: float = Field(..., description="Mean of distances from center to points on perimeter")
    mean_texture: float = Field(..., description="Standard deviation of gray-scale values")
    mean_perimeter: float = Field(..., description="Mean perimeter of cell nuclei")
    mean_area: float = Field(..., description="Mean area of cell nuclei")
    mean_smoothness: float = Field(..., description="Mean local variation in radius lengths")
    mean_compactness: float = Field(..., description="Mean of perimeter^2 / area - 1.0")
    mean_concavity: float = Field(..., description="Mean severity of concave portions")
    mean_concave_points: float = Field(..., description="Mean number of concave portions")
    mean_symmetry: float = Field(..., description="Mean symmetry of cell nuclei")
    mean_fractal_dimension: float = Field(..., description="Mean coastline approximation - 1")
    
    # Standard error values
    se_radius: float = Field(..., description="Standard error of radius")
    se_texture: float = Field(..., description="Standard error of texture")
    se_perimeter: float = Field(..., description="Standard error of perimeter")
    se_area: float = Field(..., description="Standard error of area")
    se_smoothness: float = Field(..., description="Standard error of smoothness")
    se_compactness: float = Field(..., description="Standard error of compactness")
    se_concavity: float = Field(..., description="Standard error of concavity")
    se_concave_points: float = Field(..., description="Standard error of concave points")
    se_symmetry: float = Field(..., description="Standard error of symmetry")
    se_fractal_dimension: float = Field(..., description="Standard error of fractal dimension")
    
    # Worst (largest) values
    worst_radius: float = Field(..., description="Worst (largest mean) radius")
    worst_texture: float = Field(..., description="Worst texture")
    worst_perimeter: float = Field(..., description="Worst perimeter")
    worst_area: float = Field(..., description="Worst area")
    worst_smoothness: float = Field(..., description="Worst smoothness")
    worst_compactness: float = Field(..., description="Worst compactness")
    worst_concavity: float = Field(..., description="Worst concavity")
    worst_concave_points: float = Field(..., description="Worst concave points")
    worst_symmetry: float = Field(..., description="Worst symmetry")
    worst_fractal_dimension: float = Field(..., description="Worst fractal dimension")
    
    class Config:
        json_schema_extra = {
            "example": {
                "mean_radius": 17.99,
                "mean_texture": 10.38,
                "mean_perimeter": 122.8,
                "mean_area": 1001.0,
                "mean_smoothness": 0.1184,
                "mean_compactness": 0.2776,
                "mean_concavity": 0.3001,
                "mean_concave_points": 0.1471,
                "mean_symmetry": 0.2419,
                "mean_fractal_dimension": 0.07871,
                "se_radius": 1.095,
                "se_texture": 0.9053,
                "se_perimeter": 8.589,
                "se_area": 153.4,
                "se_smoothness": 0.006399,
                "se_compactness": 0.04904,
                "se_concavity": 0.05373,
                "se_concave_points": 0.01587,
                "se_symmetry": 0.03003,
                "se_fractal_dimension": 0.006193,
                "worst_radius": 25.38,
                "worst_texture": 17.33,
                "worst_perimeter": 184.6,
                "worst_area": 2019.0,
                "worst_smoothness": 0.1622,
                "worst_compactness": 0.6656,
                "worst_concavity": 0.7119,
                "worst_concave_points": 0.2654,
                "worst_symmetry": 0.4601,
                "worst_fractal_dimension": 0.1189
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: str = Field(..., description="Predicted class: 'Malignant' or 'Benign'")
    prediction_code: int = Field(..., description="Numeric class: 0 (Malignant) or 1 (Benign)")
    probability: Dict[str, float] = Field(..., description="Class probabilities")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    message: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""
    model_type: str
    n_estimators: Optional[int]
    max_depth: Optional[int]
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    n_features: int
    classes: Dict[int, str]
    top_10_feature_importances: Dict[str, float]


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Breast Cancer Classification API",
    description="""
## Breast Cancer Wisconsin Classification API

This API predicts whether a breast tumor is **Malignant** (cancerous) or **Benign** (non-cancerous) 
based on features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

### Model Information
- **Algorithm**: Random Forest Classifier
- **Dataset**: Breast Cancer Wisconsin (Diagnostic)
- **Features**: 30 numeric features describing cell nuclei characteristics
- **Classes**: Malignant (0), Benign (1)

### Endpoints
- `GET /` - Health check and welcome message
- `GET /features` - Get list of required feature names
- `POST /predict` - Make a prediction
- `GET /model/info` - Get model metadata and performance metrics
- `GET /sample` - Get sample input data for testing

### Author
Created as part of MLOps coursework at Northeastern University.
    """,
    version="1.0.0",
    contact={
        "name": "MLOps Lab Assignment",
        "url": "https://github.com/raminmohammadi/MLOps",
    },
    license_info={
        "name": "MIT",
    }
)


# Global predictor (lazy loaded)
predictor: Optional[BreastCancerPredictor] = None


def get_model() -> BreastCancerPredictor:
    """Get or load the predictor model."""
    global predictor
    if predictor is None:
        try:
            predictor = get_predictor()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=str(e)
            )
    return predictor


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """
    Health check endpoint.
    
    Returns the API status and whether the model is loaded.
    """
    model_loaded = False
    try:
        get_model()
        model_loaded = True
    except:
        pass
    
    return HealthResponse(
        status="healthy",
        message="Welcome to the Breast Cancer Classification API! "
                "Visit /docs for interactive API documentation.",
        model_loaded=model_loaded
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Detailed health check endpoint.
    
    Verifies that the model is loaded and ready for predictions.
    """
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            message="Model is loaded and ready for predictions.",
            model_loaded=True
        )
    except HTTPException as e:
        return HealthResponse(
            status="unhealthy",
            message=e.detail,
            model_loaded=False
        )


@app.get("/features", tags=["Information"])
async def get_features() -> Dict[str, Any]:
    """
    Get the list of required feature names.
    
    Returns all 30 feature names that must be provided for prediction.
    """
    feature_names = get_feature_names()
    return {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "feature_groups": {
            "mean": feature_names[:10],
            "standard_error": feature_names[10:20],
            "worst": feature_names[20:30]
        }
    }


@app.get("/dataset", tags=["Information"])
async def get_dataset() -> Dict[str, Any]:
    """
    Get information about the training dataset.
    
    Returns metadata about the Breast Cancer Wisconsin dataset.
    """
    return get_dataset_info()


@app.get("/sample", tags=["Information"])
async def get_sample() -> Dict[str, Any]:
    """
    Get sample input data for testing.
    
    Returns a complete example of input features that can be used
    with the /predict endpoint.
    """
    sample = get_sample_data()
    return {
        "description": "Sample tumor features (first record from dataset - Malignant case)",
        "features": sample
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: TumorFeatures) -> PredictionResponse:
    """
    Make a breast cancer prediction.
    
    Accepts tumor characteristics and returns whether the tumor is
    predicted to be Malignant or Benign, along with confidence probabilities.
    
    **Input**: All 30 tumor features (see /features for the full list)
    
    **Output**: 
    - prediction: "Malignant" or "Benign"
    - prediction_code: 0 (Malignant) or 1 (Benign)
    - probability: Confidence scores for each class
    """
    model = get_model()
    
    # Convert Pydantic model to dict
    features_dict = features.model_dump()
    
    try:
        result = model.predict(features_dict)
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info() -> ModelInfoResponse:
    """
    Get information about the trained model.
    
    Returns model metadata including:
    - Model type and hyperparameters
    - Performance metrics (accuracy, precision, recall, F1)
    - Top 10 most important features
    """
    model = get_model()
    info = model.get_model_info()
    return ModelInfoResponse(**info)


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING BREAST CANCER CLASSIFICATION API")
    print("="*60)
    print("\nAccess the API at:")
    print("  - API Root:    http://localhost:8000")
    print("  - Swagger UI:  http://localhost:8000/docs")
    print("  - ReDoc:       http://localhost:8000/redoc")
    print("\nPress CTRL+C to stop the server.\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
