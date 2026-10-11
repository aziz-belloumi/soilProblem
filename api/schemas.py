from enum import Enum
from typing import List
from pydantic import BaseModel, Field, ConfigDict


class CropType(str, Enum):
    """Supported agricultural crop classifications."""
    COTTON = "Cotton"
    GROUNDNUT = "Groundnut"
    MAIZE = "Maize"
    RICE = "Rice"
    SOYBEAN = "Soybean"
    SUGARCANE = "Sugarcane"
    TOMATO = "Tomato"
    WHEAT = "Wheat"


class SoilType(str, Enum):
    """Supported soil textural classifications."""
    CLAY = "Clay"
    LOAMY = "Loamy"
    SANDY = "Sandy"
    SILTY = "Silty"


class GrowthStage(str, Enum):
    """Phenological growth stages of cultivated crops."""
    SEEDLING = "Seedling"
    VEGETATIVE = "Vegetative"
    FLOWERING = "Flowering"
    FRUITING = "Fruiting"
    MATURITY = "Maturity"


class IrrigationAction(str, Enum):
    """Discrete irrigation management decision tiers."""
    NO_IRRIGATION = "No_Irrigation"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class IrrigationInput(BaseModel):
    """
    Raw sensor & field telemetry input schema (24 input fields).
    
    The API's preprocessing layer automatically transforms these 24 raw fields
    into the 37-dimensional engineered feature vector expected by the model
    (ordinal encoding, one-hot encodings, and cyclical sin/cos transformations).
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Crop_Type": "Wheat",
                "Growth_Stage": "Flowering",
                "Soil_Type": "Clay",
                "Day_of_Year": 306,
                "Time_of_Day": 16,
                "Soil_Moisture": 34.65,
                "Soil_Temperature": 28.48,
                "Soil_pH": 7.22,
                "Soil_EC": 1.63,
                "Ambient_Temperature": 25.16,
                "Relative_Humidity": 75.88,
                "Wind_Speed": 2.06,
                "Solar_Radiation": 617.89,
                "Rainfall_24h": 1.66,
                "Rainfall_Forecast_24h": 11.78,
                "Root_Zone_Depth": 35.23,
                "Leaf_Area_Index": 4.19,
                "Crop_Coefficient": 1.15,
                "Field_Slope": 4.86,
                "Water_Availability": 65.38,
                "Previous_Irrigation_Amount": 10.5,
                "Days_Since_Last_Irrigation": 7,
                "Historical_Soil_Moisture": 24.66,
                "Evaporation_Demand": 52.41,
                "Water_Stress_Index": 76.23
            }
        }
    )

    # Categorical Inputs
    Crop_Type: CropType = Field(..., description="Crop variety being cultivated")
    Growth_Stage: GrowthStage = Field(..., description="Crop phenological growth stage (Seedling, Vegetative, Flowering, Fruiting, Maturity)")
    Soil_Type: SoilType = Field(..., description="Soil texture category (Clay, Loamy, Sandy, Silty)")

    # Temporal & Cyclical Inputs
    Day_of_Year: int = Field(..., ge=1, le=366, description="Julian calendar day (1 to 366)")
    Time_of_Day: int = Field(..., ge=0, le=23, description="Hour of the day in 24-hour format (0 to 23)")

    # Soil Telemetry
    Soil_Moisture: float = Field(..., ge=0.0, le=100.0, description="Volumetric soil water content percentage (0-100%)")
    Soil_Temperature: float = Field(..., ge=-20.0, le=70.0, description="Soil temperature in degrees Celsius")
    Soil_pH: float = Field(..., ge=0.0, le=14.0, description="Soil acidity/alkalinity level (0-14)")
    Soil_EC: float = Field(..., ge=0.0, description="Soil electrical conductivity (dS/m)")
    Historical_Soil_Moisture: float = Field(..., ge=0.0, le=100.0, description="Historical rolling baseline soil moisture percentage")

    # Meteorological & Atmospheric Observations
    Ambient_Temperature: float = Field(..., ge=-30.0, le=60.0, description="Ambient air temperature in degrees Celsius")
    Relative_Humidity: float = Field(..., ge=0.0, le=100.0, description="Ambient relative humidity percentage (0-100%)")
    Wind_Speed: float = Field(..., ge=0.0, description="Wind speed at field surface (m/s)")
    Solar_Radiation: float = Field(..., ge=0.0, description="Incoming solar irradiance (W/m²)")
    Rainfall_24h: float = Field(..., ge=0.0, description="Precipitation accumulated over previous 24 hours (mm)")
    Rainfall_Forecast_24h: float = Field(..., ge=0.0, description="Forecasted precipitation over next 24 hours (mm)")
    Evaporation_Demand: float = Field(..., ge=0.0, description="Atmospheric evaporative demand / potential evapotranspiration (mm)")

    # Agronomic & Topographical Attributes
    Root_Zone_Depth: float = Field(..., ge=0.0, description="Effective crop root zone depth (cm)")
    Leaf_Area_Index: float = Field(..., ge=0.0, description="Canopy leaf area index (LAI, dimensionless)")
    Crop_Coefficient: float = Field(..., ge=0.0, description="Single crop coefficient Kc")
    Field_Slope: float = Field(..., ge=0.0, le=100.0, description="Field surface gradient slope percentage")
    Water_Availability: float = Field(..., ge=0.0, le=100.0, description="Irrigation reservoir / aquifer availability score (0-100%)")
    Previous_Irrigation_Amount: float = Field(..., ge=0.0, description="Volume applied during last irrigation event (mm)")
    Days_Since_Last_Irrigation: int = Field(..., ge=0, description="Number of complete days since last irrigation event")
    Water_Stress_Index: float = Field(..., ge=0.0, le=100.0, description="Crop Water Stress Index CWSI (0-100 scale)")


class IrrigationBatchInput(BaseModel):
    """Batch prediction payload supporting multiple field monitoring locations."""
    items: List[IrrigationInput] = Field(..., min_length=1, description="List of 24-feature field observations")


class PredictionResponse(BaseModel):
    """Inference result with classification, regression score, and agronomic guidance."""
    predicted_action: IrrigationAction = Field(..., description="Actionable irrigation tier")
    predicted_class_index: int = Field(..., ge=0, le=3, description="Ordinal class index (0: No_Irrigation, 1: Low, 2: Medium, 3: High)")
    raw_continuous_score: float = Field(..., description="Continuous ordinal regression score from Random Forest Regressor")
    urgency_level: str = Field(..., description="Operational urgency classification")
    advisory: str = Field(..., description="Agronomic guidance message for field operators")


class BatchPredictionResponse(BaseModel):
    """Batch inference response wrapper."""
    total_count: int = Field(..., description="Total records evaluated")
    predictions: List[PredictionResponse] = Field(..., description="Array of prediction outcomes")


class HealthResponse(BaseModel):
    """System health check and model loading verification."""
    status: str
    model_loaded: bool
    total_features: int
    model_type: str
    scaling_applied: bool = Field(False, description="Whether input features are scaled (False for Random Forest)")
