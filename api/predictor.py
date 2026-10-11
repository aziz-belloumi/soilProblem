from pathlib import Path
from typing import Dict, List, Any
import joblib
import numpy as np
import pandas as pd

from api.schemas import (
    GrowthStage,
    IrrigationAction,
    IrrigationInput,
    PredictionResponse,
    BatchPredictionResponse,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Ordinal growth stage encoding mapping (0 to 4)
STAGE_ORDER: Dict[str, float] = {
    GrowthStage.SEEDLING.value: 0.0,
    GrowthStage.VEGETATIVE.value: 1.0,
    GrowthStage.FLOWERING.value: 2.0,
    GrowthStage.FRUITING.value: 3.0,
    GrowthStage.MATURITY.value: 4.0,
}

# Decision tier metadata and agronomic guidance lookup table
ADVISORY_CATALOG: Dict[int, Dict[str, Any]] = {
    0: {
        "action": IrrigationAction.NO_IRRIGATION,
        "urgency": "None / Adequate Moisture",
        "advisory": (
            "Soil moisture and root-zone water storage are currently optimal. "
            "No irrigation is needed at this time."
        ),
    },
    1: {
        "action": IrrigationAction.LOW,
        "urgency": "Low Priority",
        "advisory": (
            "Minor soil moisture deficit detected. Apply a light irrigation cycle "
            "to sustain vegetative health and avoid early stress onset."
        ),
    },
    2: {
        "action": IrrigationAction.MEDIUM,
        "urgency": "Moderate Priority",
        "advisory": (
            "Moderate water stress observed across field telemetry. Apply scheduled regular "
            "irrigation to maintain target root-zone moisture."
        ),
    },
    3: {
        "action": IrrigationAction.HIGH,
        "urgency": "Immediate Action Required",
        "advisory": (
            "Severe water stress or critical moisture depletion detected. Immediate high-volume "
            "irrigation required to prevent permanent crop wilting and yield loss."
        ),
    },
}


class IrrigationPredictor:
    """
    Inference orchestrator for Smart Irrigation Scheduling.
    
    Transforms 24 raw field observation attributes into the exact 37-dimensional
    engineered feature vector used to train the Random Forest Regressor:
      - 20 direct numeric features
      - 1 ordinal integer (Growth_Stage: 0 to 4)
      - 8 one-hot binary flags (Crop_Type)
      - 4 one-hot binary flags (Soil_Type)
      - 2 cyclical transformations of Day_of_Year (Day_sin, Day_cos)
      - 2 cyclical transformations of Time_of_Day (Time_sin, Time_cos)
      Total = 20 + 1 + 8 + 4 + 2 + 2 = 37 features
    
    Note on scaler.pkl:
      The scaler artifact (scaler.pkl) was fitted in the training notebook on X_train for
      distance/gradient-based models (e.g. MLP Regressor). Because Random Forest decision
      trees make partition splits along single dimensions and are strictly invariant to monotonic
      scaling, the Random Forest model was trained on and evaluates the unscaled 37-feature vector.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.feature_columns: List[str] = []
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        model_path = self.models_dir / "rf_regressor.pkl"
        scaler_path = self.models_dir / "scaler.pkl"
        features_path = self.models_dir / "feature_columns.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Feature columns specification not found at: {features_path}")

        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(features_path)

        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

    def is_ready(self) -> bool:
        """Verify model and feature column definitions are loaded."""
        return self.model is not None and len(self.feature_columns) == 37

    def preprocess_input(self, item: IrrigationInput) -> Dict[str, float]:
        """
        Convert 24 raw inputs into a dictionary matching the 37 model feature columns.
        """
        data = item.model_dump()

        # Initialize vector with zero defaults for all trained columns
        row: Dict[str, float] = {col: 0.0 for col in self.feature_columns}

        # 1. Direct continuous measurements (20 features)
        direct_numeric_keys = [
            "Soil_Moisture",
            "Soil_Temperature",
            "Soil_pH",
            "Soil_EC",
            "Ambient_Temperature",
            "Relative_Humidity",
            "Wind_Speed",
            "Solar_Radiation",
            "Rainfall_24h",
            "Rainfall_Forecast_24h",
            "Root_Zone_Depth",
            "Leaf_Area_Index",
            "Crop_Coefficient",
            "Field_Slope",
            "Water_Availability",
            "Previous_Irrigation_Amount",
            "Days_Since_Last_Irrigation",
            "Historical_Soil_Moisture",
            "Evaporation_Demand",
            "Water_Stress_Index",
        ]
        for key in direct_numeric_keys:
            val = data.get(key)
            if val is not None and key in row:
                row[key] = float(val)

        # 2. Ordinal encoding for Growth_Stage (1 feature)
        growth_stage_str = data["Growth_Stage"]
        if hasattr(growth_stage_str, "value"):
            growth_stage_str = growth_stage_str.value
        row["Growth_Stage"] = STAGE_ORDER.get(str(growth_stage_str), 0.0)

        # 3. One-hot encoding for Crop_Type (8 binary features)
        crop_type_str = data["Crop_Type"]
        if hasattr(crop_type_str, "value"):
            crop_type_str = crop_type_str.value
        crop_col = f"Crop_Type_{crop_type_str}"
        if crop_col in row:
            row[crop_col] = 1.0

        # 4. One-hot encoding for Soil_Type (4 binary features)
        soil_type_str = data["Soil_Type"]
        if hasattr(soil_type_str, "value"):
            soil_type_str = soil_type_str.value
        soil_col = f"Soil_Type_{soil_type_str}"
        if soil_col in row:
            row[soil_col] = 1.0

        # 5. Cyclical encodings for Day_of_Year (2 features)
        day = float(data["Day_of_Year"])
        row["Day_sin"] = float(np.sin(2 * np.pi * day / 365.0))
        row["Day_cos"] = float(np.cos(2 * np.pi * day / 365.0))

        # 6. Cyclical encodings for Time_of_Day (2 features)
        time = float(data["Time_of_Day"])
        row["Time_sin"] = float(np.sin(2 * np.pi * time / 24.0))
        row["Time_cos"] = float(np.cos(2 * np.pi * time / 24.0))

        return row

    def predict_single(self, item: IrrigationInput) -> PredictionResponse:
        """Execute single-observation prediction."""
        row_dict = self.preprocess_input(item)
        df = pd.DataFrame([row_dict])[self.feature_columns]

        raw_score = float(self.model.predict(df)[0])
        class_idx = int(np.clip(np.round(raw_score), 0, 3))
        catalog = ADVISORY_CATALOG[class_idx]

        return PredictionResponse(
            predicted_action=catalog["action"],
            predicted_class_index=class_idx,
            raw_continuous_score=round(raw_score, 4),
            urgency_level=catalog["urgency"],
            advisory=catalog["advisory"],
        )

    def predict_batch(self, items: List[IrrigationInput]) -> BatchPredictionResponse:
        """Execute multi-observation batch prediction."""
        rows = [self.preprocess_input(item) for item in items]
        df = pd.DataFrame(rows)[self.feature_columns]

        raw_scores = self.model.predict(df)
        predictions: List[PredictionResponse] = []

        for raw_val in raw_scores:
            val = float(raw_val)
            class_idx = int(np.clip(np.round(val), 0, 3))
            catalog = ADVISORY_CATALOG[class_idx]
            predictions.append(
                PredictionResponse(
                    predicted_action=catalog["action"],
                    predicted_class_index=class_idx,
                    raw_continuous_score=round(val, 4),
                    urgency_level=catalog["urgency"],
                    advisory=catalog["advisory"],
                )
            )

        return BatchPredictionResponse(
            total_count=len(predictions),
            predictions=predictions,
        )


predictor = IrrigationPredictor()
