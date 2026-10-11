import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.predictor import predictor
from api.schemas import IrrigationInput, IrrigationAction

client = TestClient(app)
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "api" / "sample_input.json"


@pytest.fixture
def sample_payload():
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Smart Irrigation Scheduling API" in data["service"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["total_features"] == 37
    assert data["model_type"] == "RandomForestRegressor"
    assert data["scaling_applied"] is False


def test_features_endpoint():
    response = client.get("/features")
    assert response.status_code == 200
    features = response.json()
    assert isinstance(features, list)
    assert len(features) == 37
    # Assert presence of key transformed features
    assert "Crop_Type_Wheat" in features
    assert "Soil_Type_Clay" in features
    assert "Growth_Stage" in features
    assert "Day_sin" in features
    assert "Day_cos" in features
    assert "Time_sin" in features
    assert "Time_cos" in features


def test_predictor_preprocessing_logic(sample_payload):
    """Verify 24 raw inputs correctly transform to 37 model inputs."""
    parsed_input = IrrigationInput(**sample_payload)
    row = predictor.preprocess_input(parsed_input)

    assert len(row) == 37
    # Flowering maps to 2.0
    assert row["Growth_Stage"] == 2.0
    # Wheat flag is 1.0, other crops are 0.0
    assert row["Crop_Type_Wheat"] == 1.0
    assert row["Crop_Type_Cotton"] == 0.0
    assert row["Crop_Type_Rice"] == 0.0
    # Clay flag is 1.0, other soils are 0.0
    assert row["Soil_Type_Clay"] == 1.0
    assert row["Soil_Type_Sandy"] == 0.0
    # Continuous features preserved
    assert row["Soil_Moisture"] == 34.65
    assert row["Days_Since_Last_Irrigation"] == 7
    # Cyclical sin/cos within bounds [-1, 1]
    assert -1.0 <= row["Day_sin"] <= 1.0
    assert -1.0 <= row["Day_cos"] <= 1.0
    assert -1.0 <= row["Time_sin"] <= 1.0
    assert -1.0 <= row["Time_cos"] <= 1.0


def test_predict_single_success(sample_payload):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_action"] in [
        IrrigationAction.NO_IRRIGATION.value,
        IrrigationAction.LOW.value,
        IrrigationAction.MEDIUM.value,
        IrrigationAction.HIGH.value,
    ]
    assert 0 <= data["predicted_class_index"] <= 3
    assert isinstance(data["raw_continuous_score"], float)
    assert len(data["urgency_level"]) > 0
    assert len(data["advisory"]) > 0


def test_predict_batch_success(sample_payload):
    batch_payload = {"items": [sample_payload, sample_payload]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert 0 <= pred["predicted_class_index"] <= 3


def test_validation_error_invalid_crop(sample_payload):
    bad_payload = dict(sample_payload)
    bad_payload["Crop_Type"] = "Avocado"  # Unsupported crop
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


def test_validation_error_invalid_growth_stage(sample_payload):
    bad_payload = dict(sample_payload)
    bad_payload["Growth_Stage"] = "Dormant"  # Unsupported stage
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_validation_error_negative_time_of_day(sample_payload):
    bad_payload = dict(sample_payload)
    bad_payload["Time_of_Day"] = 25  # Beyond 24-hour cycle
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_validation_error_missing_field(sample_payload):
    bad_payload = dict(sample_payload)
    del bad_payload["Soil_Moisture"]  # Missing required field
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
