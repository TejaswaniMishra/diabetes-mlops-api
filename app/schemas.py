"""Request and response models for the Diabetes Prediction API."""

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    """A single patient's medical measurements.

    These are the 8 features the model was trained on. Each field uses the
    same units and encoding as the Kaggle 'Diabetes Dataset'.
    """

    Pregnancies: int = Field(..., ge=0, le=20, description="Number of times pregnant")
    Glucose: float = Field(..., ge=0, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., ge=0, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, description="Diabetes pedigree function")
    Age: float = Field(..., ge=0, le=120, description="Age in years")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Pregnancies": 6,
                    "Glucose": 148.0,
                    "BloodPressure": 72.0,
                    "SkinThickness": 35.0,
                    "Insulin": 0.0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 50.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """The model's prediction for one patient."""

    prediction: int = Field(..., description="0 = no diabetes, 1 = diabetes")
    label: str = Field(..., description="Human readable prediction")
    probability: float = Field(..., description="Model confidence in the predicted class")