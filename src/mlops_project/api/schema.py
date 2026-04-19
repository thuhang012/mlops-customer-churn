from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Literal

YesNo = Literal["Yes", "No"]
Gender = Literal["Male", "Female"]
MultipleLinesType = Literal["Yes", "No", "No phone service"]
InternetServiceType = Literal["DSL", "Fiber optic", "No"]
InternetAddonType = Literal["Yes", "No", "No internet service"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodType = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

class CustomerInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    customerID: str

    gender: Gender
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0)

    PhoneService: YesNo
    MultipleLines: MultipleLinesType

    InternetService: InternetServiceType
    OnlineSecurity: InternetAddonType
    OnlineBackup: InternetAddonType
    DeviceProtection: InternetAddonType
    TechSupport: InternetAddonType
    StreamingTV: InternetAddonType
    StreamingMovies: InternetAddonType

    Contract: ContractType
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodType

    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    @field_validator("MonthlyCharges", "TotalCharges", mode="before")
    @classmethod
    def parse_float_fields(cls, v):
        if v in ("", " ", None):
            raise ValueError("Value is required")
        return float(v)

    @field_validator("customerID", mode="before")
    @classmethod
    def validate_customer_id(cls, v):
        if v in (None, "", " "):
            raise ValueError("customerID is required")
        return str(v).strip()

    @model_validator(mode="after")
    def validate_total_vs_monthly(self):
        if self.TotalCharges < self.MonthlyCharges:
            raise ValueError("TotalCharges must be greater than or equal to MonthlyCharges")
        return self

class PredictionOutput(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0.0, le=1.0)
    model_name: str
