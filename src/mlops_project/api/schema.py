from pydantic import BaseModel


class CustomerInput(BaseModel):
    tenure: int
    monthly_charges: float
    contract_type: str


class PredictionOutput(BaseModel):
    churn_probability: float
