from pydantic import BaseModel, Field

class CustomerInput(BaseModel):
    user_id: str
    age_group: int = Field(..., ge=0)
    gender: str
    country: str
    region: str
    subscription_plan: str
    monthly_fee: float = Field(..., ge=0)
    subscription_start_date: str
    subscription_end_date: str
    payment_method: str
    discount_applied: str
    title: str
    content_type: str
    genre: str
    language: str
    release_year: int = Field(..., ge=1900, le=2100) 
    device_type: str
    watch_time_minutes: int = Field(..., ge=0)
    session_count: int = Field(..., ge=0)
    completion_percentage: int = Field(..., ge=0, le=100)
    date_watched: str
    time_of_day: str
    rating: int = Field(..., ge=1, le=5)
    liked: str
    recommendation_source: str
    days_since_last_watch: int = Field(..., ge=0)
    avg_weekly_watch_time: int = Field(..., ge=0)
    content_diversity_score: float = Field(..., ge=0)

class PredictionOutput(BaseModel):
    churn_probability: float
    prediction: int
