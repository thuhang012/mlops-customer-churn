from pydantic import BaseModel
from typing import List

class CustomerInput(BaseModel):
    user_id: str
    age_group: int
    gender: str
    country: str
    region: str
    subscription_plan: str
    monthly_fee: float
    subscription_start_date: str
    subscription_end_date: str
    payment_method: str
    discount_applied: str
    title: str
    content_type: str
    genre: str
    language: str
    release_year: int
    device_type: str
    watch_time_minutes: int
    session_count: int
    completion_percentage: int
    date_watched: str
    time_of_day: str
    rating: int
    liked: str
    recommendation_source: str
    days_since_last_watch: int
    avg_weekly_watch_time: int
    content_diversity_score: float


class PredictionOutput(BaseModel):
    churn_probability: float
    prediction: int